"""Two GPU diagnostics that decide what the remaining training runs should be.

1. Cross-attention diagonality. If the motion->audio attention is near-uniform,
   the model is effectively averaging the whole track and the audio carries no
   timing information, which would justify replacing cross-attention with
   frame-aligned residual addition. If it is peaked near the diagonal, the audio
   path already delivers aligned features and the fusion rewrite is not the fix.

2. Guidance x NFE grid. Quality peaked at 2-5 ODE steps and degraded with more,
   but every row used guidance 2.0 -- so accumulated classifier-free-guidance
   overshoot is an untested alternative explanation to anything intrinsic. This
   separates the two and quantifies how much of the jitter is guidance.
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
import motion
from dataset import canonicalise
from evaluate import beat_alignment, endpoint_cm, foot_skate, hfr, motion_beats
from generate_samples import build_item, collate, sample, to_joints
from model import RhythmFlow
from smpl_fk import load_skeleton

FPS = 30


def attach_attention_probe(model):
    """Force the cross-attention to return its weights, and record them."""
    store = []

    def wrap(mha):
        orig = mha.forward

        def fwd(q, k, v, **kw):
            kw["need_weights"] = True
            kw["average_attn_weights"] = True
            out, w = orig(q, k, v, **kw)
            if w is not None:
                store.append(w.detach().float().cpu())
            return out, w
        mha.forward = fwd

    for layer in model.net.layers:
        wrap(layer.multihead_attn)
    return store


def diagonality(w, valid, band=2):
    """Share of attention mass within +/-band frames of the diagonal."""
    B, T, S = w.shape
    i = torch.arange(T)[:, None]
    j = torch.arange(S)[None, :]
    near = (i - j).abs() <= band
    out = []
    for b in range(B):
        n = int(valid[b].sum().item())
        if n < 3 * band:
            continue
        sub = w[b, :n, :n]
        sub = sub / sub.sum(-1, keepdim=True).clamp(min=1e-9)
        out.append(float(sub[near[:n, :n]].sum() / n))
    uniform = (2 * band + 1) / max(1, int(valid[0].sum().item()))
    return float(np.mean(out)) if out else float("nan"), uniform


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--smpl-model", required=True)
    p.add_argument("--clip-len", type=int, default=150)
    p.add_argument("--per-seq", type=int, default=2)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = np.load(args.stats)
    stats = {"mean": st["mean"], "std": st["std"], "style_q": st["style_q"]}
    stats_t = (torch.tensor(st["mean"], device=dev), torch.tensor(st["std"], device=dev))
    rest_np, par_np = load_skeleton(args.smpl_model)
    rest = torch.tensor(rest_np, dtype=torch.float32, device=dev)
    parents = torch.tensor(par_np, dtype=torch.long, device=dev)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = RhythmFlow(d_model=ck["args"]["d_model"], nlayers=ck["args"]["nlayers"],
                       max_len=ck["args"]["max_len"],
                       n_style=ck["ema"]["null_style"].shape[0]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    print(f"loaded step {ck['step']} on {dev}", flush=True)

    # ---- build the eval set ------------------------------------------------
    names = [l.strip() for l in open(args.split) if l.strip()]
    rng = np.random.default_rng(0)
    items, gts, audios, Ns = [], [], [], []
    for n in names:
        pth = os.path.join(args.seq_dir, n + ".npz")
        if not os.path.exists(pth):
            continue
        d = dict(np.load(pth, allow_pickle=True))
        T = len(d["root"])
        a = np.load(os.path.join(args.audio_dir, str(d["music"]) + ".npy"))
        for _ in range(args.per_seq):
            N = min(args.clip_len, T)
            s = int(rng.integers(0, max(1, T - N + 1)))
            _, _, j = canonicalise(d["rot6d"][s:s + N], d["root"][s:s + N],
                                   d["joints"][s:s + N])
            stl = kinematics.to_percentile(kinematics.descriptors(j), stats["style_q"])
            it, _ = build_item(d, s, N, a, stl, stats)
            items.append(it); audios.append(it["audio"][:N]); Ns.append(N)
    b = collate(items, dev)
    gt_joints = to_joints(b["x"], stats_t, rest, parents)
    print(f"eval clips: {len(items)}", flush=True)

    # ---- 1. attention diagonality -----------------------------------------
    store = attach_attention_probe(model)
    with torch.no_grad():
        t = torch.full((b["x"].shape[0],), 0.5, device=dev)
        obs = b["x"] * b["keyframe"][..., None]
        model(torch.randn_like(b["x"]), t, obs, b["keyframe"],
              b["audio"], b["style"], b["valid"])
    print("\n=== 1. cross-attention diagonality (motion frame -> audio frames) ===")
    print(f"  {'layer':<8} {'mass within +/-2 frames':>26} {'vs uniform':>12} {'ratio':>8}")
    ratios = []
    for i, w in enumerate(store):
        d_, u_ = diagonality(w, b["valid"])
        ratios.append(d_ / u_)
        print(f"  {i:<8} {d_:>26.4f} {u_:>12.4f} {d_/u_:>8.2f}x")
    r = float(np.mean(ratios))
    print(f"\n  mean ratio over layers: {r:.2f}x uniform")
    print("  " + ("attention is essentially UNIFORM -- the audio contributes no timing\n"
                  "  information; replacing cross-attention with frame-aligned addition is\n"
                  "  the indicated fix." if r < 1.5 else
                  "attention is PEAKED near the diagonal -- the audio path already delivers\n"
                  "  temporally aligned features, so a fusion rewrite is unlikely to be the fix."))

    # ---- 2. guidance x NFE grid -------------------------------------------
    print("\n=== 2. guidance x NFE grid ===")
    print(f"  {'guid':>5} {'NFE':>5} {'fwd':>5} {'endpt cm':>10} {'HFR %':>8} "
          f"{'skate':>8} {'BAS':>7} {'beats/s':>8}")
    e, h, s = (np.mean([endpoint_cm(gt_joints[k][:Ns[k]], gt_joints[k][:Ns[k]], Ns[k])
                        for k in range(len(Ns))]),
               np.mean([100 * hfr(gt_joints[k][:Ns[k]]) for k in range(len(Ns))]),
               np.mean([foot_skate(gt_joints[k][:Ns[k]]) for k in range(len(Ns))]))
    bg = np.nanmean([beat_alignment(gt_joints[k][:Ns[k]], audios[k]) for k in range(len(Ns))])
    bps = np.mean([len(motion_beats(gt_joints[k][:Ns[k]])) / (Ns[k] / FPS)
                   for k in range(len(Ns))])
    print(f"  {'GT':>5} {'-':>5} {'-':>5} {e:>10.2f} {h:>8.3f} {s:>8.3f} {bg:>7.3f} {bps:>8.2f}")

    for g in [1.0, 1.5, 2.0, 3.0]:
        for nfe in [2, 5, 10, 50]:
            x = sample(model, b, steps=nfe, guidance=g, seed=0)
            j = to_joints(x, stats_t, rest, parents)
            row = [np.mean([endpoint_cm(j[k][:Ns[k]], gt_joints[k][:Ns[k]], Ns[k])
                            for k in range(len(Ns))]),
                   np.mean([100 * hfr(j[k][:Ns[k]]) for k in range(len(Ns))]),
                   np.mean([foot_skate(j[k][:Ns[k]]) for k in range(len(Ns))]),
                   np.nanmean([beat_alignment(j[k][:Ns[k]], audios[k])
                               for k in range(len(Ns))]),
                   np.mean([len(motion_beats(j[k][:Ns[k]])) / (Ns[k] / FPS)
                            for k in range(len(Ns))])]
            fwd = nfe if g == 1.0 else 2 * nfe
            print(f"  {g:>5.1f} {nfe:>5} {fwd:>5} {row[0]:>10.2f} {row[1]:>8.3f} "
                  f"{row[2]:>8.3f} {row[3]:>7.3f} {row[4]:>8.2f}", flush=True)
    print("\n  fwd = network forward passes (guidance 1.0 needs one per step, "
          "higher guidance needs two)")


if __name__ == "__main__":
    main()
