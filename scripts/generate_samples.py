"""Generate qualitative samples from the held-out test split.

Three groups, written to a single npz for the renderer:

  recon   held-out sequences at several gap lengths, generated vs ground truth
  music   one fixed pair of endpoints, driven by several unseen music tracks
          (the project's central claim: same constraints, different motion)
  style   one fixed pair of endpoints and track, sweeping the kinematic dial

The test split shares no sequence and no music track with training, so both the
choreography and the audio are new to the model.
"""
import argparse, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion
import kinematics
from dataset import canonicalise, style_value
from model import RhythmFlow
from smpl_fk import load_skeleton

MAXLEN = 150


def build_item(seq, start, N, audio_full, style, stats, audio_dim=35):
    """Assemble one conditioning item; returns normalised tensors padded to MAXLEN."""
    sl = slice(start, start + N)
    rot6d, root, joints = canonicalise(seq["rot6d"][sl], seq["root"][sl], seq["joints"][sl])
    x = np.concatenate([seq["contacts"][sl], root, rot6d], axis=1).astype(np.float32)
    x = (x - stats["mean"]) / stats["std"]

    a = np.zeros((N, audio_dim), np.float32)
    n = max(0, min(N, len(audio_full) - start))
    if n > 0:
        a[:n] = audio_full[start:start + n]

    out = {k: np.zeros((MAXLEN, d), np.float32)
           for k, d in [("x", motion.DIM), ("audio", audio_dim)]}
    out["valid"] = np.zeros(MAXLEN, np.float32)
    out["keyframe"] = np.zeros(MAXLEN, np.float32)
    out["x"][:N], out["audio"][:N] = x, a
    out["valid"][:N] = 1.0
    out["keyframe"][0] = out["keyframe"][N - 1] = 1.0
    out["style"] = np.asarray(style, np.float32)
    return out, joints


def collate(items, dev):
    b = {}
    for k in ["x", "audio", "valid", "keyframe"]:
        b[k] = torch.tensor(np.stack([i[k] for i in items])).to(dev)
    b["style"] = torch.tensor(np.array([i["style"] for i in items], np.float32)).to(dev)
    return b


@torch.no_grad()
def sample(model, b, steps=50, guidance=1.0, tie_groups=None, seed=0):
    """tie_groups: lists of item indices that must share one initial noise draw.

    Without this a comparison like "same endpoints, different music" varies both
    the conditioning and the noise, so any divergence between panels is
    uninterpretable. Tying the noise makes the conditioning the only variable.
    """
    obs = b["x"] * b["keyframe"][..., None]
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(b["x"].shape, generator=g).to(b["x"].device)
    for grp in (tie_groups or []):
        for i in grp[1:]:
            x[i] = x[grp[0]]
    B = x.shape[0]
    for i in range(steps):
        t = torch.full((B,), i / steps, device=x.device)
        v = model(x, t, obs, b["keyframe"], b["audio"], b["style"], b["valid"])
        if guidance != 1.0:
            null = torch.ones(B, dtype=torch.bool, device=x.device)
            vu = model(x, t, obs, b["keyframe"], b["audio"], b["style"], b["valid"],
                       drop_audio=null, drop_style=null)
            v = vu + guidance * (v - vu)
        x = x + v / steps
    return x


def to_joints(x, stats_t, rest, parents):
    mean, std = stats_t
    x = x * std + mean
    root, rot6d = x[..., 4:7], x[..., 7:]
    B, T = x.shape[:2]
    R = motion.rotation_6d_to_matrix(rot6d.reshape(B, T, motion.NJOINTS, 6))
    return motion.forward_kinematics(R, root, rest, parents).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--smpl-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = np.load(args.stats)
    stats = {"mean": st["mean"], "std": st["std"], "style_q": st["style_q"]}
    stats_t = (torch.tensor(st["mean"], device=dev), torch.tensor(st["std"], device=dev))

    rest_np, par_np = load_skeleton(args.smpl_model)
    rest = torch.tensor(rest_np, dtype=torch.float32, device=dev)
    parents = torch.tensor(par_np, dtype=torch.long, device=dev)

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    # Infer the dial count from the checkpoint rather than the current
    # kinematics.NAMES, so an older checkpoint still loads after the dial set changes.
    n_style = ck["ema"]["null_style"].shape[0]
    if n_style != len(kinematics.NAMES):
        print(f"WARNING: checkpoint has {n_style} dials but kinematics.NAMES has "
              f"{len(kinematics.NAMES)}; descriptors may not correspond", flush=True)
    model = RhythmFlow(d_model=ck["args"]["d_model"], nlayers=ck["args"]["nlayers"],
                       max_len=ck["args"]["max_len"], n_style=n_style).to(dev)
    model.load_state_dict(ck["ema"] if args.weights == "ema" else ck["model"])
    model.eval()
    print(f"loaded step {ck['step']} ({args.weights} weights), device {dev}", flush=True)

    names = [l.strip() for l in open(args.split) if l.strip()]
    seqs = {n: dict(np.load(os.path.join(args.seq_dir, n + ".npz"), allow_pickle=True))
            for n in names if os.path.exists(os.path.join(args.seq_dir, n + ".npz"))}
    audio = {}

    def get_audio(mid):
        if mid not in audio:
            audio[mid] = np.load(os.path.join(args.audio_dir, mid + ".npy"))
        return audio[mid]

    items, meta, gts = [], [], []

    def add(group, seq_name, start, N, music_id, style, label):
        s = seqs[seq_name]
        it, gt_joints = build_item(s, start, N, get_audio(music_id), style, stats)
        items.append(it)
        gts.append(np.pad(gt_joints, ((0, MAXLEN - N), (0, 0), (0, 0))))
        meta.append(dict(group=group, seq=seq_name, N=N, music=music_id,
                         style=np.asarray(style, np.float32).tolist(),
                         label=label))

    # --- 1. reconstructions across gap lengths and genres -------------------
    ordered = sorted(seqs)
    for i, gap in enumerate([30, 60, 90, 120, 150, 45]):
        name = ordered[i * 3 % len(ordered)]
        s = seqs[name]
        N = min(gap, len(s["root"]))
        start = max(0, (len(s["root"]) - N) // 2)
        _, _, j = canonicalise(s["rot6d"][start:start + N], s["root"][start:start + N],
                            s["joints"][start:start + N])
        stl = kinematics.to_percentile(
            style_value(j[1:-1] if N > 2 else j), stats["style_q"])
        add("recon", name, start, N, str(s["music"]),
            stl, f"{name.split('_')[0]}  gap {N/30:.1f}s")

    # --- 2. same endpoints, different (unseen) music ------------------------
    base = ordered[0]
    s = seqs[base]
    N = min(150, len(s["root"]))
    start = max(0, (len(s["root"]) - N) // 2)
    _, _, j = canonicalise(s["rot6d"][start:start + N], s["root"][start:start + N],
                        s["joints"][start:start + N])
    base_style = kinematics.to_percentile(style_value(j[1:-1]), stats["style_q"])
    test_music = sorted({str(v["music"]) for v in seqs.values()})
    for mid in test_music[:5]:
        add("music", base, start, N, mid, base_style, f"music {mid}")

    # --- 3. one sweep per dial, others held at the clip's own values --------
    for k, dial in enumerate(kinematics.NAMES):
        for lvl in [0.05, 0.3, 0.5, 0.7, 0.95]:
            v = base_style.copy()
            v[k] = lvl
            add(f"style:{dial}", base, start, N, str(s["music"]), v,
                f"{dial} {lvl:.2f}")

    b = collate(items, dev)
    # Panels within the music and style comparisons share one noise draw, so the
    # only thing differing between them is the conditioning.
    groups = ["music"] + [f"style:{d}" for d in kinematics.NAMES]
    tie = [[k for k, m in enumerate(meta) if m["group"] == g] for g in groups]
    tie = [t for t in tie if len(t) > 1]
    print(f"sampling {len(items)} clips, {args.steps} steps, guidance {args.guidance}; "
          f"noise tied within {[len(t) for t in tie]}", flush=True)
    x = sample(model, b, steps=args.steps, guidance=args.guidance, tie_groups=tie)

    gen = to_joints(x, stats_t, rest, parents)
    gt = to_joints(b["x"], stats_t, rest, parents)

    # endpoint error on the conditioned frames, in centimetres
    for k, m in enumerate(meta):
        N = m["N"]
        ends = [0, N - 1]
        m["endpoint_cm"] = float(
            np.linalg.norm(gen[k, ends] - gt[k, ends], axis=-1).mean() * 100)
        m["style_achieved"] = kinematics.to_percentile(
            style_value(gen[k, 1:N - 1]), stats["style_q"]).tolist()
        m["style_raw"] = style_value(gen[k, 1:N - 1]).tolist()
        m["style_gt"] = kinematics.to_percentile(
            style_value(gt[k, 1:N - 1]), stats["style_q"]).tolist()

    np.savez_compressed(args.out, gen=gen, gt=gt, parents=par_np,
                        meta=np.array(meta, dtype=object), step=ck["step"])
    print(f"wrote {args.out}")
    for m in meta:
        req = " ".join(f"{v:.2f}" for v in m["style"])
        ach = " ".join(f"{v:.2f}" for v in m["style_achieved"])
        print(f"  {m['group']:14s} {m['label']:18s} endpoint {m['endpoint_cm']:5.1f}cm")
        print(f"       req [{req}] -> achieved [{ach}]")


if __name__ == "__main__":
    main()
