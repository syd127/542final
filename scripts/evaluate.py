"""Quantitative evaluation on the held-out test split.

Four things, none of which needs further training:

  NFE sweep        quality vs number of ODE steps, which is the efficiency claim
  SLERP baseline   pure interpolation between the two conditioned poses
  no-audio         sample with the music dropped via classifier-free guidance
  metrics          endpoint error, beat alignment, foot skate, and a
                   high-frequency power ratio that measures jitter directly

HFR is reported for ground truth too, because the interesting quantity is the
gap: real motion sets the floor that generated motion should be compared against.
"""
import argparse, os, sys
import numpy as np
import torch
from scipy.signal import welch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
import motion
from dataset import canonicalise
from generate_samples import build_item, collate, sample, to_joints
from model import RhythmFlow
from smpl_fk import load_skeleton

FPS = 30


# ---------- metrics ---------------------------------------------------------
def endpoint_cm(gen, gt, N):
    e = [0, N - 1]
    return float(np.linalg.norm(gen[e] - gt[e], axis=-1).mean() * 100)


def hfr(joints, cutoff=8.0, fps=FPS, nperseg=64):
    """Fraction of position power above `cutoff`: a direct jitter measure."""
    x = joints.reshape(len(joints), -1)
    x = x - x.mean(0, keepdims=True)
    f, p = welch(x, fs=fps, nperseg=min(nperseg, len(x)), axis=0)
    p = p.mean(1)
    tot = np.trapezoid(p, f)
    m = f >= cutoff
    return float(np.trapezoid(p[m], f[m]) / tot) if tot > 0 else 0.0


def motion_beats(joints, fps=FPS):
    """Frames where the body decelerates to a local minimum of mean joint speed."""
    v = np.linalg.norm(np.diff(kinematics.lowpass(joints), axis=0), axis=-1).mean(1)
    if len(v) < 3:
        return np.array([], int)
    lo = (v[1:-1] < v[:-2]) & (v[1:-1] < v[2:])
    return np.where(lo)[0] + 1


def beat_alignment(joints, audio, sigma=3.0, fps=FPS):
    """Standard BAS: for each music beat, closeness of the nearest motion beat."""
    ab = np.where(audio[:, 34] > 0.5)[0]
    mb = motion_beats(joints, fps)
    if len(ab) == 0 or len(mb) == 0:
        return np.nan
    d = np.abs(ab[:, None] - mb[None, :]).min(1)
    return float(np.exp(-(d ** 2) / (2 * sigma ** 2)).mean())


def foot_skate(joints, h_thresh=0.05, fps=FPS):
    """Mean horizontal foot speed while that foot is near the floor (m/s)."""
    feet = joints[:, list(motion.FOOT_JOINTS)]
    h = feet[..., motion.UP]
    vh = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1) * fps
    contact = h[1:] < h_thresh
    return float(vh[contact].mean()) if contact.any() else 0.0


# ---------- SLERP baseline --------------------------------------------------
def slerp_clip(x_denorm, N):
    """Interpolate between the two conditioned poses: slerp rotations, lerp root."""
    root, rot6d = x_denorm[..., 4:7], x_denorm[..., 7:]
    R = motion.rotation_6d_to_matrix(
        torch.from_numpy(rot6d[[0, N - 1]]).reshape(2, motion.NJOINTS, 6)).numpy()
    t = np.linspace(0, 1, N)
    out = np.empty((N, motion.NJOINTS, 3, 3))
    for j in range(motion.NJOINTS):
        sl = Slerp([0.0, 1.0], Rotation.from_matrix(R[:, j]))
        out[:, j] = sl(t).as_matrix()
    r = root[0][None] * (1 - t)[:, None] + root[N - 1][None] * t[:, None]
    return out, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--smpl-model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip-len", type=int, default=150)
    p.add_argument("--per-seq", type=int, default=2)
    p.add_argument("--guidance", type=float, default=2.0)
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

    names = [l.strip() for l in open(args.split) if l.strip()]
    rng = np.random.default_rng(0)
    items, gts, audios, Ns = [], [], [], []
    for n in names:
        p_ = os.path.join(args.seq_dir, n + ".npz")
        if not os.path.exists(p_):
            continue
        d = dict(np.load(p_, allow_pickle=True))
        T = len(d["root"])
        a = np.load(os.path.join(args.audio_dir, str(d["music"]) + ".npy"))
        for _ in range(args.per_seq):
            N = min(args.clip_len, T)
            s = int(rng.integers(0, max(1, T - N + 1)))
            _, _, j = canonicalise(d["rot6d"][s:s + N], d["root"][s:s + N],
                                   d["joints"][s:s + N])
            stl = kinematics.to_percentile(kinematics.descriptors(j), stats["style_q"])
            it, gt_j = build_item(d, s, N, a, stl, stats)
            items.append(it)
            gts.append(gt_j)
            audios.append(it["audio"][:N])
            Ns.append(N)
    print(f"eval clips: {len(items)}", flush=True)

    b = collate(items, dev)
    gt_joints = to_joints(b["x"], stats_t, rest, parents)

    def score(gen_joints, label):
        row = {"method": label}
        for key, fn in [
            ("endpoint_cm", lambda g, r, a, N: endpoint_cm(g[:N], r[:N], N)),
            ("BAS", lambda g, r, a, N: beat_alignment(g[:N], a)),
            ("HFR%", lambda g, r, a, N: 100 * hfr(g[:N])),
            ("skate", lambda g, r, a, N: foot_skate(g[:N])),
            # BAS rewards dense motion beats: more velocity minima means more
            # chances to fall near some audio beat, so a jittery model can
            # inflate its own score. Report the density so that is visible.
            ("beats/s", lambda g, r, a, N: len(motion_beats(g[:N])) / (N / FPS)),
        ]:
            vals = [fn(gen_joints[k], gt_joints[k], audios[k], Ns[k])
                    for k in range(len(Ns))]
            row[key] = float(np.nanmean(vals))
        return row

    rows = []

    # ground truth, as the reference floor for BAS / HFR / skate
    rows.append(score(gt_joints, "ground truth"))

    # SLERP baseline
    xg = (b["x"] * torch.tensor(st["std"], device=dev)
          + torch.tensor(st["mean"], device=dev)).cpu().numpy()
    sl = np.empty_like(gt_joints)
    for k, N in enumerate(Ns):
        Rm, r = slerp_clip(xg[k], N)
        Rt = torch.tensor(Rm, dtype=torch.float32, device=dev)[None]
        rt = torch.tensor(r, dtype=torch.float32, device=dev)[None]
        j = motion.forward_kinematics(Rt, rt, rest, parents)[0].cpu().numpy()
        sl[k, :N] = j
        sl[k, N:] = j[-1]
    rows.append(score(sl, "SLERP interpolation"))

    # NFE sweep
    for nfe in [1, 2, 5, 10, 25, 50]:
        x = sample(model, b, steps=nfe, guidance=args.guidance, seed=0)
        rows.append(score(to_joints(x, stats_t, rest, parents), f"flow, {nfe} steps"))
        print(f"  done NFE {nfe}", flush=True)

    # no-audio ablation: music dropped through the CFG null path
    obs = b["x"] * b["keyframe"][..., None]
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(b["x"].shape, generator=g).to(dev)
    null = torch.ones(x.shape[0], dtype=torch.bool, device=dev)
    with torch.no_grad():
        for i in range(50):
            t = torch.full((x.shape[0],), i / 50, device=dev)
            v = model(x, t, obs, b["keyframe"], b["audio"], b["style"], b["valid"],
                      drop_audio=null)
            x = x + v / 50
    rows.append(score(to_joints(x, stats_t, rest, parents), "flow, 50 steps, NO audio"))

    hdr = ["method", "endpoint_cm", "BAS", "beats/s", "HFR%", "skate"]
    w = [26, 12, 7, 9, 8, 7]
    print("\n" + "".join(h.ljust(x) for h, x in zip(hdr, w)))
    print("-" * sum(w))
    for r in rows:
        print(r["method"].ljust(w[0])
              + f"{r['endpoint_cm']:<12.2f}{r['BAS']:<7.3f}{r['beats/s']:<9.2f}"
              + f"{r['HFR%']:<8.3f}{r['skate']:<7.3f}")
    np.savez(args.out, rows=np.array(rows, dtype=object))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
