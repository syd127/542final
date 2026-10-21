"""Normalisation statistics and the style-dial quantile table.

Both are computed on canonicalised training windows, matching exactly what the
model sees, and on the training split only.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
from dataset import canonicalise, style_value, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--min-len", type=int, default=15)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--std-floor", type=float, default=1e-2,
                    help="channels with std below this are treated as frozen")
    args = ap.parse_args()

    names = [n for n in load_split(args.split)
             if os.path.exists(os.path.join(args.seq_dir, n + ".npz"))]
    rng = np.random.default_rng(0)
    cache = {}

    feats, styles = [], []
    for _ in range(args.samples):
        n = names[rng.integers(len(names))]
        if n not in cache:
            cache[n] = dict(np.load(os.path.join(args.seq_dir, n + ".npz"),
                                    allow_pickle=True))
        d = cache[n]
        T = len(d["root"])
        N = min(int(rng.integers(args.min_len, args.max_len + 1)), T)
        s = int(rng.integers(0, T - N + 1))
        sl = slice(s, s + N)
        rot6d, root, joints = canonicalise(d["rot6d"][sl], d["root"][sl], d["joints"][sl])
        feats.append(np.concatenate([d["contacts"][sl], root, rot6d], axis=1))
        styles.append(style_value(joints[1:-1] if N > 2 else joints))

    X = np.concatenate(feats, 0)
    mean, std = X.mean(0), X.std(0)

    # AIST++ is fit from COCO-17 keypoints, which have no toe or hand landmarks,
    # so the distal joints (ankles, feet, wrists, hands) carry no evidence and sit
    # at identity. Dividing those channels by their ~1e-4 std would amplify pure
    # fitting noise by thousands and hand a third of the loss budget to nothing.
    # Giving them std 1 leaves them ~0 after centring, so they cost the model
    # nothing and still decode back to identity rotations.
    frozen = std < args.std_floor
    std[frozen] = 1.0

    # one quantile table per dial, rows in kinematics.NAMES order
    S = np.atleast_2d(np.array(styles))
    q = np.stack([np.percentile(S[:, k], np.arange(101))
                  for k in range(S.shape[1])])
    np.savez(args.out, mean=mean.astype(np.float32), std=std.astype(np.float32),
             style_q=q.astype(np.float32), frozen=frozen)

    live = ~frozen
    print(f"windows {args.samples} frames {len(X)}")
    print(f"  frozen channels: {int(frozen.sum())}/{len(std)} "
          f"(joints {sorted({(i-7)//6 for i in np.where(frozen)[0] if i >= 7})})")
    print(f"  live std range [{std[live].min():.4f}, {std[live].max():.4f}]")
    for k, name in enumerate(kinematics.NAMES):
        print(f"  dial {name:<12} p1 {q[k][1]:9.3f}  p50 {q[k][50]:9.3f}  p99 {q[k][99]:9.3f}")
    if S.shape[1] > 1:
        c = np.corrcoef(S.T)
        print("  dial correlation matrix:")
        print("    " + "  ".join(f"{n[:7]:>7}" for n in kinematics.NAMES))
        for k, n in enumerate(kinematics.NAMES):
            print(f"    {n[:7]:>7} " + "  ".join(f"{c[k, j]:+7.2f}" for j in range(S.shape[1])))


if __name__ == "__main__":
    main()
