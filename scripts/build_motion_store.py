"""Rebuild full-length 30fps AIST++ SMPL sequences from the windowed HF mirror.

The official motions.zip release asset is truncated (411/1408 sequences), so the
motion source is haiphamcse/aistpp_v3, which stores the same annotations cut into
150-frame windows at stride 75. Windows tile each sequence exactly, so stitching
them recovers the original up to a tail of at most 149 frames.

Every sequence that also exists in the official release is checked bit-for-bit.
"""
import argparse, glob, os, pickle, re
import numpy as np
import pyarrow.parquet as pq

WIN, STRIDE = 150, 75


def load_windows(parquet_dir):
    seqs = {}
    for f in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.read_table(f, columns=["name", "smpl_poses", "smpl_trans", "smpl_scaling"])
        d = t.to_pydict()
        for name, poses, trans, scaling in zip(
            d["name"], d["smpl_poses"], d["smpl_trans"], d["smpl_scaling"]
        ):
            seq, w = re.match(r"(.+)_w(\d+)$", name).groups()
            seqs.setdefault(seq, {})[int(w)] = (
                np.asarray(poses, np.float32).reshape(WIN, 72),
                np.asarray(trans, np.float32).reshape(WIN, 3),
                np.asarray(scaling, np.float32).ravel()[0],
            )
    return seqs


def stitch(windows):
    n = max(windows) + 1
    assert set(windows) == set(range(n)), "non-contiguous windows"
    length = WIN + STRIDE * (n - 1)
    poses = np.zeros((length, 72), np.float32)
    trans = np.zeros((length, 3), np.float32)
    for w in range(n):
        s = w * STRIDE
        poses[s : s + WIN] = windows[w][0]
        trans[s : s + WIN] = windows[w][1]
    return poses, trans, windows[0][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--official-motions", required=True, help="dir of authentic .pkl, for verification")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seqs = load_windows(args.parquet_dir)
    print(f"stitching {len(seqs)} sequences")

    checked = bad = 0
    for seq, windows in sorted(seqs.items()):
        poses, trans, scaling = stitch(windows)

        ref_path = os.path.join(args.official_motions, seq + ".pkl")
        if os.path.exists(ref_path):
            with open(ref_path, "rb") as fh:
                ref = pickle.load(fh)
            rp = np.asarray(ref["smpl_poses"], np.float32)[::2][: len(poses)]
            rt = np.asarray(ref["smpl_trans"], np.float32)[::2][: len(poses)]
            checked += 1
            if not (np.array_equal(rp, poses) and np.array_equal(rt, trans)):
                bad += 1
                print(f"  MISMATCH {seq}")

        np.savez(
            os.path.join(args.out, seq + ".npz"),
            smpl_poses=poses, smpl_trans=trans, smpl_scaling=scaling, fps=np.int32(30),
        )

    print(f"verified {checked} sequences against the official release, {bad} mismatches")
    total = sum(WIN + STRIDE * (max(w) ) for w in seqs.values())
    print(f"wrote {len(seqs)} sequences, {total} frames = {total/30/3600:.2f} h @30fps")


if __name__ == "__main__":
    main()
