"""Does each style dial actually control what it claims to?

For every dial: monotonicity of achieved-vs-requested, the gain (slope, 1.0 =
perfectly calibrated), and the achieved range. Then the leakage matrix
M_ij = d(achieved_j)/d(requested_i), which must be read against the data prior
C_ij -- the correlation the dials already have in real motion. Leakage that
matches the prior means the model is following the data manifold, which is
correct behaviour for a generative model; only excess leakage is entanglement.
"""
import argparse, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--seq-dir")
    ap.add_argument("--split")
    args = ap.parse_args()

    meta = list(np.load(args.samples, allow_pickle=True)["meta"])
    names = kinematics.NAMES
    K = len(names)

    print("=== calibration on held-out clips (conditioned on their own GT dials) ===")
    rec = [m for m in meta if m["group"] == "recon"]
    if rec:
        req = np.array([m["style"] for m in rec])
        ach = np.array([m["style_achieved"] for m in rec])
        print(f"  {'dial':<9} {'mean |err|':>11} {'bias':>7}  (percentile units)")
        for k, n in enumerate(names):
            e = ach[:, k] - req[:, k]
            print(f"  {n:<9} {np.abs(e).mean():11.3f} {e.mean():+7.3f}")

    print("\n=== per-dial response ===")
    print(f"  {'dial':<9} {'spearman':>9} {'gain':>6} {'achieved range':>22} {'n':>3}")
    M = np.full((K, K), np.nan)
    for i, n in enumerate(names):
        g = [m for m in meta if m["group"] == f"style:{n}"]
        if len(g) < 3:
            continue
        r = np.array([m["style"][i] for m in g])
        o = np.argsort(r)
        r = r[o]
        A = np.array([m["style_achieved"] for m in g])[o]
        rho = spearmanr(r, A[:, i]).statistic
        for j in range(K):
            M[i, j] = np.polyfit(r, A[:, j], 1)[0]
        print(f"  {n:<9} {rho:>9.2f} {M[i, i]:6.2f} "
              f"{A[:, i].min():9.2f} -> {A[:, i].max():<9.2f} {len(g):>3}")

    # data prior: correlation of the dials in real training motion, in the same
    # percentile space the dials are requested in
    C = None
    if args.seq_dir and args.split:
        rng = np.random.default_rng(0)
        q = np.load(args.stats)["style_q"]
        nm = [l.strip() for l in open(args.split) if l.strip()]
        nm = [x for x in nm if os.path.exists(os.path.join(args.seq_dir, x + ".npz"))]
        rows = []
        for _ in range(600):
            d = dict(np.load(os.path.join(
                args.seq_dir, nm[rng.integers(len(nm))] + ".npz"), allow_pickle=True))
            T = len(d["root"])
            N = min(int(rng.integers(15, 151)), T)
            s = int(rng.integers(0, T - N + 1))
            rows.append(kinematics.to_percentile(
                kinematics.descriptors(d["joints"][s:s + N]), q))
        C = np.corrcoef(np.array(rows).T)

    print("\n=== leakage matrix M (row = dial swept, col = dial measured) ===")
    hdr = "  " + " " * 9 + "".join(f"{n[:7]:>9}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        print(f"  {n:<9}" + "".join(f"{M[i, j]:9.2f}" for j in range(K)))

    if C is not None:
        print("\n=== data prior C (correlation of dials in real motion) ===")
        print(hdr)
        for i, n in enumerate(names):
            print(f"  {n:<9}" + "".join(f"{C[i, j]:9.2f}" for j in range(K)))

        print("\n=== excess leakage M - C (off-diagonal; ~0 means it follows the data) ===")
        print(hdr)
        off = []
        for i, n in enumerate(names):
            row = ""
            for j in range(K):
                if i == j:
                    row += f"{'--':>9}"
                else:
                    e = M[i, j] - C[i, j]
                    off.append(abs(e))
                    row += f"{e:+9.2f}"
            print(f"  {n:<9}" + row)
        print(f"\n  mean |excess off-diagonal| = {np.nanmean(off):.3f}")

    print("\n=== music swap: same endpoints, same dials, tied noise ===")
    mus = [m for m in meta if m["group"] == "music"]
    if mus:
        A = np.array([m["style_achieved"] for m in mus])
        print(f"  {len(mus)} unseen tracks; spread in achieved dials "
              f"(only the audio differs):")
        for k, n in enumerate(names):
            print(f"    {n:<9} min {A[:, k].min():.2f}  max {A[:, k].max():.2f}  "
                  f"range {A[:, k].ptp() if hasattr(A[:, k], 'ptp') else np.ptp(A[:, k]):.2f}")
        print(f"  endpoint error {np.mean([m['endpoint_cm'] for m in mus]):.1f} cm mean")


if __name__ == "__main__":
    main()
