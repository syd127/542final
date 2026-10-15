"""Where does motion energy live in frequency, and where does jitter live?

Compares the power spectrum of generated joint trajectories against ground truth
on the same held-out clips. Human dance motion is essentially bandlimited; a
generated-minus-real excess concentrated at high frequency is jitter, and tells
us where to put a filter cutoff for the style descriptors.
"""
import argparse, os, sys
import numpy as np
from scipy.signal import welch

FPS = 30


NPERSEG = 32  # fixed so every clip yields the same frequency bins


def psd(joints, fps=FPS, nperseg=NPERSEG):
    """Mean PSD over joints and axes of a [T,J,3] trajectory, after detrending."""
    x = joints.reshape(len(joints), -1)
    x = x - x.mean(0, keepdims=True)
    f, p = welch(x, fs=fps, nperseg=nperseg, axis=0)
    return f, p.mean(1)


def band_fractions(f, p, edges):
    tot = np.trapezoid(p, f)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        out.append(float(np.trapezoid(p[m], f[m]) / tot) if m.sum() > 1 else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    d = np.load(args.samples, allow_pickle=True)
    gen, gt, meta = d["gen"], d["gt"], list(d["meta"])

    edges = [0, 2, 4, 6, 8, 10, 12, 15]
    acc_gen, acc_gt, F = [], [], None
    for k, m in enumerate(meta):
        if m["group"] != "recon":
            continue
        N = m["N"]
        if N < NPERSEG:
            continue
        f, pg = psd(gen[k, :N])
        _, pr = psd(gt[k, :N])
        F = f
        acc_gen.append(pg)
        acc_gt.append(pr)

    pg, pr = np.mean(acc_gen, 0), np.mean(acc_gt, 0)
    print(f"held-out recon clips analysed: {len(acc_gen)}   (Nyquist {FPS/2} Hz)\n")
    print(f"  {'band (Hz)':<12} {'ground truth':>14} {'generated':>12} {'excess':>10}")
    bg, br = band_fractions(F, pg, edges), band_fractions(F, pr, edges)
    for (lo, hi), a, b in zip(zip(edges[:-1], edges[1:]), br, bg):
        print(f"  {str(lo)+'-'+str(hi):<12} {100*a:>13.2f}% {100*b:>11.2f}% {100*(b-a):>+9.2f}%")

    # power above a cutoff, which is what a low-pass would remove
    print(f"\n  {'cutoff':<12} {'GT power above':>16} {'gen power above':>17}  ratio")
    for c in [4, 6, 8, 10, 12]:
        m = F >= c
        a = np.trapezoid(pr[m], F[m]) / np.trapezoid(pr, F)
        b = np.trapezoid(pg[m], F[m]) / np.trapezoid(pg, F)
        print(f"  >{c:>2} Hz{'':<6} {100*a:>15.3f}% {100*b:>16.3f}%  {b/max(a,1e-12):>6.1f}x")

    # reference: clean training motion, to locate the annotation noise floor
    names = [l.strip() for l in open(args.split) if l.strip()][:40]
    acc = []
    for n in names:
        p = os.path.join(args.seq_dir, n + ".npz")
        if not os.path.exists(p):
            continue
        j = np.load(p, allow_pickle=True)["joints"]
        if len(j) < 150:
            continue
        f, pp = psd(j[:150])
        acc.append(pp)
    if acc:
        pt = np.mean(acc, 0)
        print(f"\n  training motion ({len(acc)} clips), power above cutoff:")
        for c in [4, 6, 8, 10, 12]:
            m = f >= c
            print(f"    >{c:>2} Hz: {100*np.trapezoid(pt[m], f[m])/np.trapezoid(pt, f):.3f}%")


if __name__ == "__main__":
    main()
