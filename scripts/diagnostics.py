"""Two diagnostics that need neither a GPU nor the model.

1. BAS chance level. Beat alignment is scored by how close each music beat is to
   some motion beat. With 3-5 velocity minima per second there may be almost no
   dynamic range left: score ground-truth motion against its OWN track and
   against WRONG tracks. If those are nearly equal, BAS cannot demonstrate beat
   synchronisation on these window lengths and no training run will change that.

2. Output low-pass as post-processing. The samples are already generated, so
   filtering them costs nothing. Reports jitter, foot skate and endpoint error
   before and after, with the two conditioned keyframes re-pinned so the filter
   cannot be credited with breaking the constraint it was given.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
import motion
from evaluate import beat_alignment, endpoint_cm, foot_skate, hfr, motion_beats

FPS = 30


def chance_level(samples, audio_dir, seed=0):
    d = np.load(samples, allow_pickle=True)
    gen, gt, meta = d["gen"], d["gt"], list(d["meta"])
    tracks = sorted({m["music"] for m in meta})
    aud = {t: np.load(os.path.join(audio_dir, t + ".npy")) for t in tracks}
    rng = np.random.default_rng(seed)

    rows = []
    for k, m in enumerate(meta):
        if m["group"] != "recon":
            continue
        N, true_t = m["N"], m["music"]
        others = [t for t in tracks if t != true_t]
        a_true = np.zeros((N, 35), np.float32)
        n = min(N, len(aud[true_t]))
        a_true[:n] = aud[true_t][:n]

        def bas(joints, track):
            a = np.zeros((N, 35), np.float32)
            nn = min(N, len(aud[track]))
            a[:nn] = aud[track][:nn]
            return beat_alignment(joints[:N], a)

        wrong = [t for t in rng.choice(others, size=min(5, len(others)), replace=False)]
        rows.append(dict(
            gt_true=bas(gt[k], true_t),
            gt_wrong=np.nanmean([bas(gt[k], t) for t in wrong]),
            gen_true=bas(gen[k], true_t),
            gen_wrong=np.nanmean([bas(gen[k], t) for t in wrong]),
            gt_beats=len(motion_beats(gt[k, :N])) / (N / FPS),
            gen_beats=len(motion_beats(gen[k, :N])) / (N / FPS),
        ))

    print("=== 1. BAS chance level (ground truth vs its own track vs wrong tracks) ===")
    print(f"  clips: {len(rows)}\n")
    print(f"  {'':<22} {'own track':>11} {'wrong tracks':>14} {'headroom':>10}")
    for lbl, a, b in [("ground-truth motion", "gt_true", "gt_wrong"),
                      ("generated motion", "gen_true", "gen_wrong")]:
        t_, w_ = np.nanmean([r[a] for r in rows]), np.nanmean([r[b] for r in rows])
        print(f"  {lbl:<22} {t_:>11.3f} {w_:>14.3f} {t_-w_:>+10.3f}")
    print(f"\n  motion beats/s: ground truth {np.mean([r['gt_beats'] for r in rows]):.2f}, "
          f"generated {np.mean([r['gen_beats'] for r in rows]):.2f}")
    gtt, gtw = np.nanmean([r["gt_true"] for r in rows]), np.nanmean([r["gt_wrong"] for r in rows])
    print(f"\n  VERDICT: ground truth beats its own track by {gtt-gtw:+.3f} over wrong tracks.")
    print("  " + ("BAS has usable dynamic range; a beat-sync claim is testable."
                  if gtt - gtw > 0.05 else
                  "BAS has NO usable dynamic range on these windows -- even real dance\n"
                  "  scores the same against the wrong music. No training run can\n"
                  "  demonstrate beat synchronisation with this metric. Report as a finding."))


def postfilter(samples, cutoffs=(8.0, 6.0)):
    d = np.load(samples, allow_pickle=True)
    gen, gt, meta = d["gen"], d["gt"], list(d["meta"])
    idx = [k for k, m in enumerate(meta) if m["group"] == "recon"]

    def score(joints_list):
        return (np.mean([endpoint_cm(joints_list[i][:meta[k]["N"]],
                                     gt[k][:meta[k]["N"]], meta[k]["N"])
                         for i, k in enumerate(idx)]),
                np.mean([100 * hfr(joints_list[i][:meta[k]["N"]]) for i, k in enumerate(idx)]),
                np.mean([foot_skate(joints_list[i][:meta[k]["N"]]) for i, k in enumerate(idx)]))

    print("\n=== 2. Output low-pass as post-processing (keyframes re-pinned) ===")
    print(f"  {'variant':<26} {'endpoint cm':>12} {'HFR %':>9} {'skate':>8}")
    e, h, s = score([gt[k] for k in idx])
    print(f"  {'ground truth':<26} {e:>12.2f} {h:>9.3f} {s:>8.3f}")
    e, h, s = score([gen[k] for k in idx])
    print(f"  {'generated, unfiltered':<26} {e:>12.2f} {h:>9.3f} {s:>8.3f}")
    for c in cutoffs:
        out = []
        for k in idx:
            N = meta[k]["N"]
            j = kinematics.lowpass(gen[k][:N], cutoff=c).copy()
            j[0], j[N - 1] = gen[k][0], gen[k][N - 1]   # re-pin the conditioned frames
            out.append(np.pad(j, ((0, len(gen[k]) - N), (0, 0), (0, 0))))
        e, h, s = score(out)
        print(f"  {f'generated, {c:g} Hz low-pass':<26} {e:>12.2f} {h:>9.3f} {s:>8.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--audio-dir", required=True)
    args = p.parse_args()
    chance_level(args.samples, args.audio_dir)
    postfilter(args.samples)


if __name__ == "__main__":
    main()
