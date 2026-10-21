"""Render generated motion to skeleton videos and bundle them into one HTML file.

The output is self-contained (videos are base64-embedded), so it can be copied
off the cluster and opened directly in a browser.
"""
import argparse, base64, html, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

NJ = 24


def bone_list(parents):
    return [(int(parents[j]), j) for j in range(1, NJ)]


def draw(ax, joints, parents, color, alpha=1.0, lw=2.0):
    arts = []
    for p, j in bone_list(parents):
        ln, = ax.plot(*zip(joints[p], joints[j]), color=color, alpha=alpha, lw=lw)
        arts.append(ln)
    sc = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                    s=6, color=color, alpha=alpha, depthshade=False)
    arts.append(sc)
    return arts


def render_panels(panels, labels, parents, N, path, fps=30, dpi=72):
    """panels: list of [T,24,3] arrays, all shown side by side in one video."""
    n = len(panels)
    fig = plt.figure(figsize=(3.1 * n, 3.5), facecolor="#111318")
    axes = [fig.add_subplot(1, n, i + 1, projection="3d") for i in range(n)]

    allj = np.concatenate([p[:N] for p in panels], 0)
    ctr = allj.reshape(-1, 3).mean(0)
    rad = max(np.abs(allj.reshape(-1, 3) - ctr).max(), 0.6)

    for ax, lab in zip(axes, labels):
        ax.set_facecolor("#111318")
        ax.set_title(lab, color="#e6e8ee", fontsize=9, pad=2)
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_ylim(0, 2 * rad)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=12, azim=-70)
        ax.grid(False)
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_pane_color((0.07, 0.08, 0.10, 1.0))
            a.line.set_color((0, 0, 0, 0))
            a.set_ticks([])

    writer = FFMpegWriter(fps=fps, bitrate=700,
                          metadata={"artist": "RhythmFlow"})
    with writer.saving(fig, path, dpi=dpi):
        for t in range(N):
            arts = []
            for ax, pj in zip(axes, panels):
                # faint ghosts of the two conditioned endpoint poses
                arts += draw(ax, pj[0], parents, "#4a5568", alpha=0.35, lw=1.0)
                arts += draw(ax, pj[N - 1], parents, "#4a5568", alpha=0.35, lw=1.0)
                col = "#7dd3fc" if t not in (0, N - 1) else "#fbbf24"
                arts += draw(ax, pj[t], parents, col, alpha=1.0, lw=2.2)
            writer.grab_frame()
            for a in arts:
                a.remove()
    plt.close(fig)
    return os.path.getsize(path)


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def video_block(path, caption):
    return (f'<figure><video src="data:video/mp4;base64,{b64(path)}" '
            f'controls autoplay loop muted playsinline></video>'
            f'<figcaption>{caption}</figcaption></figure>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    d = np.load(args.samples, allow_pickle=True)
    gen, gt, parents, meta = d["gen"], d["gt"], d["parents"], list(d["meta"])
    step = int(d["step"])

    sections = {"recon": [], "music": [], "style": []}
    total = 0

    for k, m in enumerate(meta):
        N = m["N"]
        if m["group"] == "recon":
            path = os.path.join(args.workdir, f"recon_{k}.mp4")
            sz = render_panels([gt[k], gen[k]], ["ground truth", "generated"],
                               parents, N, path)
            cap = (f"<b>{html.escape(m['label'])}</b> &middot; unseen music "
                   f"{html.escape(m['music'])} &middot; endpoint error "
                   f"{m['endpoint_cm']:.1f} cm")
            sections["recon"].append(video_block(path, cap))
            total += sz

    style_groups = [f"style:{d}" for d in kinematics.NAMES]
    for grp in ["music"] + style_groups:
        idx = [k for k, m in enumerate(meta) if m["group"] == grp]
        if not idx:
            continue
        N = meta[idx[0]]["N"]
        path = os.path.join(args.workdir, f"{grp}.mp4")
        labels = [meta[k]["label"] for k in idx]
        sz = render_panels([gen[k] for k in idx], labels, parents, N, path)
        if grp == "music":
            cap = ("Identical start and end pose in every panel; only the music "
                   "differs. All five tracks are unseen in training. Noise is tied "
                   "across panels, so the audio is the only variable.")
            sections["music"].append(video_block(path, cap))
        else:
            dial = grp.split(":", 1)[1]
            di = kinematics.NAMES.index(dial)
            pairs = ", ".join(
                f"{meta[k]['style'][di]:.2f}&rarr;{meta[k]['style_achieved'][di]:.2f}"
                for k in idx)
            cap = (f"<b>{html.escape(dial)}</b> swept; the other four dials held at "
                   f"this clip's own values. requested&rarr;achieved: {pairs}")
            sections["style"].append(video_block(path, cap))
        total += sz

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RhythmFlow Samples</title>
<style>
  :root {{ --bg:#0d0f14; --panel:#161922; --fg:#e6e8ee; --muted:#9aa3b2; --accent:#7dd3fc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:40px 16px 80px; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-.02em; }}
  .sub {{ color:var(--muted); margin:0 0 36px; }}
  h2 {{ font-size:18px; margin:40px 0 6px; border-left:3px solid var(--accent);
        padding-left:10px; }}
  .note {{ color:var(--muted); margin:0 0 18px; font-size:14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px; }}
  .wide .grid {{ grid-template-columns:1fr; }}
  figure {{ margin:0; background:var(--panel); border:1px solid #242938;
            border-radius:10px; overflow:hidden; }}
  video {{ width:100%; display:block; background:#111318; }}
  figcaption {{ padding:10px 12px; font-size:13px; color:var(--muted); }}
  b {{ color:var(--fg); }}
  .legend {{ display:flex; gap:18px; flex-wrap:wrap; color:var(--muted);
             font-size:13px; margin-bottom:28px; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%;
          margin-right:6px; vertical-align:middle; }}
</style></head><body><div class="wrap">
<h1>RhythmFlow &mdash; held-out samples</h1>
<p class="sub">Checkpoint at step {step:,} (EMA weights). Every clip below uses a
sequence <em>and</em> a music track the model never saw: the AIST++ crossmodal
test split shares no sequence and no track with training.</p>
<div class="legend">
  <span><span class="dot" style="background:#7dd3fc"></span>generated / interior frames</span>
  <span><span class="dot" style="background:#fbbf24"></span>conditioned keyframe</span>
  <span><span class="dot" style="background:#4a5568"></span>ghost: target start &amp; end pose</span>
</div>

<h2>1. Reconstruction on held-out clips</h2>
<p class="note">Ground truth beside the generated motion, over a range of gap
lengths. The two are not expected to match frame for frame &mdash; the model
generates a plausible transition, not the recorded one. What should match is the
start and end pose, and the overall character of the movement.</p>
<div class="grid">{''.join(sections['recon'])}</div>

<h2>2. Same endpoints, different music</h2>
<p class="note">The core question of the project. Start and end poses are held
fixed across all panels; only the audio changes. If the model is genuinely using
the music, these should diverge in the interior while still meeting at both ends.</p>
<div class="wide"><div class="grid">{''.join(sections['music'])}</div></div>

<h2>3. Style dial</h2>
<p class="note">Same endpoints, same music, sweeping the kinematic style scalar
from low to high. Higher values should read as faster, sharper movement.</p>
<div class="wide"><div class="grid">{''.join(sections['style'])}</div></div>
</div></body></html>"""

    with open(args.out, "w") as f:
        f.write(page)
    print(f"videos {total/1e6:.1f} MB -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
