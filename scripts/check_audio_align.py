"""Confirm where each AIST++ sequence starts inside its music track.

The HF mirror ships a per-window audio slice, so each slice can be located
inside the full track by cross-correlation. If window w of a sequence lands at
w*75 frames (2.5 s), motion frame 0 corresponds to music time 0 and audio can be
indexed with the motion frame number directly.
"""
import glob, io, os, re, sys
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import correlate

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def locate(clip, track, sr):
    """Offset in seconds of `clip` inside `track`, by normalised cross-correlation."""
    c = correlate(track, clip, mode="valid", method="fft")
    energy = np.sqrt(correlate(track ** 2, np.ones(len(clip)), mode="valid", method="fft"))
    score = c / (energy * np.linalg.norm(clip) + 1e-9)
    i = int(np.argmax(score))
    return i / sr, float(score[i])


def main():
    files = sorted(glob.glob(os.path.join(B, "aistpp_hf_v3", "*.parquet")))
    d = pq.read_table(files[0], columns=["name", "audio"]).to_pydict()
    index = {n: i for i, n in enumerate(d["name"])}

    targets = [n for n in d["name"] if re.search(r"_w[012]$", n)][:12]
    tracks = {}
    print(f"{'window':<34} {'expected':>9} {'found':>9} {'corr':>6}")
    for name in targets:
        seq, w = re.match(r"(.+)_w(\d+)$", name).groups()
        music = re.match(r".*_(m\w+?)_ch\d+$", seq).group(1)
        if music not in tracks:
            y, sr = sf.read(os.path.join(B, "aist_music", "wav", "wav", music + ".wav"))
            tracks[music] = (y.mean(1).astype(np.float64), sr)
        track, sr = tracks[music]

        clip, _ = sf.read(io.BytesIO(d["audio"][index[name]]["bytes"]))
        clip = clip.mean(1).astype(np.float64)
        off, score = locate(clip, track, sr)
        print(f"{name:<34} {int(w)*75/30:>8.2f}s {off:>8.2f}s {score:>6.3f}")


if __name__ == "__main__":
    main()
