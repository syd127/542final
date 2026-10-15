"""Extract the 35-d librosa feature stack for every AIST++ music piece, at 30 fps.

Sample rate is chosen as FPS * HOP_LENGTH so that one STFT hop is exactly one
motion frame, which keeps audio and motion index-aligned with no resampling of
the feature sequence afterwards.

Layout (35 channels): envelope 1 | mfcc 20 | chroma 12 | onset one-hot 1 | beat one-hot 1
"""
import argparse, os, glob
import numpy as np
import librosa

FPS, HOP = 30, 512
SR = FPS * HOP  # 15360 Hz -> exactly 30 feature frames per second


def features(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)          # [T]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=20).T     # [T,20]
    chroma = librosa.feature.chroma_cens(
        y=y, sr=sr, hop_length=HOP, n_chroma=12).T                           # [T,12]

    T = min(len(env), len(mfcc), len(chroma))
    env, mfcc, chroma = env[:T], mfcc[:T], chroma[:T]

    peaks = librosa.util.peak_pick(
        env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=5)
    onset = np.zeros(T, np.float32)
    onset[peaks[peaks < T]] = 1.0

    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP, tightness=100)
    beat = np.zeros(T, np.float32)
    beat[beat_frames[beat_frames < T]] = 1.0

    return np.concatenate(
        [env[:, None], mfcc, chroma, onset[:, None], beat[:, None]], axis=1
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    wavs = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    for i, w in enumerate(wavs, 1):
        name = os.path.splitext(os.path.basename(w))[0]
        f = features(w)
        np.save(os.path.join(args.out, name + ".npy"), f)
        print(f"[{i:2d}/{len(wavs)}] {name}: {f.shape} ({f.shape[0]/FPS:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
