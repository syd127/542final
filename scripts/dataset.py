"""Inbetweening clips from AIST++: variable-length windows with 1-frame endpoints.

Each sample is a window of N frames (N sampled in [min_len, max_len], i.e.
0.5-5 s at 30 fps) padded to max_len. Frames 0 and N-1 are the keyframes; the
model sees their values plus a binary observation mask and must produce the
interior. Windows are canonicalised to the first frame so the task is invariant
to where in the room the dancer happens to be.
"""
import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinematics
import motion

FPS = 30


def canonicalise(rot6d, root, joints):
    """Translate the first frame's root to the xz origin and face it along +z."""
    yaw = motion.facing_yaw(joints[0])
    R = motion.yaw_matrix(-yaw)
    t0 = np.array([root[0, 0], 0.0, root[0, 2]], np.float32)

    root_c = (root - t0) @ R.T
    joints_c = (joints - t0) @ R.T

    # Only the root joint's rotation is global; the rest are parent-relative.
    rot = rot6d.reshape(len(rot6d), motion.NJOINTS, 6).copy()
    m0 = motion.rotation_6d_to_matrix(torch.from_numpy(rot[:, 0])).numpy()
    rot[:, 0] = motion.matrix_to_rotation_6d(
        torch.from_numpy(R[None] @ m0)).numpy()
    return rot.reshape(len(rot6d), -1), root_c, joints_c


def style_value(joints, fps=FPS):
    """Raw style descriptor vector for a clip; see scripts/kinematics.py."""
    return kinematics.descriptors(joints, fps=fps)


class AISTPPInbetween(Dataset):
    def __init__(self, seq_dir, audio_dir, names, min_len=15, max_len=150,
                 samples_per_epoch=20000, train=True, stats=None, seed=0):
        self.seq_dir, self.audio_dir = seq_dir, audio_dir
        self.min_len, self.max_len = min_len, max_len
        self.train, self.stats = train, stats
        self.samples_per_epoch = samples_per_epoch

        self.seqs, self.lengths = [], []
        for n in names:
            p = os.path.join(seq_dir, n + ".npz")
            if not os.path.exists(p):
                continue
            with np.load(p, allow_pickle=True) as d:
                T = len(d["root"])
            if T >= min_len:
                self.seqs.append(n)
                self.lengths.append(T)
        self._audio, self._seq = {}, {}
        if not train:  # deterministic eval windows
            rng = np.random.default_rng(seed)
            self.fixed = [(i, int(rng.integers(min_len, max_len + 1)))
                          for i in range(len(self.seqs))]
            self.fixed = [(i, min(N, self.lengths[i]),
                           int(rng.integers(0, max(1, self.lengths[i] - N + 1))))
                          for i, N in self.fixed]

    def __len__(self):
        return self.samples_per_epoch if self.train else len(self.fixed)

    def _load_seq(self, i):
        if i not in self._seq:
            self._seq[i] = dict(np.load(
                os.path.join(self.seq_dir, self.seqs[i] + ".npz"), allow_pickle=True))
        return self._seq[i]

    def _load_audio(self, music):
        if music not in self._audio:
            self._audio[music] = np.load(os.path.join(self.audio_dir, music + ".npy"))
        return self._audio[music]

    def __getitem__(self, idx):
        if self.train:
            rng = np.random
            i = rng.randint(len(self.seqs))
            N = int(rng.randint(self.min_len, self.max_len + 1))
            N = min(N, self.lengths[i])
            start = int(rng.randint(0, self.lengths[i] - N + 1))
        else:
            i, N, start = self.fixed[idx]

        d = self._load_seq(i)
        sl = slice(start, start + N)
        rot6d, root, joints = canonicalise(
            d["rot6d"][sl], d["root"][sl], d["joints"][sl])
        contacts = d["contacts"][sl]

        x = np.concatenate([contacts, root, rot6d], axis=1).astype(np.float32)

        audio_full = self._load_audio(str(d["music"]))
        a = np.zeros((N, audio_full.shape[1]), np.float32)
        n = max(0, min(N, len(audio_full) - start))
        if n > 0:
            a[:n] = audio_full[start:start + n]

        style_raw = style_value(joints[1:-1] if N > 2 else joints)
        style = (kinematics.to_percentile(style_raw, self.stats["style_q"])
                 if self.stats is not None
                 else np.zeros(len(kinematics.NAMES), np.float32))

        T = self.max_len
        out = {
            "x": np.zeros((T, motion.DIM), np.float32),
            "audio": np.zeros((T, audio_full.shape[1]), np.float32),
            "valid": np.zeros(T, np.float32),
            "keyframe": np.zeros(T, np.float32),
        }
        out["x"][:N], out["audio"][:N] = x, a
        out["valid"][:N] = 1.0
        out["keyframe"][0] = out["keyframe"][N - 1] = 1.0

        if self.stats is not None:
            m, s = self.stats["mean"], self.stats["std"]
            out["x"] = ((out["x"] - m) / s * out["valid"][:, None]).astype(np.float32)

        out = {k: torch.from_numpy(v) for k, v in out.items()}
        out["style"] = torch.tensor(np.asarray(style, np.float32))
        out["length"] = torch.tensor(N, dtype=torch.long)
        return out


def load_split(path):
    return [l.strip() for l in open(path) if l.strip()]
