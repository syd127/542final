"""Kinematic style descriptors.

Single place where the style dials are defined. Everything else in the pipeline
(normalisation stats, dataset, training, sampling sweeps) reads NAMES and calls
descriptors(), so changing the dial set only means editing this file.

Filtering rationale, measured on this dataset (scripts/spectral_analysis.py):
real AIST++ motion carries 99.9% of its power below 4 Hz and only 0.019% above
6 Hz, while generated motion carries 20-300x more high-frequency power than the
data's own noise floor. Any descriptor built on raw finite differences therefore
measures jitter as much as choreography: unfiltered, generated clips score 3.98x
their ground truth on mean acceleration; at a 6 Hz cutoff that falls to 1.35x,
and ground truth itself loses almost nothing.

A 6 Hz cutoff is deliberate rather than maximally debiasing. Filtering at 4 Hz
would drive the ratio to 1.05x, but that hides a genuine model defect behind the
metric instead of measuring it. 6 Hz keeps essentially all real signal.
"""
import numpy as np
from scipy.signal import butter, filtfilt

FPS = 30
CUTOFF_HZ = 6.0
FILTER_ORDER = 4

# Distal SMPL joints are frozen at identity in AIST++ (fit from COCO-17, which
# has no toe or hand landmarks), so their *rotations* carry no information.
# Their positions still move, driven by the parent chain, and remain usable.
FROZEN_ROT_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)
ROOT = 0


def lowpass(x, cutoff=CUTOFF_HZ, fps=FPS, order=FILTER_ORDER, axis=0):
    """Zero-phase Butterworth low-pass. Zero-phase matters: a causal filter would
    shift the signal in time and corrupt any derivative-based descriptor."""
    if cutoff is None or cutoff >= fps / 2:
        return np.asarray(x, np.float64)
    x = np.asarray(x, np.float64)
    # filtfilt needs more samples than its transient; fall back for short clips.
    if x.shape[axis] <= 3 * (2 * order + 1):
        return x
    b, a = butter(order, cutoff / (fps / 2), btype="low")
    return filtfilt(b, a, x, axis=axis)


def derivatives(joints, fps=FPS):
    """Filtered position, velocity, acceleration, jerk for [T,J,3] in metres."""
    p = lowpass(joints, fps=fps)
    v = np.gradient(p, 1.0 / fps, axis=0)
    a = np.gradient(v, 1.0 / fps, axis=0)
    j = np.gradient(a, 1.0 / fps, axis=0)
    return p, v, a, j


# --- dial definitions ------------------------------------------------------
# Filled in from the descriptor design; NAMES fixes the vector order.
NAMES: list[str] = ["accel"]


def descriptors(joints, fps=FPS):
    """[T,J,3] joint positions in metres -> raw descriptor vector [len(NAMES)]."""
    joints = np.asarray(joints, np.float64)
    if len(joints) < 5:
        return np.zeros(len(NAMES), np.float64)
    _, _, a, _ = derivatives(joints, fps)
    return np.array([np.linalg.norm(a, axis=-1).mean()], np.float64)


def to_percentile(raw, quantiles):
    """Map raw descriptors to [0,1] against per-dial training quantile tables.

    quantiles: [K,101]. Rank normalisation is used rather than a z-score because
    these statistics are heavy-tailed and bounded below by zero, so a z-score
    would leave the dial's usable range concentrated in a narrow band.
    """
    raw = np.atleast_1d(raw)
    return np.array([np.searchsorted(quantiles[k], raw[k]) / 100.0
                     for k in range(len(raw))], np.float32)
