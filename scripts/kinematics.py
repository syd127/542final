"""Kinematic style descriptors.

Single place where the style dials are defined. Everything else in the pipeline
(normalisation stats, dataset, training, sampling sweeps) reads NAMES and calls
descriptors(), so changing the dial set only means editing this file.

FILTERING
---------
Measured on this dataset (scripts/spectral_analysis.py): real AIST++ motion
carries 99.9% of its position power below 4 Hz and only 0.019% above 6 Hz, while
generated motion carries 20-300x more high-frequency power than the data's own
noise floor -- roughly 30 dB of broadband jitter from ~4 Hz upward. Because
velocity, acceleration and jerk weight the spectrum by w^2, w^4, w^6, any
descriptor built on unfiltered finite differences measures that jitter rather
than the choreography: unfiltered, generated clips score 3.98x their ground
truth on mean acceleration, falling to 1.35x at a 6 Hz cutoff.

So positions are band-limited ONCE with a zero-phase 4th-order Butterworth, and
every nonlinear step (norms, ratios, means) happens afterwards. Zero-phase
matters because a causal filter would shift the signal in time and misalign
derivatives across joints. 6 Hz is the standard biomechanics choice and a
residual analysis of these spectra puts the knee at 5-7 Hz (Winter,
*Biomechanics and Motor Control of Human Movement*, 4th ed., ch. 3).

DIALS
-----
Five dials, each a plain kinematic statistic chosen to be robust to that jitter.
The perceptual vocabulary is Laban Movement Analysis (Effort / Shape), but no
numerical LMA mapping is canonical -- these are LMA-*inspired* proxies, not LMA
factors, and should be described that way.

Deliberately rejected: SPARC and log dimensionless jerk. Both are standard
smoothness measures (Balasubramanian et al., J. NeuroEng. Rehabil. 12:112, 2015),
but SPARC is unusable on the short windows we train on -- rank correlation
between a 15-frame estimate and the full-clip value is ~0, since its frequency
resolution collapses -- and LDLJ is strongly noise-sensitive and confounded with
articulation energy. Both remain useful as *evaluation* metrics on long clips.

Also rejected: a tempo dial. It is computable (Hjorth mobility of velocity) and
informative, but it carries a ~10% upward bias from residual jitter and largely
duplicates information the music conditioning already supplies -- which would
muddy the central experiment of whether the music is driving the motion.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import ndtri

FPS = 30
CUTOFF_HZ = 6.0
FILTER_ORDER = 4

# Standing height of the neutral SMPL rest skeleton (pelvis->ankle plus
# pelvis->head), metres. One shared skeleton here, so dividing by it is a
# constant; it is kept in the definitions so the dials transfer to other data.
BODY_HEIGHT = 1.4998

# AIST++ is fit from COCO-17, which has no toe or hand landmarks, so these
# joints' ROTATIONS sit frozen at identity. Their positions still move, driven
# by the parent chain, so they are excluded from articulation sums but their
# positions remain valid for reach.
FROZEN_ROT_JOINTS = (7, 8, 10, 11, 20, 21, 22, 23)
LIVE_JOINTS = tuple(j for j in range(24) if j not in FROZEN_ROT_JOINTS)
END_EFFECTORS = (20, 21, 7, 8)          # wrists, ankles
UPPER = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19)
LOWER = (1, 2, 4, 5)
ROOT, UP = 0, 1

NAMES = ["energy", "travel", "level", "reach", "updown"]


def lowpass(x, cutoff=CUTOFF_HZ, fps=FPS, order=FILTER_ORDER, axis=0):
    """Zero-phase Butterworth low-pass on positions."""
    x = np.asarray(x, np.float64)
    if cutoff is None or cutoff >= fps / 2:
        return x
    n = x.shape[axis]
    sos = butter(order, cutoff / (fps / 2), btype="low", output="sos")
    padlen = min(3 * (2 * len(sos) + 1), n - 1)
    if n < 9:
        return x
    return sosfiltfilt(sos, x, axis=axis, padlen=padlen)


def _yaw_canonical(p):
    """Per-frame: subtract the pelvis and rotate out its heading.

    Removes global translation and turning, so a pirouette does not read as limb
    articulation and travelling does not inflate articulation energy.
    """
    q = p - p[:, ROOT:ROOT + 1, :]
    across = q[:, 2] - q[:, 1]                       # right hip -> left hip
    yaw = np.arctan2(-across[:, 2], across[:, 0])    # heading from hip axis
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, z = q[..., 0].copy(), q[..., 2].copy()
    q[..., 0] = c[:, None] * x + s[:, None] * z
    q[..., 2] = -s[:, None] * x + c[:, None] * z
    return q


def descriptors(joints, fps=FPS):
    """[T,J,3] joint positions in metres, floor at y=0 -> raw dial vector [5]."""
    joints = np.asarray(joints, np.float64)
    if len(joints) < 9:
        return np.zeros(len(NAMES), np.float64)

    p = lowpass(joints, fps=fps)
    q = _yaw_canonical(p)
    dt = 1.0 / fps
    qd = np.gradient(q, dt, axis=0)
    L = BODY_HEIGHT

    # D1 energy: RMS body-relative speed of the live joints [body-heights/s]
    energy = np.sqrt((qd[:, list(LIVE_JOINTS)] ** 2).sum(-1).mean()) / L

    # D2 travel: mean horizontal pelvis speed [body-heights/s]
    rd = np.gradient(p[:, ROOT], dt, axis=0)
    travel = np.linalg.norm(rd[:, [0, 2]], axis=-1).mean() / L

    # D3 level: mean pelvis height [dimensionless, ~0.6 standing]
    level = p[:, ROOT, UP].mean() / L

    # D4 reach: mean end-effector distance from the pelvis [body-heights]
    reach = np.linalg.norm(q[:, list(END_EFFECTORS)], axis=-1).mean() / L

    # D5 updown: upper- vs lower-body share of kinetic energy, in [-1, 1].
    # The eps floor is tied to overall energy so that a nearly static clip does
    # not produce a ratio of two small noisy numbers.
    pu = (qd[:, list(UPPER)] ** 2).sum(-1).mean()
    pl = (qd[:, list(LOWER)] ** 2).sum(-1).mean()
    updown = (pu - pl) / (pu + pl + 1e-3 * (energy * L) ** 2 + 1e-12)

    return np.array([energy, travel, level, reach, updown], np.float64)


def to_percentile(raw, quantiles):
    """Raw dials -> conditioning values in [0,1] via quantile-normal mapping.

    Rank-normalising to uniform is robust to the heavy tails these statistics
    have, but it compresses the extremes into the same spacing as the bulk.
    Mapping rank through the normal inverse CDF spreads the tails back out, and
    the result is rescaled to [0,1] for the model's Fourier embedding.
    """
    raw, out = np.atleast_1d(raw), []
    for k in range(len(raw)):
        pct = np.searchsorted(quantiles[k], raw[k]) / 100.0
        z = float(np.clip(ndtri(np.clip(pct, 0.005, 0.995)), -3.0, 3.0))
        out.append((z + 3.0) / 6.0)
    return np.array(out, np.float32)
