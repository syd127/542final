"""Rotation conversions, differentiable SMPL FK, and the 151-d motion representation.

Representation, per frame (matching EDGE so numbers stay comparable):
    [0:4]    foot contact logits  (L-ankle, L-toe, R-ankle, R-toe)
    [4:7]    root translation, metres, floor-aligned
    [7:151]  24 joints x 6D rotation

6D rotation follows the pytorch3d convention (first two rows of the matrix).
"""
import numpy as np
import torch
import torch.nn.functional as F

NJOINTS, DIM = 24, 151
FOOT_JOINTS = (7, 10, 8, 11)  # L ankle, L toe, R ankle, R toe
UP = 1  # y is vertical in SMPL


def axis_angle_to_matrix(aa):
    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    k = aa / theta.clamp(min=1e-8)
    K = torch.zeros(aa.shape[:-1] + (3, 3), dtype=aa.dtype, device=aa.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
    return eye + s * K + (1 - c) * (K @ K)


def matrix_to_rotation_6d(m):
    return m[..., :2, :].reshape(m.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def forward_kinematics(rotmats, root_trans, rest, parents):
    """rotmats [B,T,24,3,3], root_trans [B,T,3] -> joints [B,T,24,3] in metres."""
    offsets = rest.clone()
    offsets[1:] -= rest[parents[1:]]

    gR = [rotmats[:, :, 0]]
    gt = [offsets[0].expand(rotmats.shape[0], rotmats.shape[1], 3)]
    for j in range(1, NJOINTS):
        p = parents[j]
        gR.append(gR[p] @ rotmats[:, :, j])
        gt.append(gt[p] + (gR[p] @ offsets[j]))
    joints = torch.stack(gt, dim=2)
    return joints + root_trans[:, :, None, :]


def foot_contacts(joints, height_thresh=0.05, speed_thresh=0.20, fps=30):
    """Binary contact labels [T,4] from floor-aligned joints [T,24,3] in metres."""
    feet = joints[:, list(FOOT_JOINTS)]
    height = feet[..., UP]
    speed = np.zeros(feet.shape[:2], np.float32)
    speed[1:] = np.linalg.norm(np.diff(feet, axis=0), axis=-1) * fps
    speed[0] = speed[1]
    return ((height < height_thresh) & (speed < speed_thresh)).astype(np.float32)


def facing_yaw(joints):
    """Yaw angle (radians about the up axis) that the body faces, from one frame [24,3]."""
    across = joints[2] - joints[1]           # right hip -> left hip
    fwd = np.cross(across, np.array([0.0, 1.0, 0.0]))
    fwd[UP] = 0.0
    n = np.linalg.norm(fwd)
    if n < 1e-8:
        return 0.0
    fwd = fwd / n
    return float(np.arctan2(fwd[0], fwd[2]))  # angle from +z


def yaw_matrix(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], np.float32)
