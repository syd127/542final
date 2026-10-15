"""Minimal SMPL forward kinematics for AIST++ (joints only, no mesh).

AIST++ is fit with the neutral model at zero betas, so the rest-pose joint
locations are a single constant stored in the model file as ``J``. That plus the
kinematic tree is all FK needs; the mesh template, blend shapes and skinning
weights are only required for vertices.

The model pickle is a Python-2 chumpy pickle, so it is read with an unpickler
that allows only numpy/scipy reconstruction and stubs chumpy out.
"""
import pickle

import numpy as np

# COCO-17 index -> SMPL-24 index, for the joints the two conventions share.
COCO_TO_SMPL = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
                11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}

_ALLOWED = {
    "numpy", "numpy.core.multiarray", "numpy._core.multiarray",
    "scipy.sparse", "scipy.sparse.csc", "scipy.sparse._csc",
    "copy_reg", "copyreg", "__builtin__", "builtins",
}


class _ChumpyStub:
    """Placeholder for chumpy arrays, which SMPL only needs for shape blending."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _ChumpyStub
        root = module.split(".")[0]
        if module in _ALLOWED or root in {"numpy", "scipy"}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_skeleton(model_path):
    """Return (rest_joints [24,3] float64, parents [24] int, -1 at the root)."""
    with open(model_path, "rb") as fh:
        model = _SafeUnpickler(fh, encoding="latin1").load()
    rest = np.asarray(model["J"], np.float64)
    parents = np.asarray(model["kintree_table"][0], np.int64).copy()
    parents[0] = -1  # stored as a sentinel (uint32 max) in the release
    return rest, parents


def rodrigues(axis_angle):
    """Axis-angle [..., 3] -> rotation matrices [..., 3, 3]."""
    aa = np.asarray(axis_angle, np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 0)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def forward_kinematics(poses, trans, rest, parents, scaling=1.0):
    """SMPL joint positions in AIST++ world units.

    poses   [T, 72] axis-angle, poses[:, :3] is the root orientation
    trans   [T, 3]  root translation, already in world units
    scaling scalar  AIST++ ``smpl_scaling``

    Returns [T, 24, 3]. Mirrors the AIST++ API convention, where the SMPL
    output is scaled and the stored translation is then added in world units.
    """
    poses = np.asarray(poses, np.float64).reshape(len(poses), -1, 3)
    R = rodrigues(poses)                                   # [T, 24, 3, 3]
    T, J = R.shape[0], R.shape[1]

    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]                       # bone vectors

    glob_R = np.empty((T, J, 3, 3))
    glob_t = np.empty((T, J, 3))
    glob_R[:, 0] = R[:, 0]
    glob_t[:, 0] = offsets[0]
    for j in range(1, J):
        p = parents[j]
        glob_R[:, j] = glob_R[:, p] @ R[:, j]
        glob_t[:, j] = glob_t[:, p] + (glob_R[:, p] @ offsets[j])

    return glob_t * scaling + np.asarray(trans, np.float64)[:, None, :]


def estimate_floor(joints, foot_joints=(7, 8, 10, 11), percentile=1.0, up_axis=1):
    """Floor height, as a low percentile of foot height over the sequence."""
    return float(np.percentile(joints[:, list(foot_joints), up_axis], percentile))
