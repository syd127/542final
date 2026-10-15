"""Precompute the per-sequence motion arrays used for training.

For each AIST++ sequence: convert SMPL axis-angle to 6D rotations, put the root
translation in metres (dividing out the per-sequence smpl_scaling), run FK,
align the floor to y=0, and derive foot contacts.

Windowing and canonicalisation happen at sampling time, not here.
"""
import argparse, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion
from smpl_fk import load_skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--smpl-model", required=True)
    ap.add_argument("--ignore-list", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rest_np, parents_np = load_skeleton(args.smpl_model)
    rest = torch.tensor(rest_np, dtype=torch.float32)
    parents = torch.tensor(parents_np, dtype=torch.long)

    ignore = {l.strip() for l in open(args.ignore_list) if l.strip()}
    files = sorted(f for f in os.listdir(args.motion_dir) if f.endswith(".npz"))

    kept, skipped, floors = 0, 0, []
    for fn in files:
        seq = fn[:-4]
        if seq in ignore:
            skipped += 1
            continue
        d = np.load(os.path.join(args.motion_dir, fn))
        poses = torch.tensor(d["smpl_poses"], dtype=torch.float32)
        scaling = float(d["smpl_scaling"].ravel()[0])
        root = torch.tensor(d["smpl_trans"], dtype=torch.float32) / scaling  # -> metres

        rotmats = motion.axis_angle_to_matrix(poses.reshape(-1, motion.NJOINTS, 3))
        rot6d = motion.matrix_to_rotation_6d(rotmats)                       # [T,24,6]
        joints = motion.forward_kinematics(
            rotmats[None], root[None], rest, parents)[0]                    # [T,24,3]

        floor = float(np.percentile(
            joints[:, list(motion.FOOT_JOINTS), motion.UP].numpy(), 1.0))
        joints[..., motion.UP] -= floor
        root[:, motion.UP] -= floor
        floors.append(floor)

        contacts = motion.foot_contacts(joints.numpy())
        np.savez(
            os.path.join(args.out, seq + ".npz"),
            rot6d=rot6d.reshape(len(poses), -1).numpy().astype(np.float32),
            root=root.numpy().astype(np.float32),
            joints=joints.numpy().astype(np.float32),
            contacts=contacts,
            music=re.match(r".*_(m\w+?)_ch\d+$", seq).group(1),
        )
        kept += 1

    print(f"wrote {kept} sequences, skipped {skipped} in ignore_list")
    print(f"floor height: mean {np.mean(floors):.3f} m  std {np.std(floors):.3f}  "
          f"range [{np.min(floors):.3f}, {np.max(floors):.3f}]")


if __name__ == "__main__":
    main()
