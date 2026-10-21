"""Train RhythmFlow with conditional flow matching.

Linear (optimal-transport) path: x_t = (1-t)*x_0 + t*x_1 with target velocity
x_1 - x_0. Auxiliary kinematic losses are applied to the clean motion recovered
from the predicted velocity, x1_hat = x_t + (1-t)*v, following FlowerDance.

Jobs on this cluster are capped at 8 hours, so training checkpoints regularly
and exits cleanly at --max-hours; rerunning the same command resumes.
"""
import argparse, os, sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import motion
from dataset import AISTPPInbetween, load_split
import kinematics
from model import RhythmFlow
from smpl_fk import load_skeleton


def masked_mse(a, b, valid):
    d = (a - b) ** 2
    if d.dim() == 4:
        d = d.mean(-1)
    d = d.mean(-1) if d.dim() == 3 else d
    return (d * valid).sum() / valid.sum().clamp(min=1)


def decode(x, mean, std, rest, parents):
    """Normalised features -> (contacts, joints) in metres."""
    x = x * std + mean
    contacts, root, rot6d = x[..., :4], x[..., 4:7], x[..., 7:]
    B, T = x.shape[:2]
    R = motion.rotation_6d_to_matrix(rot6d.reshape(B, T, motion.NJOINTS, 6))
    return contacts, motion.forward_kinematics(R, root, rest, parents)


def build_loaders(args, stats):
    sp = args.split_dir
    tr_names = load_split(os.path.join(sp, "crossmodal_train.txt"))
    va_names = load_split(os.path.join(sp, "crossmodal_val.txt"))
    common = dict(seq_dir=args.seq_dir, audio_dir=args.audio_dir,
                  min_len=args.min_len, max_len=args.max_len, stats=stats)

    if args.overfit:
        ds = AISTPPInbetween(names=tr_names, train=False,
                             samples_per_epoch=0, **common)
        ds.fixed = ds.fixed[:args.overfit]
        train_ds = val_ds = ds
    else:
        train_ds = AISTPPInbetween(names=tr_names, train=True,
                                   samples_per_epoch=args.samples_per_epoch, **common)
        val_ds = AISTPPInbetween(names=va_names, train=False,
                                 samples_per_epoch=0, **common)

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=0, pin_memory=True)
    return tl, vl


def losses(model, batch, args, mean, std, rest, parents, train=True):
    x1 = batch["x"]
    valid, kf = batch["valid"], batch["keyframe"]
    B = x1.shape[0]
    obs = x1 * kf[..., None]

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
    target = x1 - x0

    if train:
        drop_a = torch.rand(B, device=x1.device) < args.cfg_drop
        # Dials are dropped independently so the model learns every subset, which
        # is what lets one dial be swept at inference while the rest sit neutral.
        # A whole-vector drop is also forced sometimes, otherwise the fully
        # unconditional branch that guidance extrapolates from is almost never
        # trained (independent drops would co-occur with probability cfg_drop^K).
        style = batch["style"]
        K = style.shape[1] if style.dim() > 1 else 1
        drop_s = torch.rand(B, K, device=x1.device) < args.cfg_drop
        drop_s |= (torch.rand(B, 1, device=x1.device) < args.cfg_drop)
    else:
        K = batch["style"].shape[1] if batch["style"].dim() > 1 else 1
        drop_a = torch.zeros(B, dtype=torch.bool, device=x1.device)
        drop_s = torch.zeros(B, K, dtype=torch.bool, device=x1.device)

    v = model(xt, t, obs, kf, batch["audio"], batch["style"], valid, drop_a, drop_s)

    out = {"v": masked_mse(v, target, valid)}
    x1_hat = xt + (1 - t)[:, None, None] * v

    c_hat, j_hat = decode(x1_hat, mean, std, rest, parents)
    c_gt, j_gt = decode(x1, mean, std, rest, parents)

    out["pos"] = masked_mse(j_hat, j_gt, valid)
    vm = valid[:, 1:] * valid[:, :-1]
    out["vel"] = masked_mse(j_hat[:, 1:] - j_hat[:, :-1],
                            j_gt[:, 1:] - j_gt[:, :-1], vm)
    out["contact"] = masked_mse(c_hat, c_gt, valid)

    feet = list(motion.FOOT_JOINTS)
    fv = (j_hat[:, 1:, feet] - j_hat[:, :-1, feet]).norm(dim=-1)
    out["skate"] = ((fv * c_gt[:, 1:]).mean(-1) * vm).sum() / vm.sum().clamp(min=1)

    out["total"] = (out["v"] + args.w_pos * out["pos"] + args.w_vel * out["vel"]
                    + args.w_contact * out["contact"] + args.w_skate * out["skate"])
    # endpoint error in metres, diagnostic only
    with torch.no_grad():
        ends = kf.bool()
        out["endpoint_m"] = (j_hat - j_gt).norm(dim=-1).mean(-1)[ends].mean()
    return out


@torch.no_grad()
def sample(model, batch, args, steps=20, guidance=1.0):
    """Euler integration of the learned velocity field from noise to motion."""
    x1, valid, kf = batch["x"], batch["valid"], batch["keyframe"]
    obs = x1 * kf[..., None]
    x = torch.randn_like(x1)
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        v = model(x, t, obs, kf, batch["audio"], batch["style"], valid)
        if guidance != 1.0:
            vu = model(x, t, obs, kf, batch["audio"], batch["style"], valid,
                       drop_audio=torch.ones(x.shape[0], dtype=torch.bool, device=x.device),
                       drop_style=torch.ones(x.shape[0], dtype=torch.bool, device=x.device))
            v = vu + guidance * (v - vu)
        x = x + v / steps
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--smpl-model", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--nlayers", type=int, default=8)
    p.add_argument("--min-len", type=int, default=15)
    p.add_argument("--max-len", type=int, default=150)
    p.add_argument("--samples-per-epoch", type=int, default=20000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cfg-drop", type=float, default=0.1)
    p.add_argument("--w-pos", type=float, default=0.5)
    p.add_argument("--w-vel", type=float, default=0.5)
    p.add_argument("--w-contact", type=float, default=0.1)
    p.add_argument("--w-skate", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--max-hours", type=float, default=7.5)
    p.add_argument("--deadline", type=float, default=0.0,
                   help="absolute epoch seconds to stop by; overrides --max-hours. "
                        "Set from the job start time so setup counts against the budget.")
    p.add_argument("--sample-steps", type=int, default=20)
    p.add_argument("--overfit", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""),
          flush=True)

    st = np.load(args.stats)
    stats = {"mean": st["mean"], "std": st["std"], "style_q": st["style_q"]}
    mean = torch.tensor(st["mean"], device=dev)
    std = torch.tensor(st["std"], device=dev)

    rest_np, par_np = load_skeleton(args.smpl_model)
    rest = torch.tensor(rest_np, dtype=torch.float32, device=dev)
    parents = torch.tensor(par_np, dtype=torch.long, device=dev)

    tl, vl = build_loaders(args, stats)
    model = RhythmFlow(d_model=args.d_model, nlayers=args.nlayers,
                       max_len=args.max_len,
                       n_style=len(kinematics.NAMES)).to(dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.1f} M | "
          f"train clips/epoch {len(tl.dataset)}", flush=True)

    step = 0
    ckpt = os.path.join(args.out_dir, "last.pt")
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        ema = {k: v.to(dev) for k, v in s["ema"].items()}; step = s["step"]
        print(f"resumed from step {step}", flush=True)

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema, "step": step, "args": vars(args)}, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)

    t0, run = time.time(), True
    while run and step < args.steps:
        for batch in tl:
            if step >= args.steps or not run:
                break
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            model.train()
            L = losses(model, batch, args, mean, std, rest, parents, train=True)

            for g in opt.param_groups:
                g["lr"] = args.lr * min(1.0, (step + 1) / args.warmup)
            opt.zero_grad(set_to_none=True)
            L["total"].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema) \
                        if v.dtype.is_floating_point else ema[k].copy_(v)
            step += 1

            if step % args.log_every == 0:
                el = (time.time() - t0) / 3600
                print(f"step {step:7d} | total {L['total']:.4f} v {L['v']:.4f} "
                      f"pos {L['pos']:.4f} vel {L['vel']:.4f} con {L['contact']:.4f} "
                      f"skate {L['skate']:.4f} | endpt {L['endpoint_m']*100:.1f}cm "
                      f"| gn {gn:.2f} | {el:.2f}h", flush=True)

            if step % args.val_every == 0:
                model.eval()
                with torch.no_grad():
                    vs, ep_s, rec_s = [], [], []
                    for vb in vl:
                        vb = {k: v.to(dev) for k, v in vb.items()}
                        vs.append(losses(model, vb, args, mean, std, rest, parents,
                                         train=False))
                        # What actually matters: error after integrating the ODE,
                        # not the training-time estimate averaged over all noise levels.
                        xs = sample(model, vb, args, steps=args.sample_steps)
                        _, j_s = decode(xs, mean, std, rest, parents)
                        _, j_gt = decode(vb["x"], mean, std, rest, parents)
                        ends = vb["keyframe"].bool()
                        ep_s.append((j_s - j_gt).norm(dim=-1).mean(-1)[ends].mean().item())
                        rec_s.append(masked_mse(j_s, j_gt, vb["valid"]).item())
                    if vs:
                        print(f"  val | total {np.mean([x['total'].item() for x in vs]):.4f} "
                              f"v {np.mean([x['v'].item() for x in vs]):.4f} "
                              f"| sampled: endpt {np.mean(ep_s)*100:.1f}cm "
                              f"recon {np.mean(rec_s):.4f}", flush=True)

            if step % args.ckpt_every == 0:
                save()
            over = (time.time() > args.deadline if args.deadline > 0
                    else (time.time() - t0) / 3600 > args.max_hours)
            if over:
                print("time budget reached; checkpointing and exiting", flush=True)
                run = False

    save()
    print(f"stopped at step {step}", flush=True)


if __name__ == "__main__":
    main()
