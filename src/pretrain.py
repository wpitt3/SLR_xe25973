#!/usr/bin/env python3
import argparse, os, time, yaml, random
import numpy as np, torch
from types import SimpleNamespace
from torch.utils.data import IterableDataset, DataLoader
from load_shards import iter_shards, shard_paths
from pretrain_model import MaskedKeypointPretrainer

CSL_NEWS_FRACTION = 0.217

# Follow datasets.augment_preprocess_inputs
# match pretraining to fine-tuning
def normalise(pose01):
    out = np.empty_like(pose01, dtype=np.float32)
    out[..., 0] = 2.0 * pose01[..., 0] - 1.0
    out[..., 1] = 1.0 - 2.0 * pose01[..., 1]
    return out

# Read shards as buffered stream
class ShardStream(IterableDataset):
    def __init__(self, shards, max_T=250, shuffle_buf=500):
        self.shards, self.max_T, self.buf = shards, max_T, shuffle_buf

    def __iter__(self):
        buf = []
        for key, pose, conf, _ in iter_shards(None, shards=self.shards):
            pose = pose.astype(np.float32)
            conf = conf.astype(np.float32)
            if pose.shape[0] > self.max_T:
                s = random.randint(0, pose.shape[0] - self.max_T)
                pose, conf = pose[s:s + self.max_T], conf[s:s + self.max_T]
            pose = normalise(pose)
            buf.append((pose, conf))
            if len(buf) >= self.buf:
                random.shuffle(buf)
                while buf:
                    yield buf.pop()
        random.shuffle(buf)
        while buf:
            yield buf.pop()


# pads batch so they are all the same length
def collate(batch):
    T = max(p.shape[0] for p, _ in batch)
    T = ((T + 3) // 4) * 4
    B = len(batch)
    kp = torch.zeros(B, 3, T, 133)
    cf = torch.zeros(B, T, 133)
    for i, (p, c) in enumerate(batch):
        t = p.shape[0]
        kp[i, :2, :t] = torch.from_numpy(p).permute(2, 0, 1)
        kp[i, 2, :t] = torch.from_numpy(c)
        cf[i, :t] = torch.from_numpy(c)
    return kp, cf


def load_net_cfg(config_path):
    cfg = yaml.safe_load(open(config_path))
    return cfg['model']['RecognitionNetwork']['DSTA-Net']


def select_fraction_of_shards(all_shards, frac):
    # shards only contain 0.217 of CSL-News
    n = max(1, round(frac / CSL_NEWS_FRACTION * len(all_shards)))
    return all_shards[:min(n, len(all_shards))]

# calculate validation loss
@torch.no_grad()
def evaluate(model, val_shards, dev, batch_size, max_T, n_batches=15):
    model.eval()
    dl = DataLoader(ShardStream(val_shards, max_T=max_T, shuffle_buf=1),
                    batch_size=batch_size, collate_fn=collate)
    tot, n = 0.0, 0
    for kp, cf in dl:
        torch.manual_seed(n)
        tot += model(kp.to(dev), cf.to(dev)).item(); n += 1
        if n >= n_batches:
            break
    model.train()
    return tot / max(n, 1)


MODEL_PARAMS = ['p_span', 'p_joint', 'p_stream', 'span_frac', 'span_min', 'span_max', 'joint_frac', 'joint_win', 'stream_win', 'jitter_std', 'jitter_frac']


def get_set_corruption_args(args):
    return {k: getattr(args, k) for k in MODEL_PARAMS if getattr(args, k) is not None}


def train_fraction(net_cfg, shards, val_shards, args, dev, tag):
    model = MaskedKeypointPretrainer(net_cfg, SimpleNamespace(), **get_set_corruption_args(args)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    dl = DataLoader(ShardStream(shards, max_T=args.max_T), batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers)
    step, run, t0, best = 0, None, time.time(), float('inf')
    ckpt = os.path.join(args.out, f"enc_{tag}.pth")
    while step < args.steps:
        for kp, cf in dl:
            kp, cf = kp.to(dev), cf.to(dev)
            loss = model(kp, cf)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            if step % args.log_every == 0:
                ips = args.batch_size * step / (time.time() - t0)
                print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                      f"(ema {run:.4f}) {ips:.1f} clip/s")
            if val_shards and step % args.eval_every == 0:
                v = evaluate(model, val_shards, dev, args.batch_size, args.max_T)
                star = "  *" if v < best else ""
                print(f"step {step} VAL {v:.4f}{star}")
                if v < best:
                    best = v
                    torch.save({'model': model.encoder.state_dict(),
                                'meta': {'tag': tag, 'step': step, 'val': v}}, ckpt)
            if step >= args.steps:
                break
    if not val_shards:
        torch.save({'model': model.encoder.state_dict(),
                    'meta': {'tag': tag, 'step': step}}, ckpt)
    print(f"done, best val {best:.4f} -> {ckpt}")

    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--val-shards", default=None)
    ap.add_argument("--config", default="configs/csl-daily_s2g.yaml")
    ap.add_argument("--out", default="pretrain_out")
    ap.add_argument("--batch-size", default=16)
    ap.add_argument("--lr", default=0.001)
    ap.add_argument("--max-T", default=250)
    ap.add_argument("--num-workers", default=2)
    ap.add_argument("--steps", default=8000)
    ap.add_argument("--fraction", default=0.20)
    ap.add_argument("--log-every", default=20)
    ap.add_argument("--eval-every", default=500)

    # Corruption config
    ap.add_argument("--p-span", default=None)
    ap.add_argument("--p-joint", default=None)
    ap.add_argument("--p-stream", default=None)
    ap.add_argument("--span-frac", default=None)
    ap.add_argument("--span-min", default=None)
    ap.add_argument("--span-max", default=None)
    ap.add_argument("--joint-frac", default=None)
    ap.add_argument("--joint-win", default=None)
    ap.add_argument("--stream-win", default=None)
    ap.add_argument("--jitter-std", default=None)
    ap.add_argument("--jitter-frac", default=None)

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net_cfg = load_net_cfg(args.config)
    all_shards = shard_paths(args.shards)
    val_shards = shard_paths(args.val_shards) if args.val_shards else None
    print(f"overrides: {get_set_corruption_args(args) or 'default'}")

    sub = select_fraction_of_shards(all_shards, args.fraction)
    tag = f"{args.fraction:.2f}"
    print(f"\n=== fraction {tag}: {len(sub)}/{len(all_shards)} shards ===")
    train_fraction(net_cfg, sub, val_shards, args, dev, tag)


if __name__ == "__main__":
    main()
