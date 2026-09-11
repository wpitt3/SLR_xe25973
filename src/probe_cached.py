#!/usr/bin/env python3

import argparse, os, glob, yaml, time, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from types import SimpleNamespace
from torch.utils.data import DataLoader, Dataset
from datasets import S2T_Dataset
from Tokenizer import GlossTokenizer_S2G
from recognition import DSTA, ctc_decode_func
from metrics import wer_list

DSTA_ANCHORS = ('left_input_map', 'right_input_map', 'face_input_map',
                'body_input_map', 'left_graph_layers', 'right_graph_layers',
                'face_graph_layers', 'body_graph_layers')


def standardise_key_names(sd, enc_keys):
    if any(k in enc_keys for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        pos = min((k.find(a) for a in DSTA_ANCHORS if a in k), default=-1)
        if pos != -1:
            out[k[pos:]] = v
    return out


def build_encoder(net_cfg, encoder_arg, dev):
    enc = DSTA(cfg=net_cfg, args=SimpleNamespace(), num_channel=3).to(dev)
    if encoder_arg != "random":
        raw = torch.load(encoder_arg, map_location='cpu')
        sd = raw['model'] if isinstance(raw, dict) and 'model' in raw else raw
        sd = standardise_key_names(sd, set(enc.state_dict()))
        ret = enc.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(ret.unexpected_keys)

        print(f"enc {encoder_arg}: loaded {loaded} | missing {len(ret.missing_keys)} "
              f"unexpected {len(ret.unexpected_keys)}")
    else:
        print("enc random-init encoder")
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc


def make_loader(cfg, tok, split, phase, bs):
    ds = S2T_Dataset(path=cfg['data'][f'{split}_label_path'], tokenizer=tok,
                     config=cfg, args=SimpleNamespace(), phase=phase,
                     training_refurbish=False)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      collate_fn=ds.collate_fn, num_workers=2)


@torch.no_grad()
def encode_once(enc, loader, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    n = 0
    for src in loader:
        feat, *_ = enc(src)
        L = src['new_src_lengths']
        for i in range(feat.shape[0]):
            Li = int(L[i])
            np.savez(os.path.join(cache_dir, f"{n:06d}.npz"),
                     feat=feat[i, :Li].cpu().half().numpy(),
                     length=Li, gloss=src['gloss'][i])
            n += 1
    print(f"cache {n} samples -> {cache_dir}")
    return n


class CachedFeats(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=True)
        return d['feat'].astype(np.float32), int(d['length']), str(d['gloss'])


def make_cached_collate(tok):
    def collate(batch):
        T = max(f.shape[0] for f, _, _ in batch)
        X = torch.zeros(len(batch), T, 1024)
        lens, gl = [], []
        for i, (f, L, g) in enumerate(batch):
            X[i, :f.shape[0]] = torch.from_numpy(f)
            lens.append(L); gl.append(g)
        return X, torch.tensor(lens), tok(gl), gl
    return collate


@torch.no_grad()
def evaluate(probe, dev, dev_dl, tok):
    probe.eval(); hyps, refs = [], []
    for X, L, gi, gl in dev_dl:
        logp = F.log_softmax(probe(X.to(dev)), -1)
        dec = ctc_decode_func(
            np.concatenate((logp[:, :, 1:].cpu().numpy(),
                            logp[:, :, 0, None].cpu().numpy()), -1).transpose(1, 0, 2),
            L.cpu(), beam_size=5)
        hyps += [' '.join(h) for h in tok.convert_ids_to_tokens(dec)]
        refs += list(gl)
    probe.train()
    return wer_list(hypotheses=hyps, references=refs)['wer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/csl-daily_s2g.yaml")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--cache", default="probe_cache")
    ap.add_argument("--epochs", default=40)
    ap.add_argument("--batch-size", default=16)
    ap.add_argument("--lr", default=0.001)
    ap.add_argument("--seed", default=0)
    ap.add_argument("--recache", action="store_true", help="force re-featurize")
    args = ap.parse_args()

    import random as _random
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); _random.seed(args.seed)
    print(f"[seed] {args.seed}")

    cfg = yaml.safe_load(open(args.config))
    net_cfg = cfg['model']['RecognitionNetwork']['DSTA-Net']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = GlossTokenizer_S2G(cfg['model']['RecognitionNetwork']['GlossTokenizer'])
    V = len(tok)

    # cache keyed by encoder AND seed: different seeds give different augmentation
    # draws (and, for --encoder random, different random encoders), so repeats
    # are genuinely independent rather than reusing one cached feature set.
    tag = os.path.basename(args.encoder).replace('.', '_') + f"_seed{args.seed}"
    cdir_tr = os.path.join(args.cache, tag, 'train')
    cdir_dv = os.path.join(args.cache, tag, 'dev')
    if args.recache or not os.path.isdir(cdir_tr):
        enc = build_encoder(net_cfg, args.encoder, dev)
        encode_once(enc, make_loader(cfg, tok, 'train', 'train', args.batch_size), dev, cdir_tr)
        encode_once(enc, make_loader(cfg, tok, 'dev', 'dev', args.batch_size), dev, cdir_dv)
        del enc; torch.cuda.empty_cache()
    else:
        print(f"cache reuse {cdir_tr}")

    collate = make_cached_collate(tok)
    train_dl = DataLoader(CachedFeats(cdir_tr), batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=2)
    dev_dl = DataLoader(CachedFeats(cdir_dv), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=2)

    probe = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                          nn.Dropout(0.1), nn.Linear(512, V)).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum')


    best, t0 = 1e9, time.time()
    for ep in range(args.epochs):
        run = None
        for X, L, gi, gl in train_dl:
            logp = F.log_softmax(probe(X.to(dev)), -1).permute(1, 0, 2)   # T,B,V
            loss = ctc(logp, gi['gloss_labels'].to(dev), L.to(dev),
                       gi['gls_lengths'].to(dev)) / X.shape[0]
            opt.zero_grad(); loss.backward(); opt.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        w = evaluate(probe, dev, dev_dl, tok)
        best = min(best, w)
        print(f"ep {ep+1}/{args.epochs} ctc {run:.3f} DEV WER {w:.2f} "f"(best {best:.2f}) {time.time()-t0:.0f}s")
    print(f"\nPROBE {args.encoder} seed{args.seed}: best dev WER {best:.2f}")


if __name__ == "__main__":
    main()
