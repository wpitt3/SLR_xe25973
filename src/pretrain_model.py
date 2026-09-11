#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from recognition import DSTA

class Simple_Decoder(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv = nn.Conv2d(C, C, 1)
        self.out = nn.Conv2d(C, 2, 1)

    def forward(self, f, T):
        node = f.shape[-1]
        f = F.gelu(self.conv(f))
        f = F.interpolate(f, size=(T, node), mode='bilinear', align_corners=False)
        return self.out(f)


class MaskedKeypointPretrainer(nn.Module):
    STREAMS = ['left', 'right', 'face', 'body']

    def __init__(self, cfg, args,
                 p_span=0.30, p_joint=0.60, p_stream=0.10,
                 span_frac=0.15, span_min=6, span_max=16,
                 joint_frac=0.20, joint_win=0.40, stream_win=0.40,
                 lam_vel=0.5, tau=0.3, delta=0.05,
                 jitter_std=0.0, jitter_frac=0.0,
                 stream_w=None):
        super().__init__()
        self.cfg = cfg
        self.p_span, self.p_joint, self.p_stream = p_span, p_joint, p_stream
        self.span_frac, self.span_min, self.span_max = span_frac, span_min, span_max
        self.joint_frac, self.joint_win, self.stream_win = joint_frac, joint_win, stream_win
        self.lam_vel, self.tau, self.delta = lam_vel, tau, delta
        self.jitter_std, self.jitter_frac = jitter_std, jitter_frac
        self.stream_w = stream_w or {'left': 1.5, 'right': 1.5, 'face': 0.5, 'body': 1.0}

        self.encoder = DSTA(cfg=cfg, args=args, num_channel=3, mode="SLR")
        self.dec = nn.ModuleDict({s: Simple_Decoder(self.encoder.out_channels) for s in self.STREAMS})

    # Follow DSTA.forward up to before pooling
    def _encode_streams(self, x):
        enc = self.encoder
        left = enc.left_input_map(x[:, :, :, self.cfg['left']])
        right = enc.right_input_map(x[:, :, :, self.cfg['right']])
        face = enc.face_input_map(x[:, :, :, self.cfg['face']])
        body = enc.body_input_map(x[:, :, :, self.cfg['body']])
        for m in enc.face_graph_layers:
            face = m(face)
        for m in enc.left_graph_layers:
            left = m(left)
        for m in enc.right_graph_layers:
            right = m(right)
        for m in enc.body_graph_layers:
            body = m(body)
        return {'left': left, 'right': right, 'face': face, 'body': body}

    # Build mask for each clip in batch
    # Uses full 133 keypoints when should use 79
    def build_mask(self, B, T, V, device, force=None):
        mask = torch.zeros(B, T, V, dtype=torch.bool, device=device)
        for b in range(B):
            r = force or self._pick_mode()
            if r == 'span':
                covered, target = 0, max(1, int(self.span_frac * T))
                while covered < target:
                    L = random.randint(self.span_min, self.span_max)
                    s = random.randint(0, max(0, T - L))
                    mask[b, s:s + L, :] = True
                    covered += L
            elif r == 'joint':
                s = random.choice(self.STREAMS)
                idx = list(self.cfg[s])
                k = max(1, int(self.joint_frac * len(idx)))
                jsel = random.sample(idx, k)
                L = max(1, int(self.joint_win * T));
                st = random.randint(0, max(0, T - L))
                mask[b, st:st + L, jsel] = True
            else: # stream
                s = random.choice(self.STREAMS)
                idx = list(self.cfg[s])
                L = max(1, int(self.stream_win * T));
                st = random.randint(0, max(0, T - L))
                mask[b, st:st + L, idx] = True
        return mask

    def _pick_mode(self):
        r = random.random()
        if r < self.p_span:
            return 'span'
        if r < self.p_span + self.p_joint:
            return 'joint'
        return 'stream'

    # apply random jitter to keypoints based upon jitter_frac of shoulder width
    def _apply_jitter(self, keypoint):
        B = keypoint.shape[0]
        pick = torch.rand(B, device=keypoint.device) < self.jitter_frac
        if not bool(pick.any()):
            return keypoint
        out = keypoint.clone()
        noise = torch.randn(B, 2, keypoint.shape[2], keypoint.shape[3],
                            device=keypoint.device) * self.jitter_std
        out[pick, :2] = out[pick, :2] + noise[pick]
        return out

    def forward(self, keypoint, conf, force_mask=None):
        B, C, T, V = keypoint.shape
        mask = self.build_mask(B, T, V, keypoint.device, force=force_mask)
        x = keypoint
        if self.training and self.jitter_std > 0 and self.jitter_frac > 0:
            x = self._apply_jitter(x)
        x = x.masked_fill(mask.unsqueeze(1), 0.0)
        feats = self._encode_streams(x)

        total, wsum = 0.0, 0.0
        for s in self.STREAMS:
            idx = self.cfg[s]
            pred = self.dec[s](feats[s], T)
            tgt = keypoint[:, :2][..., idx]
            m_s = mask[..., idx].float()
            c_s = conf[..., idx]
            w = m_s * torch.clamp((c_s - self.tau) / (1 - self.tau), 0, 1)

            pos = F.huber_loss(pred, tgt, reduction='none', delta=self.delta).sum(1)
            dp = pred[:, :, 1:] - pred[:, :, :-1]
            dt = tgt[:, :, 1:] - tgt[:, :, :-1]
            wv = torch.minimum(w[:, 1:], w[:, :-1])
            vel = F.huber_loss(dp, dt, reduction='none', delta=self.delta).sum(1)

            l = (pos * w).sum() / (w.sum() + 1e-6) \
                + self.lam_vel * (vel * wv).sum() / (wv.sum() + 1e-6)
            sw = self.stream_w[s]
            total = total + sw * l
            wsum += sw
        return total / wsum
