"""RhythmFlow: a conditional velocity field for music-driven motion inbetweening.

Conditioning:
  keyframes  values + binary observation mask, concatenated to the input
  music      per-frame librosa features, via cross-attention
  style      a scalar kinematic dial, embedded and added to every token

Classifier-free guidance is supported on music and style independently: music is
dropped by zeroing its features, style by swapping in a learned null embedding.
"""
import math
import torch
import torch.nn as nn


def sinusoidal(values, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=values.device) / half)
    ang = values[:, None].float() * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class RhythmFlow(nn.Module):
    def __init__(self, dim=151, audio_dim=35, d_model=512, nhead=8, nlayers=8,
                 ff=2048, dropout=0.1, max_len=150, n_style=1, style_feat=64):
        super().__init__()
        self.d_model = d_model

        self.in_proj = nn.Linear(dim * 2 + 1, d_model)   # x_t | keyframe values | mask
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.audio_pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        # Style is a vector of n_style kinematic dials, each in [0,1]. Every dial
        # gets its own Fourier embedding; the concatenation is projected to d_model.
        # Dials can be dropped individually for classifier-free guidance, so a
        # single dial can be swept at sampling time while the rest stay neutral.
        self.n_style, self.style_feat = n_style, style_feat
        self.style_mlp = nn.Sequential(
            nn.Linear(n_style * style_feat, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
        self.null_style = nn.Parameter(torch.randn(n_style, style_feat) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model, nhead, ff, dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.net = nn.TransformerDecoder(layer, nlayers)

        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x_t, t, obs, kf_mask, audio, style, valid,
                drop_audio=None, drop_style=None):
        B, T, _ = x_t.shape

        if drop_audio is not None:
            audio = audio * (~drop_audio).float()[:, None, None]

        h = self.in_proj(torch.cat([x_t, obs, kf_mask[..., None]], dim=-1))
        h = h + self.pos[:, :T]

        cond = self.t_mlp(sinusoidal(t, self.d_model))

        if style.dim() == 1:                      # tolerate a single-dial call
            style = style[:, None]
        se = sinusoidal(style.reshape(-1), self.style_feat).view(
            B, self.n_style, self.style_feat)
        if drop_style is not None:
            if drop_style.dim() == 1:             # drop every dial at once
                drop_style = drop_style[:, None].expand(B, self.n_style)
            se = torch.where(drop_style[..., None],
                             self.null_style[None].expand_as(se), se)
        s = self.style_mlp(se.reshape(B, -1))
        h = h + (cond + s)[:, None, :]

        mem = self.audio_proj(audio) + self.audio_pos[:, :T]

        pad = valid < 0.5
        h = self.net(h, mem, tgt_key_padding_mask=pad, memory_key_padding_mask=pad)
        return self.out(h) * valid[..., None]
