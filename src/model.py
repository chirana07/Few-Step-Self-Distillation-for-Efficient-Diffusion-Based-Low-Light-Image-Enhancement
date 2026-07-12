import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from modules import (
    Swish,
    Upsample,
    Downsample,
    AttentionBlock,
    ConditionEncoder,
    ResBlockSFT,
    get_timestep_embedding,
)


class ResidualConditionedUNet(nn.Module):
    def __init__(self, use_illum_prior=False):
        super().__init__()
        if use_illum_prior:
            raise ValueError(
                "This six-channel architecture does not implement an illumination-prior input. "
                "Legacy checkpoints may contain a stale use_illum_prior metadata flag."
            )
        self.conf = Config()

        ch = self.conf.CHANNELS
        ch_mult = self.conf.CHANNEL_MULT
        time_dim = ch * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(ch, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )

        self.head = nn.Conv2d(6, ch, 3, padding=1)

        self.cond_encoder = ConditionEncoder(
            in_ch=3,
            base_ch=ch,
            ch_mult=tuple(ch_mult),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        curr_ch = ch
        feat_chs = [curr_ch]

        for idx, mult in enumerate(ch_mult):
            out_ch = ch * mult
            cond_ch = self.cond_encoder.out_channels[idx]

            for _ in range(self.conf.RES_BLOCKS):
                self.downs.append(
                    ResBlockSFT(curr_ch, out_ch, time_dim, cond_ch)
                )
                curr_ch = out_ch
                feat_chs.append(curr_ch)

                if idx == len(ch_mult) - 1:
                    self.downs.append(AttentionBlock(curr_ch))

            if idx != len(ch_mult) - 1:
                self.downs.append(Downsample(curr_ch))
                feat_chs.append(curr_ch)

        bottleneck_cond_ch = self.cond_encoder.out_channels[-1]
        self.mid1 = ResBlockSFT(curr_ch, curr_ch, time_dim, bottleneck_cond_ch)
        self.mid_attn = AttentionBlock(curr_ch)
        self.mid2 = ResBlockSFT(curr_ch, curr_ch, time_dim, bottleneck_cond_ch)

        for idx, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            cond_ch = self.cond_encoder.out_channels[idx]

            for _ in range(self.conf.RES_BLOCKS):
                skip_ch = feat_chs.pop()
                self.ups.append(
                    ResBlockSFT(curr_ch + skip_ch, out_ch, time_dim, cond_ch)
                )
                curr_ch = out_ch

                if idx == len(ch_mult) - 1:
                    self.ups.append(AttentionBlock(curr_ch))

            if idx != 0:
                self.ups.append(Upsample(curr_ch))

        self.final_norm = nn.GroupNorm(32, curr_ch)
        self.final_act = Swish()

        self.delta_head = nn.Conv2d(curr_ch, 3, 3, padding=1)
        self.gate_head = nn.Conv2d(curr_ch, 3, 3, padding=1)

    def forward(self, x, t, low_light, return_residual=False):
        t_emb = self.time_mlp(get_timestep_embedding(t, self.conf.CHANNELS))
        cond_feats = self.cond_encoder(low_light)

        x_in = torch.cat([x, low_light], dim=1)
        h = self.head(x_in)

        skips = [h]
        cond_idx = 0

        for layer in self.downs:
            if isinstance(layer, ResBlockSFT):
                h = layer(h, t_emb, cond_feats[cond_idx])
            else:
                h = layer(h)

            if not isinstance(layer, AttentionBlock):
                skips.append(h)

            if isinstance(layer, Downsample):
                cond_idx += 1

        h = self.mid1(h, t_emb, cond_feats[-1])
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, cond_feats[-1])

        cond_idx = len(cond_feats) - 1
        for layer in self.ups:
            if isinstance(layer, ResBlockSFT):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb, cond_feats[cond_idx])

            elif isinstance(layer, Upsample):
                h = layer(h)
                cond_idx = max(cond_idx - 1, 0)

            else:
                h = layer(h)

        h = self.final_act(self.final_norm(h))

        delta = torch.tanh(self.delta_head(h))
        gate = torch.sigmoid(self.gate_head(h))

        pred_residual = gate * delta
        pred_img = torch.clamp(low_light + pred_residual, -1.0, 1.0)

        if return_residual:
            return pred_img, pred_residual
        return pred_img
