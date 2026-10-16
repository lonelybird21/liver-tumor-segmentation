"""Residual U-Net (2D or 3D) with deep supervision and optional attention gates.

Design follows what consistently wins CT segmentation benchmarks (nnU-Net /
"ResEnc" U-Net): InstanceNorm + LeakyReLU, strided-conv downsampling, residual
encoder blocks, transposed-conv upsampling, auxiliary segmentation heads at
lower decoder resolutions (deep supervision).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn


def _ops(dims: int):
    if dims == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.InstanceNorm3d, nn.Dropout3d
    return nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d, nn.Dropout2d


class ConvNormAct(nn.Module):
    def __init__(self, dims, cin, cout, stride=1, kernel=3, act=True):
        super().__init__()
        Conv, _, Norm, _ = _ops(dims)
        self.conv = Conv(cin, cout, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.norm = Norm(cout, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, dims, cin, cout, stride=1, dropout=0.0):
        super().__init__()
        Conv, _, Norm, Drop = _ops(dims)
        self.c1 = ConvNormAct(dims, cin, cout, stride)
        self.drop = Drop(dropout) if dropout > 0 else nn.Identity()
        self.c2 = ConvNormAct(dims, cout, cout, act=False)
        self.skip = (
            nn.Sequential(Conv(cin, cout, 1, stride=stride, bias=False), Norm(cout, affine=True))
            if (stride != 1 or cin != cout)
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.c2(self.drop(self.c1(x))) + self.skip(x))


class PlainBlock(nn.Module):
    def __init__(self, dims, cin, cout, stride=1, dropout=0.0):
        super().__init__()
        _, _, _, Drop = _ops(dims)
        self.body = nn.Sequential(
            ConvNormAct(dims, cin, cout, stride),
            Drop(dropout) if dropout > 0 else nn.Identity(),
            ConvNormAct(dims, cout, cout),
        )

    def forward(self, x):
        return self.body(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018) that re-weights skip features."""

    def __init__(self, dims, skip_ch, gate_ch, inter_ch):
        super().__init__()
        Conv, _, Norm, _ = _ops(dims)
        self.wx = nn.Sequential(Conv(skip_ch, inter_ch, 1, bias=False), Norm(inter_ch, affine=True))
        self.wg = nn.Sequential(Conv(gate_ch, inter_ch, 1, bias=False), Norm(inter_ch, affine=True))
        self.psi = nn.Sequential(Conv(inter_ch, 1, 1), nn.Sigmoid())
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, skip, gate):
        return skip * self.psi(self.act(self.wx(skip) + self.wg(gate)))


class UNet(nn.Module):
    def __init__(
        self,
        dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 32,
        max_filters: int = 320,
        n_stages: int = 5,
        blocks_per_stage: Sequence[int] = (1, 2, 2, 2, 2),
        residual: bool = True,
        deep_supervision: bool = True,
        attention_gates: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dims = dims
        self.deep_supervision = deep_supervision
        Conv, ConvT, _, _ = _ops(dims)
        Block = ResBlock if residual else PlainBlock
        ch = [min(base_filters * 2**i, max_filters) for i in range(n_stages)]
        self.channels = ch

        self.encoder = nn.ModuleList()
        cin = in_channels
        for i in range(n_stages):
            layers = [Block(dims, cin, ch[i], stride=1 if i == 0 else 2, dropout=dropout)]
            layers += [Block(dims, ch[i], ch[i], dropout=dropout) for _ in range(blocks_per_stage[i] - 1)]
            self.encoder.append(nn.Sequential(*layers))
            cin = ch[i]

        self.up, self.gates, self.decoder, self.heads = (nn.ModuleList() for _ in range(4))
        for i in range(n_stages - 2, -1, -1):
            self.up.append(ConvT(ch[i + 1], ch[i], 2, stride=2, bias=False))
            self.gates.append(AttentionGate(dims, ch[i], ch[i], max(ch[i] // 2, 1)) if attention_gates else nn.Identity())
            self.decoder.append(PlainBlock(dims, 2 * ch[i], ch[i], dropout=0.0))
            self.heads.append(Conv(ch[i], num_classes, 1))

    def forward(self, x):
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)
        x = skips.pop()
        outputs: List[torch.Tensor] = []
        for up, gate, dec, head in zip(self.up, self.gates, self.decoder, self.heads):
            x = up(x)
            skip = skips.pop()
            if isinstance(gate, AttentionGate):
                skip = gate(skip, x)
            x = dec(torch.cat([x, skip], dim=1))
            outputs.append(head(x))
        if self.training and self.deep_supervision:
            return outputs[::-1]  # full resolution first, then 1/2, 1/4, ...
        return outputs[-1]
