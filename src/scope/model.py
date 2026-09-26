"""SCOPE: mask-aware spatial encoder, predictor, nonlinear decoder and EMA teacher."""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    resolution: int = 128
    patch_size: int = 8
    encoder_width: int = 512
    encoder_depth: int = 12
    encoder_heads: int = 8
    predictor_width: int = 256
    predictor_depth: int = 2
    predictor_heads: int = 8
    mlp_ratio: int = 4
    anchor_grid: int = 16
    anchor_channels: int = 128
    decoder_width: int = 64
    decoder_output_channels: int = 2
    dropout: float = 0.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name != "dropout" and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution % self.patch_size:
            raise ValueError("patch_size must divide resolution")
        patch_grid = self.resolution // self.patch_size
        if patch_grid % self.anchor_grid:
            raise ValueError("anchor_grid must divide the patch grid without upsampling")
        scale = self.resolution // self.anchor_grid
        if scale & (scale - 1):
            raise ValueError("resolution/anchor_grid must be a power of two for the spatial decoder")
        if self.encoder_width % self.encoder_heads or self.predictor_width % self.predictor_heads:
            raise ValueError("transformer widths must be divisible by their head counts")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")


def _transformer(width: int, heads: int, depth: int, spec: ModelSpec) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=width * spec.mlp_ratio,
        dropout=spec.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
    # TransformerEncoder clones its supplied layer, including the same random
    # initial weights. Initialize each cloned block independently here.
    for block in encoder.layers:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
    return encoder


class SpatialAnchorEncoder(nn.Module):
    """Encode [observed a, mask a, observed u, mask u] into spatial tokens."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.patch_grid = spec.resolution // spec.patch_size
        self.patch_embed = nn.Conv2d(4, spec.encoder_width, spec.patch_size, spec.patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.patch_grid**2, spec.encoder_width))
        self.encoder = _transformer(spec.encoder_width, spec.encoder_heads, spec.encoder_depth, spec)
        self.norm = nn.LayerNorm(spec.encoder_width)
        self.spatial_pool = nn.AdaptiveAvgPool2d((spec.anchor_grid, spec.anchor_grid))
        self.anchor_head = nn.Linear(spec.encoder_width, spec.anchor_channels)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (4, self.spec.resolution, self.spec.resolution):
            raise ValueError("inputs must have shape [batch, 4, resolution, resolution]")
        if not inputs.is_floating_point():
            raise ValueError("inputs must be floating point")
        # Validation is outside the graph only for actual data, not meta-device counting.
        if inputs.device.type != "meta":
            if not bool(torch.isfinite(inputs).all()):
                raise ValueError("inputs must be finite")
            masks = inputs[:, (1, 3)]
            if not bool(((masks == 0) | (masks == 1)).all()):
                raise ValueError("observation mask channels must be binary")
            if not bool((inputs[:, (0, 2)] * (1 - masks) == 0).all()):
                raise ValueError("unobserved field entries must be zero after normalization")
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2) + self.position
        tokens = self.norm(self.encoder(tokens))
        spatial = tokens.transpose(1, 2).reshape(
            len(inputs), self.spec.encoder_width, self.patch_grid, self.patch_grid
        )
        pooled = self.spatial_pool(spatial).flatten(2).transpose(1, 2)
        return self.anchor_head(pooled)


class SpatialPredictor(nn.Module):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        spec.validate()
        self.input_projection = nn.Linear(spec.anchor_channels, spec.predictor_width)
        self.position = nn.Parameter(torch.zeros(1, spec.anchor_grid**2, spec.predictor_width))
        self.transformer = _transformer(
            spec.predictor_width, spec.predictor_heads, spec.predictor_depth, spec
        )
        self.norm = nn.LayerNorm(spec.predictor_width)
        self.output_projection = nn.Linear(spec.predictor_width, spec.anchor_channels)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, anchor_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(anchor_tokens) + self.position
        return self.output_projection(self.norm(self.transformer(hidden)))


class NonlinearPatchDecoder(nn.Module):
    """Decode spatial tokens into field patches, followed by nonlinear refinement.

    Every recovered field passes through the predicted latent. There is no
    observed-field skip connection and no hard replacement of observed values.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        patch = spec.resolution // spec.anchor_grid
        self.patch = patch
        hidden = max(2 * spec.anchor_channels, 32)
        self.patch_head = nn.Sequential(
            nn.Linear(spec.anchor_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * patch**2),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(2, spec.decoder_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(spec.decoder_width, spec.decoder_width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(spec.decoder_width, 2, 3, padding=1),
        )

    def forward(self, latent):
        grid, patch = self.spec.anchor_grid, self.patch
        if latent.ndim != 3 or latent.shape[1:] != (grid**2, self.spec.anchor_channels):
            raise ValueError("Unexpected predicted spatial latent shape")
        tiles = self.patch_head(latent).reshape(len(latent), grid, grid, 2, patch, patch)
        field = tiles.permute(0, 3, 1, 4, 2, 5).reshape(
            len(latent),
            2,
            self.spec.resolution,
            self.spec.resolution,
        )
        return field + self.refinement(field)


class SCOPE(nn.Module):
    def __init__(self, spec=None):
        super().__init__()
        self.spec = spec or ModelSpec()
        self.spec.validate()
        if self.spec.decoder_output_channels != 2:
            raise ValueError("SCOPE recovers both a and u")
        self.encoder = SpatialAnchorEncoder(self.spec)
        # Same-size adaptive pooling is exactly identity. Avoid its unsupported
        # deterministic CUDA backward without changing values or parameters.
        if self.encoder.patch_grid == self.spec.anchor_grid:
            self.encoder.spatial_pool = nn.Identity()
        self.predictor = SpatialPredictor(self.spec)
        # Derived solely from masks, never from hidden fields or target values.
        self.observation_condition = nn.Linear(2, self.spec.anchor_channels)
        self.mask_pool = nn.AdaptiveAvgPool2d((self.spec.anchor_grid, self.spec.anchor_grid))
        self.decoder = NonlinearPatchDecoder(self.spec)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.register_buffer("ema_updates", torch.tensor(0, dtype=torch.long))

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, observations):
        encoded = self.encoder(observations)
        condition = self.mask_pool(observations[:, (1, 3)]).flatten(2).transpose(1, 2)
        predicted = self.predictor(encoded + self.observation_condition(condition))
        return {"fields": self.decoder(predicted), "predicted_latent": predicted, "context_latent": encoded}

    @staticmethod
    def full_pair_view(fields):
        ones = torch.ones_like(fields[:, 0])
        return torch.stack((fields[:, 0], ones, fields[:, 1], ones), dim=1)

    @torch.no_grad()
    def update_target(self, momentum):
        if not math.isfinite(momentum) or not 0 <= momentum <= 1:
            raise ValueError("Invalid EMA momentum")
        for teacher, student in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            teacher.lerp_(student.detach(), 1 - momentum)
        for teacher, student in zip(self.target_encoder.buffers(), self.encoder.buffers(), strict=True):
            teacher.copy_(student)
        self.ema_updates.add_(1)
