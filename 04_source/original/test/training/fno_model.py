"""Small 3D Fourier neural operator and physics diagnostics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class SpectralConv3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z)
        self.weights = nn.ParameterList(
            [nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(4)]
        )

    @staticmethod
    def multiply(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", values, weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny, nz = values.shape
        transformed = torch.fft.rfftn(values.float(), dim=(-3, -2, -1), norm="ortho")
        output = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny,
            nz // 2 + 1,
            dtype=torch.cfloat,
            device=values.device,
        )
        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2)
        mz = min(self.modes_z, nz // 2 + 1)
        output[:, :, :mx, :my, :mz] = self.multiply(
            transformed[:, :, :mx, :my, :mz], self.weights[0][:, :, :mx, :my, :mz]
        )
        output[:, :, -mx:, :my, :mz] = self.multiply(
            transformed[:, :, -mx:, :my, :mz], self.weights[1][:, :, :mx, :my, :mz]
        )
        output[:, :, :mx, -my:, :mz] = self.multiply(
            transformed[:, :, :mx, -my:, :mz], self.weights[2][:, :, :mx, :my, :mz]
        )
        output[:, :, -mx:, -my:, :mz] = self.multiply(
            transformed[:, :, -mx:, -my:, :mz], self.weights[3][:, :, :mx, :my, :mz]
        )
        return torch.fft.irfftn(output, s=(nx, ny, nz), dim=(-3, -2, -1), norm="ortho")


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, *modes)
        self.local = nn.Conv3d(width, width, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return functional.gelu(self.spectral(values) + self.local(values))


class FNO3dStep(nn.Module):
    """Predict one normalized state and hard-enforce static solid/inlet values."""

    def __init__(
        self,
        input_channels: int,
        width: int,
        layers: int,
        modes: tuple[int, int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.padding_x, self.padding_y = padding
        self.lift = nn.Conv3d(input_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock3d(width, modes) for _ in range(layers)])
        self.project = nn.Sequential(
            nn.Conv3d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(2 * width, 4, kernel_size=1),
        )
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        current: torch.Tensor,
        fluid_mask: torch.Tensor,
        inlet_reference: torch.Tensor,
    ) -> torch.Tensor:
        values = self.lift(features)
        values = functional.pad(
            values,
            (0, 0, self.padding_y, self.padding_y, self.padding_x, self.padding_x),
        )
        for block in self.blocks:
            values = block(values)
        values = values[
            ...,
            self.padding_x : -self.padding_x,
            self.padding_y : -self.padding_y,
            :,
        ]
        prediction = current + self.project(values) * fluid_mask
        inlet_mask = torch.zeros_like(prediction)
        inlet_mask[:, :3, 0, :, :] = 1.0
        return prediction * (1.0 - inlet_mask) + inlet_reference * inlet_mask


def masked_channel_mse(
    prediction: torch.Tensor, target: torch.Tensor, fluid_mask: torch.Tensor
) -> torch.Tensor:
    denominator = fluid_mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * fluid_mask).sum() / (
        denominator * prediction.shape[1]
    )


def normalized_divergence_loss(
    prediction: torch.Tensor,
    fluid_mask: torch.Tensor,
    field_mean: torch.Tensor,
    field_std: torch.Tensor,
    spacing: torch.Tensor,
    divergence_scale: torch.Tensor,
) -> torch.Tensor:
    velocity = prediction[:, :3] * field_std[:, :3] + field_mean[:, :3]
    dx = spacing[:, 0].reshape(-1, 1, 1, 1)
    dy = spacing[:, 1].reshape(-1, 1, 1, 1)
    dz = spacing[:, 2].reshape(-1, 1, 1, 1)
    divergence = (
        (velocity[:, 0, 2:, 1:-1, 1:-1] - velocity[:, 0, :-2, 1:-1, 1:-1])
        / (2.0 * dx)
        + (velocity[:, 1, 1:-1, 2:, 1:-1] - velocity[:, 1, 1:-1, :-2, 1:-1])
        / (2.0 * dy)
        + (velocity[:, 2, 1:-1, 1:-1, 2:] - velocity[:, 2, 1:-1, 1:-1, :-2])
        / (2.0 * dz)
    )
    center = fluid_mask[:, 0, 1:-1, 1:-1, 1:-1]
    stencil = (
        center
        * fluid_mask[:, 0, 2:, 1:-1, 1:-1]
        * fluid_mask[:, 0, :-2, 1:-1, 1:-1]
        * fluid_mask[:, 0, 1:-1, 2:, 1:-1]
        * fluid_mask[:, 0, 1:-1, :-2, 1:-1]
        * fluid_mask[:, 0, 1:-1, 1:-1, 2:]
        * fluid_mask[:, 0, 1:-1, 1:-1, :-2]
    )
    scale = divergence_scale.reshape(-1, 1, 1, 1)
    return (((divergence / scale) ** 2) * stencil).sum() / stencil.sum().clamp_min(1.0)
