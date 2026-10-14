"""Dataset-independent submission for Desafio 2.

Scaled tanh and orthogonal weights. The bounded, centered activation and
controlled singular values improve training stability in deep plain MLPs.
All constants are fixed: no dataset access or dependence on batch statistics.
"""
import math

import torch


ACTIVATION_SCALE = 0.5
HIDDEN_GAIN = 2.2


def ativacao(x: torch.Tensor) -> torch.Tensor:
    """Apply a stateless elementwise activation, without trainable parameters."""
    return ACTIVATION_SCALE * torch.tanh(x)


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    """Fill weights and biases in-place using only layer metadata."""
    # A rectangular orthogonal matrix has mean row squared norm
    # min(fan_in, fan_out) / fan_out. Compensate narrow input projections.
    rectangular_gain = math.sqrt(max(fan_in, fan_out) / fan_in)
    if camada == 1:
        gain = 0.5 * rectangular_gain
    elif camada == n_camadas:
        gain = 2.0 * rectangular_gain
    else:
        # The activation scale cancels half this gain: the local factor near
        # zero is 1.1. Saturation limits growth away from zero.
        gain = HIDDEN_GAIN
    torch.nn.init.orthogonal_(W, gain=gain)
    b.zero_()
