import math

import torch


ACTIVATION_SCALE = 0.5
HIDDEN_GAIN = 2.2


def ativacao(x: torch.Tensor) -> torch.Tensor:
    return ACTIVATION_SCALE * torch.tanh(x)


@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    
    rectangular_gain = math.sqrt(max(fan_in, fan_out) / fan_in)
    if camada == 1:
        gain = 0.5 * rectangular_gain
    elif camada == n_camadas:
        gain = 2.0 * rectangular_gain
    else:
        gain = HIDDEN_GAIN
    torch.nn.init.orthogonal_(W, gain=gain)
    b.zero_()
