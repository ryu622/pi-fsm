"""PINN model definitions (research_plan.md 3.1).

F(t) = NN_F(t), k(t) = NN_k(t): small player-agnostic MLPs, softplus output
(F, k are physically non-negative). x_hat(t, v0): the trajectory network,
conditioned on the segment's initial speed v0 rather than a per-segment
learned embedding (see the phase-2 plan's design note 3).

x_hat uses a "hard constraint" ansatz so the initial conditions implied by
the segment rotation (x(0)=0, y(0)=0, dx/dt(0)=v0, dy/dt(0)=0) hold exactly
for any network weights, rather than being enforced only approximately via
an extra loss term:

    x(t) = v0*t + t^2 * net_x(t, v0)
    y(t) = t^2 * net_y(t, v0)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, n_hidden: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ForceNet(nn.Module):
    """F(t) >= 0 (per-unit-mass driving force magnitude)."""

    def __init__(self, hidden: int = 32, n_hidden: int = 3):
        super().__init__()
        self.mlp = MLP(1, 1, hidden, n_hidden)
        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.mlp(t))


class ResistanceNet(nn.Module):
    """k(t) >= 0 (per-unit-mass viscous resistance coefficient)."""

    def __init__(self, hidden: int = 32, n_hidden: int = 3):
        super().__init__()
        self.mlp = MLP(1, 1, hidden, n_hidden)
        self.softplus = nn.Softplus()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.mlp(t))


class TrajectoryNet(nn.Module):
    """x_hat(t, v0) -> (x, y) in the rotated per-segment frame."""

    def __init__(self, hidden: int = 64, n_hidden: int = 4):
        super().__init__()
        self.mlp = MLP(2, 2, hidden, n_hidden)

    def forward(self, t: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """t, v0: shape (N, 1). Returns shape (N, 2)."""
        raw = self.mlp(torch.cat([t, v0], dim=-1))
        x = v0 * t + t**2 * raw[:, :1]
        y = t**2 * raw[:, 1:]
        return torch.cat([x, y], dim=-1)
