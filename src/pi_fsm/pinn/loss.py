"""Loss functions for the PINN (research_plan.md 3.2), with the fixed
driving-force-direction n=(1,0) design decision (phase-2 plan, note 2):

    L_physics(t) = || m*x_ddot(t) - F(t)*(1,0) + k(t)*x_dot(t) ||^2

x_dot, x_ddot come from autograd through TrajectoryNet, not finite
differences — the standard PINN approach.
"""

import torch

from pi_fsm.pinn.models import ForceNet, ResistanceNet, TrajectoryNet


def trajectory_derivatives(
    traj_net: TrajectoryNet, t: torch.Tensor, v0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (position, velocity, acceleration), each shape (N, 2).

    t must require grad. Uses the sum-then-grad trick: since each sample's
    position depends only on its own t (batch elements are independent),
    d(sum_i x_i)/dt gives the per-sample derivative in one backward pass.
    """
    pos = traj_net(t, v0)
    grad_out = torch.ones_like(pos[:, :1])
    vel = torch.cat(
        [
            torch.autograd.grad(pos[:, i : i + 1], t, grad_out, create_graph=True)[0]
            for i in range(2)
        ],
        dim=-1,
    )
    acc = torch.cat(
        [
            torch.autograd.grad(vel[:, i : i + 1], t, grad_out, create_graph=True)[0]
            for i in range(2)
        ],
        dim=-1,
    )
    return pos, vel, acc


def data_loss(
    traj_net: TrajectoryNet,
    t_obs: torch.Tensor,
    v0_obs: torch.Tensor,
    xy_obs: torch.Tensor,
) -> torch.Tensor:
    pred = traj_net(t_obs, v0_obs)
    return ((pred - xy_obs) ** 2).sum(dim=-1).mean()


def physics_loss(
    traj_net: TrajectoryNet,
    force_net: ForceNet,
    resistance_net: ResistanceNet,
    t_colloc: torch.Tensor,
    v0_colloc: torch.Tensor,
    mass: float = 1.0,
) -> torch.Tensor:
    t_colloc = t_colloc.clone().requires_grad_(True)
    _, vel, acc = trajectory_derivatives(traj_net, t_colloc, v0_colloc)

    F = force_net(t_colloc)  # (N, 1)
    k = resistance_net(t_colloc)  # (N, 1)

    res_x = mass * acc[:, :1] - F + k * vel[:, :1]  # n=(1,0)
    res_y = mass * acc[:, 1:] + k * vel[:, 1:]  # n_y=0: pure damping off-axis
    return (res_x**2 + res_y**2).mean()


def total_loss(
    traj_net: TrajectoryNet,
    force_net: ForceNet,
    resistance_net: ResistanceNet,
    t_obs: torch.Tensor,
    v0_obs: torch.Tensor,
    xy_obs: torch.Tensor,
    t_colloc: torch.Tensor,
    v0_colloc: torch.Tensor,
    gamma: float,
    mass: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (total, data_loss, physics_loss)."""
    l_data = data_loss(traj_net, t_obs, v0_obs, xy_obs)
    l_phys = physics_loss(traj_net, force_net, resistance_net, t_colloc, v0_colloc, mass)
    return l_data + gamma * l_phys, l_data, l_phys
