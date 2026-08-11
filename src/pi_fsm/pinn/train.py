"""Training loop for the PINN (research_plan.md 4.3 phase 2 / 4.4
checkpointing). Shared by scripts/phase2_sanity_check.py and
scripts/phase2_pilot_train.py.

gamma follows a linear warmup curriculum (0 -> gamma_final over
gamma_warmup_epochs) per research_plan.md 3.2 / 4.3's "learning
stabilization" note: too much physics weight early on tends to fight the
still-random trajectory network.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from pi_fsm.pinn.loss import total_loss
from pi_fsm.pinn.models import ForceNet, ResistanceNet, TrajectoryNet


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    epochs: int = 2000
    lr: float = 1e-3
    lr_min_factor: float = 0.02  # cosine-anneal lr down to lr * lr_min_factor
    grad_clip: float = 1.0
    gamma_final: float = 1.0
    gamma_warmup_epochs: int = 500
    n_colloc: int = 2000
    t_max: float = 10.0
    v0_min: float = 0.0
    v0_max: float = 8.0
    mass: float = 1.0
    traj_hidden: int = 64
    traj_n_hidden: int = 4
    force_hidden: int = 32
    force_n_hidden: int = 3
    device: str = field(default_factory=resolve_device)
    checkpoint_dir: str | None = None
    checkpoint_every: int = 500
    log_every: int = 100
    seed: int = 0


@dataclass
class TrainResult:
    traj_net: TrajectoryNet
    force_net: ForceNet
    resistance_net: ResistanceNet
    history: pd.DataFrame  # columns: epoch, total, data, physics, gamma


def _save_checkpoint(path: Path, epoch: int, traj, force, resist, opt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "traj_net": traj.state_dict(),
            "force_net": force.state_dict(),
            "resistance_net": resist.state_dict(),
            "optimizer": opt.state_dict(),
        },
        path,
    )


def train_pinn(segments_df: pd.DataFrame, config: TrainConfig | None = None) -> TrainResult:
    """segments_df: long format with columns t, x, y, v0 (rotated frame,
    see segments.extract_sprints). One row per (segment, frame)."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = config.device

    traj_net = TrajectoryNet(hidden=config.traj_hidden, n_hidden=config.traj_n_hidden).to(device)
    force_net = ForceNet(hidden=config.force_hidden, n_hidden=config.force_n_hidden).to(device)
    resistance_net = ResistanceNet(hidden=config.force_hidden, n_hidden=config.force_n_hidden).to(device)
    params = list(traj_net.parameters()) + list(force_net.parameters()) + list(
        resistance_net.parameters()
    )
    opt = torch.optim.Adam(params, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.epochs, eta_min=config.lr * config.lr_min_factor
    )

    t_obs = torch.tensor(segments_df["t"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(-1)
    v0_obs = torch.tensor(segments_df["v0"].to_numpy(), dtype=torch.float32, device=device).unsqueeze(-1)
    xy_obs = torch.tensor(segments_df[["x", "y"]].to_numpy(), dtype=torch.float32, device=device)

    rows = []
    for epoch in range(config.epochs):
        gamma = config.gamma_final * min(1.0, (epoch + 1) / max(1, config.gamma_warmup_epochs))

        t_colloc = torch.rand(config.n_colloc, 1, device=device) * config.t_max
        v0_colloc = (
            torch.rand(config.n_colloc, 1, device=device) * (config.v0_max - config.v0_min)
            + config.v0_min
        )

        opt.zero_grad()
        loss, l_data, l_phys = total_loss(
            traj_net, force_net, resistance_net,
            t_obs, v0_obs, xy_obs, t_colloc, v0_colloc,
            gamma=gamma, mass=config.mass,
        )
        loss.backward()
        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
        opt.step()
        scheduler.step()

        if epoch % config.log_every == 0 or epoch == config.epochs - 1:
            rows.append(
                {
                    "epoch": epoch,
                    "total": loss.item(),
                    "data": l_data.item(),
                    "physics": l_phys.item(),
                    "gamma": gamma,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if config.checkpoint_dir and (epoch + 1) % config.checkpoint_every == 0:
            _save_checkpoint(
                Path(config.checkpoint_dir) / f"epoch_{epoch+1}.pt",
                epoch, traj_net, force_net, resistance_net, opt,
            )

    return TrainResult(
        traj_net=traj_net,
        force_net=force_net,
        resistance_net=resistance_net,
        history=pd.DataFrame(rows),
    )
