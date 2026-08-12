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

from pi_fsm.pinn.loss import physics_loss, total_loss
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
    beta: float = 0.0  # smoothness regularization weight on dF/dt, dk/dt (0 = off)
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
    stage1_history: pd.DataFrame | None = None  # set only by train_two_stage


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
        loss, l_data, l_phys, l_smooth = total_loss(
            traj_net, force_net, resistance_net,
            t_obs, v0_obs, xy_obs, t_colloc, v0_colloc,
            gamma=gamma, mass=config.mass, beta=config.beta,
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
                    "smooth": l_smooth.item(),
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


def train_two_stage(
    segments_df: pd.DataFrame,
    stage1_config: TrainConfig | None = None,
    stage2_config: TrainConfig | None = None,
) -> TrainResult:
    """Two-stage training (see documents/phase2_pilot_results.md "次の一手"):

    Stage 1: fit x_hat(t, v0) on data loss alone (gamma forced to 0).
    Stage 2: freeze x_hat's weights; fit F(t), k(t) on physics loss alone.

    Motivation: jointly optimizing x_hat and F,k let them co-adapt into a
    trajectory-consistent but physically degenerate split (seed-to-seed
    Vmax=F/k CV of 30-45% even with a fixed schedule and wide v0 range —
    see the Colab seed-sweep results). Freezing x_hat after it has already
    fit the data well removes that moving target, so F,k only need to
    explain one fixed trajectory rather than co-adapt with it.
    """
    stage1_config = stage1_config or TrainConfig(gamma_final=0.0)
    stage1 = train_pinn(segments_df, stage1_config)

    stage2_config = stage2_config or TrainConfig(
        epochs=stage1_config.epochs,
        lr=stage1_config.lr,
        lr_min_factor=stage1_config.lr_min_factor,
        grad_clip=stage1_config.grad_clip,
        n_colloc=stage1_config.n_colloc,
        t_max=stage1_config.t_max,
        v0_min=stage1_config.v0_min,
        v0_max=stage1_config.v0_max,
        mass=stage1_config.mass,
        force_hidden=stage1_config.force_hidden,
        force_n_hidden=stage1_config.force_n_hidden,
        device=stage1_config.device,
        log_every=stage1_config.log_every,
        seed=stage1_config.seed,
    )
    device = stage2_config.device

    traj_net = stage1.traj_net
    for p in traj_net.parameters():
        p.requires_grad_(False)

    force_net = ForceNet(hidden=stage2_config.force_hidden, n_hidden=stage2_config.force_n_hidden).to(device)
    resistance_net = ResistanceNet(
        hidden=stage2_config.force_hidden, n_hidden=stage2_config.force_n_hidden
    ).to(device)
    params = list(force_net.parameters()) + list(resistance_net.parameters())
    opt = torch.optim.Adam(params, lr=stage2_config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=stage2_config.epochs, eta_min=stage2_config.lr * stage2_config.lr_min_factor
    )

    rows = []
    for epoch in range(stage2_config.epochs):
        t_colloc = torch.rand(stage2_config.n_colloc, 1, device=device) * stage2_config.t_max
        v0_colloc = (
            torch.rand(stage2_config.n_colloc, 1, device=device)
            * (stage2_config.v0_max - stage2_config.v0_min)
            + stage2_config.v0_min
        )

        opt.zero_grad()
        l_phys = physics_loss(traj_net, force_net, resistance_net, t_colloc, v0_colloc, mass=stage2_config.mass)
        l_phys.backward()
        if stage2_config.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, stage2_config.grad_clip)
        opt.step()
        scheduler.step()

        if epoch % stage2_config.log_every == 0 or epoch == stage2_config.epochs - 1:
            rows.append({"epoch": epoch, "physics": l_phys.item(), "lr": scheduler.get_last_lr()[0]})

    return TrainResult(
        traj_net=traj_net,
        force_net=force_net,
        resistance_net=resistance_net,
        history=pd.DataFrame(rows),
        stage1_history=stage1.history,
    )
