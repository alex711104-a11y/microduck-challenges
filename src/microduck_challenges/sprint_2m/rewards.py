"""Custom reward terms for the 2 m Sprint."""

import torch


def exceed_command(
    env,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward forward-body speed beyond the sampled twist command.

    This is bounded and proportional to excess speed only.  It supplements,
    rather than replaces, command tracking: the sweep commands 0.20-1.20 m/s,
    while this term rewards any additional forward progress.
    """
    vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
    commanded_vx = env.command_manager.get_command(command_name)[:, 0]
    excess = (vx - commanded_vx).clamp(min=0.0)
    return torch.tanh(excess / std)
