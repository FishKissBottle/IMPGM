import torch
from IMPGM_Config import *



def power_scheduler(timesteps=1000, min_beta=0.0001, max_beta=0.999, power_val=3):

    x = torch.linspace(1 / timesteps, 1.0, timesteps, dtype=torch.float32)

    alpha_bar_t = 1.0 - (x ** power_val)
    alpha_bar_prev = 1 - (
        torch.cat([torch.tensor([0.0], dtype=torch.float32), x[:-1]]) ** power_val
    )

    beta_t = 1 - (alpha_bar_t / alpha_bar_prev) 
    beta_t = torch.clamp(beta_t, min=min_beta, max=max_beta)
    
    return beta_t



def linear_scheduler(timesteps=1000, min_beta=0.0001, max_beta=0.01):

    beta_t = torch.linspace(min_beta, max_beta, timesteps, dtype=torch.float32)

    return beta_t



def cosine_scheduler(timesteps=1000, min_beta=0.0001, max_beta=0.999, s=0.008):

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    
    # clamp beta to avoid extreme values
    beta_t = torch.clip(betas, min_beta, max_beta)

    return beta_t



def sigmoid_scheduler(timesteps=1000, min_beta=0.0001, max_beta=0.999, start=-12.0, end=-2.0):

    x = torch.linspace(start, end, timesteps, dtype=torch.float32)
    sig = torch.sigmoid(x)
    beta_t = min_beta + (max_beta - min_beta) * sig

    return beta_t



SUPPORTED_SCHEDULER_TYPES = {"cosine", "linear", "power", "sigmoid"}


def resolve_scheduler_hyperparams(
    scheduler_type: str,
    *,
    min_beta: float = 1e-4,
    max_beta: float = 0.999,
    cosine_s: float = 0.008,
    sigmoid_start: float = -12.0,
    sigmoid_end: float = -2.0,
):
    """Normalize scheduler hyperparameters to match runtime defaults.

    In particular, preserve the historical linear-schedule default
    ``max_beta=0.01`` when callers provide the repository-wide default
    ``max_beta=0.999`` without explicitly overriding it.
    """
    scheduler_type = str(scheduler_type).lower().strip()
    resolved = {
        "min_beta": float(min_beta),
        "max_beta": float(max_beta),
        "cosine_s": float(cosine_s),
        "sigmoid_start": float(sigmoid_start),
        "sigmoid_end": float(sigmoid_end),
    }

    if scheduler_type == "linear" and resolved["max_beta"] == 0.999:
        resolved["max_beta"] = 0.01

    return resolved


def build_beta_schedule(
    scheduler_type: str,
    timesteps: int,
    power_val: int = 3,
    min_beta: float = 1e-4,
    max_beta: float = 0.999,
    cosine_s: float = 0.008,
    sigmoid_start: float = -12.0,
    sigmoid_end: float = -2.0,
):
    """Central factory for constructing a beta_t schedule.

    Parameters
    ----------
    scheduler_type : str
        One of "cosine", "linear", "power", "sigmoid".
    timesteps : int
        Number of diffusion steps.
    power_val : int, optional
        Power exponent when ``scheduler_type == "power"``.
    min_beta : float, optional
        Lower clamp bound for beta_t.
    max_beta : float, optional
        Upper clamp bound for beta_t.
    cosine_s : float, optional
        Offset parameter for the cosine schedule.
    sigmoid_start : float, optional
        Start of the sigmoid input range.
    sigmoid_end : float, optional
        End of the sigmoid input range.

    Returns
    -------
    torch.Tensor
        1-D tensor of shape ``(timesteps,)`` containing beta_t.
    """
    scheduler_type = str(scheduler_type).lower().strip()
    resolved = resolve_scheduler_hyperparams(
        scheduler_type,
        min_beta=min_beta,
        max_beta=max_beta,
        cosine_s=cosine_s,
        sigmoid_start=sigmoid_start,
        sigmoid_end=sigmoid_end,
    )

    if scheduler_type == "cosine":
        return cosine_scheduler(
            timesteps=timesteps,
            min_beta=resolved["min_beta"],
            max_beta=resolved["max_beta"],
            s=resolved["cosine_s"],
        )
    elif scheduler_type == "linear":
        return linear_scheduler(
            timesteps=timesteps,
            min_beta=resolved["min_beta"],
            max_beta=resolved["max_beta"],
        )
    elif scheduler_type == "power":
        return power_scheduler(
            timesteps=timesteps,
            min_beta=resolved["min_beta"],
            max_beta=resolved["max_beta"],
            power_val=power_val,
        )
    elif scheduler_type == "sigmoid":
        return sigmoid_scheduler(
            timesteps=timesteps,
            min_beta=resolved["min_beta"],
            max_beta=resolved["max_beta"],
            start=resolved["sigmoid_start"],
            end=resolved["sigmoid_end"],
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type!r}. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_SCHEDULER_TYPES))}."
        )


def build_beta_schedule_from_config(task_config=None):
    if task_config is None:
        from IMPGM_Config import FGGEN_DIFFUSION_CONFIG
        task_config = FGGEN_DIFFUSION_CONFIG

    scheduler_type = str(task_config.SCHEDULER_TYPE).lower().strip()

    return build_beta_schedule(
        scheduler_type=scheduler_type,
        timesteps=task_config.STEPS,
        power_val=task_config.SCHEDULER_POWER_VAL,
        min_beta=task_config.SCHEDULER_MIN_BETA,
        max_beta=task_config.SCHEDULER_MAX_BETA,
        cosine_s=task_config.SCHEDULER_COSINE_S,
        sigmoid_start=task_config.SCHEDULER_SIGMOID_START,
        sigmoid_end=task_config.SCHEDULER_SIGMOID_END,
    )


def build_scheduler_tag(scheduler_type: str = None, power_val: int = None) -> str:
    """Return a filesystem-friendly scheduler tag for naming experiments.

    Examples
    --------
    >>> build_scheduler_tag("cosine")
    'cosine'
    >>> build_scheduler_tag("power", 3)
    'power_p3'
    """
    if scheduler_type is None:
        from IMPGM_Config import SCHEDULER_TYPE, SCHEDULER_POWER_VAL
        scheduler_type = SCHEDULER_TYPE
        if power_val is None:
            power_val = SCHEDULER_POWER_VAL

    scheduler_type = str(scheduler_type).lower().strip()
    if scheduler_type == "power" and power_val is not None:
        return f"{scheduler_type}_p{power_val}"
    return scheduler_type
