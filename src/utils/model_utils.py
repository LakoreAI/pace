import random
import string
from datetime import datetime

import torch


def get_run_name(
    model_name: str,
    dset_name: str,
    lr: float,
    batch_size: int,
    experiment_type: str = "",
    random_suffix: bool = True,
    random_suffix_len: int = 6,
) -> str:
    """Generate a unique, sortable run name with a timestamp (YYYY-MM-DD)."""
    now = datetime.now().strftime("%Y-%m-%d")
    rand_suffix = (
        "".join(
            random.Random().choices(
                string.ascii_letters + string.digits, k=random_suffix_len
            )
        )
        if random_suffix
        else ""
    )
    return f"{model_name}_{dset_name}_{experiment_type}_lr{lr}_bs{batch_size}_{now}_{rand_suffix}"


def count_parameters(model: torch.nn.Module, verbose: bool = True) -> dict:
    """Count parameters in a PyTorch model.

    from src.utils.model_utils import count_parameters
    count_parameters(model)
    """
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_all - n_trainable
    if verbose:
        print(
            "Parameter Count: all {:,d}; trainable {:,d}; frozen {:,d}".format(
                n_all, n_trainable, n_frozen
            )
        )
    return {
        "n_all": n_all,
        "n_trainable": n_trainable,
        "n_frozen": n_frozen,
    }


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: derive NumPy / random seeds from torch's."""
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def rng_state() -> dict:
    import numpy as np

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    import numpy as np

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def autocast_dtype(name: str, device: torch.device) -> torch.dtype:
    """ "bf16" (default) or "fp16". Falls back to fp16 where bf16 is unsupported
    (e.g. pre-Ampere GPUs) so the same config runs locally and on RunPod.
    """
    if name not in ("bf16", "fp16"):
        raise ValueError(f"amp_dtype must be 'bf16' or 'fp16', got {name!r}")
    if (
        name == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        print("  [amp] bf16 not supported on this GPU, using fp16")
        return torch.float16
    return torch.bfloat16 if name == "bf16" else torch.float16
