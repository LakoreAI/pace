"""SpecAugment as used by the original one-pass detector on TUT 2017.

The original (DemisEom/SpecAugment, TensorFlow) draws one frequency mask and
`time_mask_num` time masks per batch, shares them across every example in the
batch, fills with 0, and does no time warping. This reproduces that.

Also: mixup, FilterAugment and Freq-MixStyle, following the training recipes of
recent SED work (arXiv:2409.09546, arXiv:2601.04178).
"""

from typing import Optional

import torch


def spec_augment_batch(
    x: torch.Tensor,
    freq_param: int = 8,
    freq_mask_num: int = 1,
    time_param: int = 25,
    time_mask_num: int = 2,
    fill: str = "zero",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """x: (B, n_frames, n_mels). Returns a masked copy.

    `fill="zero"` is the original behaviour. Since the cache is log-mel in dB
    (values <= 0), zero is the *loudest* level; `fill="min"` instead masks with
    each example's minimum, i.e. silence.
    """
    if fill not in ("zero", "min"):
        raise ValueError(f"fill must be 'zero' or 'min', got {fill!r}")
    x = x.clone()
    n_frames, n_mels = x.shape[1], x.shape[2]
    floor = None
    if fill == "min":
        floor = x.amin(dim=(1, 2), keepdim=True).expand_as(x)

    def masked(region: tuple) -> torch.Tensor:
        return floor[region] if fill == "min" else 0.0

    def draw(high: int) -> int:
        return int(torch.randint(0, max(high, 1), (1,), generator=generator))

    for _ in range(freq_mask_num):
        f = draw(freq_param)
        f0 = draw(n_mels - f)
        region = (slice(None), slice(None), slice(f0, f0 + f))
        x[region] = masked(region)
    for _ in range(time_mask_num):
        t = draw(time_param)
        t0 = draw(n_frames - t)
        region = (slice(None), slice(t0, t0 + t), slice(None))
        x[region] = masked(region)
    return x


def mixup(
    x: torch.Tensor,
    alpha: float = 0.2,
    p: float = 1.0,
    generator: Optional[torch.Generator] = None,
):
    """Mix each example with a shuffled partner. Returns (x_mixed, perm, lam).

    Target mixing is head-specific (frame-wise labels: `max` or convex mix;
    event-level targets: union of both event sets), so it is left to the caller.
    """
    if float(torch.rand(1, generator=generator)) > p:
        perm = torch.arange(x.shape[0], device=x.device)
        return x, perm, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.shape[0], generator=generator).to(x.device)
    return lam * x + (1.0 - lam) * x[perm], perm, lam


def filter_augment(
    x: torch.Tensor,
    db_range: tuple = (-6.0, 6.0),
    n_band: tuple = (3, 6),
    min_bw: int = 6,
    p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """FilterAugment, step variant (Nam, Kim, Park, ICASSP 2022), on log-mel in dB.

    Splits the mel axis into a random number of bands and adds a random gain in
    `db_range` to each, per example. x: (B, n_frames, n_mels).
    """
    if float(torch.rand(1, generator=generator)) > p:
        return x
    b, _, n_mels = x.shape
    out = x.clone()
    for i in range(b):
        n = int(torch.randint(n_band[0], n_band[1] + 1, (1,), generator=generator))
        n = max(1, min(n, n_mels // max(min_bw, 1)))
        inner = torch.randperm(n_mels - 1, generator=generator)[: n - 1] + 1
        edges = [0] + sorted(inner.tolist()) + [n_mels]
        gains = torch.empty(n).uniform_(*db_range, generator=generator)
        for (lo, hi), g in zip(zip(edges[:-1], edges[1:]), gains):
            out[i, :, lo:hi] += float(g)
    return out


def freq_mixstyle(
    x: torch.Tensor,
    alpha: float = 0.3,
    p: float = 0.7,
    eps: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Freq-MixStyle (Kim et al. 2022; Schmid et al. 2022): mix per-frequency
    statistics (mean / std over time) between examples, applied with probability p.

    x: (B, n_frames, n_mels).
    """
    if float(torch.rand(1, generator=generator)) > p:
        return x
    b = x.shape[0]
    mu = x.mean(dim=1, keepdim=True)
    sig = (x.var(dim=1, keepdim=True) + eps).sqrt()
    x_norm = (x - mu) / sig
    lam = torch.distributions.Beta(alpha, alpha).sample((b, 1, 1)).to(x.device, x.dtype)
    perm = torch.randperm(b, generator=generator).to(x.device)
    mu_mix = lam * mu + (1 - lam) * mu[perm]
    sig_mix = lam * sig + (1 - lam) * sig[perm]
    return x_norm * sig_mix + mu_mix


def copy_paste_events(
    x: torch.Tensor,
    target: torch.Tensor,
    classes: int,
    p: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy one active event patch from a donor example into a recipient.

    The computer-vision Copy-Paste augmentation adapted to frame-wise SED:
    `x` is (B, cells, mels) and `target` is (B, cells, classes) presence.
    A contiguous active run of one class is copied (all mel bins) to a random
    position in another example and the class is marked active there. This is
    the on-GPU analogue of the SCAPER recipe (add isolated events to a
    background) and is currently defined for frame-wise targets only.

    Caveat: pasted audio may contain other events whose labels are unknown;
    this is the same label-noise trade-off as Copy-Paste in vision.
    """
    if target.dim() != 3 or target.shape[-1] != classes:
        raise ValueError(
            "copy_paste_events expects frame-wise targets (B, cells, classes)"
        )
    x = x.clone()
    target = target.clone()
    batch, cells, _ = x.shape
    for i in range(batch):
        if float(torch.rand(1, generator=generator)) > p:
            continue
        active_classes = torch.nonzero((target > 0).any(dim=(0, 1))).flatten().tolist()
        if not active_classes:
            continue
        c = active_classes[
            int(torch.randint(0, len(active_classes), (1,), generator=generator))
        ]
        donors = [j for j in range(batch) if bool((target[j, :, c] > 0).any())]
        if not donors:
            continue
        donor = donors[int(torch.randint(0, len(donors), (1,), generator=generator))]
        idx = torch.nonzero(target[donor, :, c] > 0).flatten()
        k = int(torch.randint(0, len(idx), (1,), generator=generator))
        start, end = int(idx[k]), int(idx[k])
        while end + 1 < cells and target[donor, end + 1, c] > 0:
            end += 1
        while start - 1 >= 0 and target[donor, start - 1, c] > 0:
            start -= 1
        length = end - start + 1
        if length > cells:
            continue
        new_start = int(torch.randint(0, cells - length + 1, (1,), generator=generator))
        x[i, new_start : new_start + length] = x[donor, start : end + 1]
        target[i, new_start : new_start + length, c] = 1.0
    return x, target


def apply_recipe(
    x: torch.Tensor,
    target: torch.Tensor,
    recipe: dict,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the configured recipe augmentations to one batch.

    Order: Copy-Paste -> FilterAugment -> Freq-MixStyle -> mixup. Targets are
    mixed convexly under mixup; this is the same for PACEModel `(cells, 3C)` and
    frame-wise `(cells, C)` layouts. Unknown keys raise, so a typo cannot
    silently disable an augmentation.
    """
    allowed = {"mixup", "filter_augment", "freq_mixstyle", "copy_paste"}
    unknown = set(recipe) - allowed
    if unknown:
        raise ValueError(
            f"unknown recipe keys {sorted(unknown)}, expected {sorted(allowed)}"
        )

    if "copy_paste" in recipe:
        x, target = copy_paste_events(
            x, target, generator=generator, **recipe["copy_paste"]
        )
    if "filter_augment" in recipe:
        x = filter_augment(x, generator=generator, **recipe["filter_augment"])
    if "freq_mixstyle" in recipe:
        x = freq_mixstyle(x, generator=generator, **recipe["freq_mixstyle"])
    if "mixup" in recipe:
        x, perm, lam = mixup(x, generator=generator, **recipe["mixup"])
        target = lam * target + (1.0 - lam) * target[perm]
    return x, target
