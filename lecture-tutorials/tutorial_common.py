"""Shared helpers for the DeepInverse "Inverse Problems" lecture tutorials.

Every notebook imports this module so the whole set shares one device, one seed,
the lecture deck's color palette, the one hero object, a consistent PSNR/SSIM
readout, and a save-to-``figures/`` helper.

The spine equation the whole lecture restates, one term at a time:

    x_hat = argmin_x   D(A x, y)      +      R(x)
                     |-data fidelity-|     |--prior--|
                     (physics + noise)     (hand-crafted -> learned)

Every tutorial keeps **D** (the physics) fixed and makes the prior **R** smarter:
Tikhonov -> TV/L1 (sparsity) -> Plug-and-Play / RED (a learned denoiser).

Verified against deepinv 0.4.1 (see ../README.md and ../NOTES.md).
"""

import matplotlib

matplotlib.use("Agg")  # render to files, not a GUI window (safe in notebooks & CI)

# Silence benign warnings BEFORE importing deepinv so the lecture output stays
# readable: deepinv/torch emit UserWarnings (meshgrid indexing, the deprecated
# ``.device`` attribute, the Tomography ``normalize`` default) and tqdm warns when
# ipywidgets is absent. Comment these out while debugging.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*IProgress.*")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import deepinv as dinv
from deepinv.loss.metric import PSNR, SSIM
from deepinv.utils.phantoms import generate_shepp_logan


# --------------------------------------------------------------------------- #
# Reproducibility & device
# --------------------------------------------------------------------------- #
SEED = 0


def set_seed(seed: int = SEED) -> None:
    """Seed torch + numpy so every run of every notebook is identical."""
    torch.manual_seed(seed)
    np.random.seed(seed)


# CPU by default for portability: a laptop with no GPU reproduces the whole
# lecture in a couple of minutes. deepinv's auto-picker (cuda > mps > cpu) is one
# line away if you want to go faster:  DEVICE = dinv.utils.get_device()
DEVICE = torch.device("cpu")

set_seed()
print(f"deepinv {dinv.__version__} | torch {torch.__version__} | device {DEVICE}")


# --------------------------------------------------------------------------- #
# Lecture deck palette  (x = object, A = operator, y = measurement)
# --------------------------------------------------------------------------- #
PALETTE = {
    "x": "#D7263D",      # object / ground truth  -> red
    "A": "#1F4E79",      # forward operator       -> navy
    "y": "#2E7D32",      # measurements           -> green
    "slate": "#44546A",
    "gray": "#9a9a9a",
}
_CURVE_COLORS = [PALETTE["A"], PALETTE["x"], PALETTE["y"], PALETTE["slate"], PALETTE["gray"]]

# Slide-friendly matplotlib defaults.
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.prop_cycle": plt.cycler(color=_CURVE_COLORS),
    }
)

# The spine equation as a LaTeX string, for reuse in markdown / figure suptitles.
SPINE = (
    r"$\hat{x} = \arg\min_x\ "
    r"\underbrace{D(Ax, y)}_{\mathrm{data\ fidelity}} + "
    r"\underbrace{R(x)}_{\mathrm{prior}}$"
)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# The one hero object: a Shepp-Logan phantom, our photoacoustic/CT stand-in.
# --------------------------------------------------------------------------- #
def load_hero(size: int = 128) -> torch.Tensor:
    """Return the shared ``(1, 1, size, size)`` phantom in ``[0, 1]`` on ``DEVICE``.

    This is the recurring "object" x across tutorials 2-5, so the narrative has a
    single through-line. (Tutorial 1's operator gallery instead uses
    modality-appropriate images: a brain slice for MRI, a natural image for blur.)
    """
    phantom = generate_shepp_logan(size)            # (size, size) tensor in [0, 1]
    return phantom.reshape(1, 1, size, size).to(DEVICE)


# --------------------------------------------------------------------------- #
# The fixed physics D we keep across tutorials 2-5 (sparse-view CT = PAT stand-in).
# --------------------------------------------------------------------------- #
def ct_physics(angles: int, sigma: float = 0.0, size: int = 128):
    """A normalized sparse-view CT (Radon) operator, our stand-in for the PAT wave
    operator: linear, ill-posed, limited-view.

    ``normalize=True`` rescales A so that ||A^T A|| ~ 1, which makes a gradient
    stepsize ~ 1 work out of the box (see ``stepsize_for`` below). Pass ``sigma>0``
    to attach Gaussian measurement noise (applied by ``physics(x)``, not ``A(x)``).
    """
    noise = dinv.physics.GaussianNoise(sigma=sigma) if sigma > 0 else None
    return dinv.physics.Tomography(
        angles=angles,
        img_width=size,
        normalize=True,
        noise_model=noise,
        device=DEVICE,
    )


def stepsize_for(physics, x: torch.Tensor) -> float:
    """Gradient stepsize 1 / ||A^T A|| for the data term 0.5||Ax - y||^2.

    ``compute_sqnorm`` returns the squared spectral norm (the Lipschitz constant of
    the data-term gradient); with ``normalize=True`` this is ~1.
    """
    return 1.0 / physics.compute_sqnorm(x, tol=1e-4, verbose=False).item()


# --------------------------------------------------------------------------- #
# Metrics: the same readout everywhere, so "visibly better" becomes a number.
# --------------------------------------------------------------------------- #
_psnr = PSNR()
_ssim = SSIM()


def psnr(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """PSNR(estimate, reference) in dB as a python float."""
    return _psnr(x_hat, x).mean().item()


def ssim(x_hat: torch.Tensor, x: torch.Tensor) -> float:
    """SSIM(estimate, reference) as a python float."""
    return _ssim(x_hat, x).mean().item()


def title_psnr(name: str, x_hat: torch.Tensor, x: torch.Tensor) -> str:
    """Figure title like ``'PnP\\nPSNR 27.9 dB'`` for consistent labelling."""
    return f"{name}\nPSNR {psnr(x_hat, x):.1f} dB"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save_images(imgs, titles=None, fname=None, cmap="gray", cbar=False,
                suptitle=None, figsize=None, **kwargs) -> None:
    """Plot a row of images via ``deepinv.utils.plot`` and save to ``figures/<fname>``.

    ``imgs`` is a list of ``(1, 1, H, W)`` tensors (or a ``{title: tensor}`` dict).
    When ``figsize`` is not given it scales with the number of panels so that
    multi-line PSNR titles don't collide (``deepinv.utils.plot`` packs tightly).
    """
    save_fn = str(FIG_DIR / fname) if fname else None
    if figsize is None:
        n = len(imgs) if hasattr(imgs, "__len__") else 1
        figsize = (max(4.0, 3.3 * n), 4.3 if suptitle else 4.0)
    dinv.utils.plot(
        imgs,
        titles=titles,
        cmap=cmap,
        cbar=cbar,
        suptitle=suptitle,
        figsize=figsize,
        save_fn=save_fn,
        show=False,
        dpi=150,
        **kwargs,
    )
    if save_fn:
        print(f"saved {save_fn}")


def save_curves(curves: dict, fname, xlabel="iteration", ylabel="cost",
                title=None, logy=True, logx=False, markers=False) -> None:
    """Convergence / sweep plot in the deck palette; saves ``figures/<fname>``.

    ``curves`` maps a label to a sequence of y-values, e.g.
    ``{"ISTA": ista_cost, "FISTA": fista_cost}``. The x-axis is the index unless a
    ``(xs, ys)`` tuple is given as the value.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, (label, vals) in enumerate(curves.items()):
        if isinstance(vals, tuple) and len(vals) == 2:
            xs, ys = vals
        else:
            xs, ys = range(len(vals)), vals
        ax.plot(
            xs, ys, label=label, color=_CURVE_COLORS[i % len(_CURVE_COLORS)],
            lw=2, marker="o" if markers else None, ms=4,
        )
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = str(FIG_DIR / fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
