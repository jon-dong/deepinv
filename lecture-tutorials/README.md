# Inverse Problems with DeepInverse — lecture tutorials

Runnable [DeepInverse](https://deepinv.github.io/) tutorials accompanying the lecture
*"Inverse Problems: From Physics to Deep Learning"*. They are the **"do it for real
with the library"** companion to the from-scratch numpy notebooks: same teaching arc,
but using the actual `deepinv` API, so you can `pip install deepinv` and reproduce or
extend everything after the talk.

## Getting started

Verified end-to-end on **CPU** (each notebook runs in a couple of minutes). No GPU needed.

```bash
# from this folder
uv venv && source .venv/bin/activate          # or: python -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt            # or: pip install -r requirements.txt
python -m ipykernel install --user --name deepinv-lecture
jupyter lab                                    # open 01_forward_operators.ipynb first, then 02 … 05
```

In Jupyter, select the **deepinv-lecture** kernel (or just use your venv's Python), and
run each notebook top-to-bottom. Figures render inline.

**Versions (pinned for reproducibility):** `deepinv==0.4.1`, `torch==2.9.1`, Python 3.11.
These are the pins that matter; [`requirements.txt`](requirements.txt) lists the rest, and
each notebook prints `deepinv.__version__` on import.

## The one equation

Every notebook restates **one equation** and changes exactly **one term** of it:

```
x̂ = argmin_x   D(A x, y)        +        R(x)
              └ data fidelity ┘          └ prior ┘
              (physics + noise)     (hand-crafted → learned)
```

The physics **D** (a sparse-view CT operator — linear, ill-posed, limited-view) is held
**fixed**; only the prior **R** gets smarter: Tikhonov → TV/sparsity → a learned denoiser.

## The tutorials

| # | Notebook | Lecture § | (D, R, algorithm) | Figures | Takeaway |
|---|----------|-----------|-------------------|---------|----------|
| 1 | [01_forward_operators](01_forward_operators.ipynb) | §2 | the operator **A** | [ct](figures/01_ct.png), [mri](figures/01_mri.png), [blur](figures/01_blur.png), [inpainting](figures/01_inpainting.png) | Wildly different physics (CT, MRI, blur, inpainting) share **one** interface: `.A`, `.A_adjoint`, `.A_dagger`. That operator **A** is what lives inside the data term **D**. |
| 2 | [02_naive_inverse_illposed](02_naive_inverse_illposed.ipynb) | §3.1–3.5 | no prior (R = 0) | [naive recon](figures/02_naive_recon.png), [singular values](figures/02_singular_values.png) | The operator has tiny singular values; the pseudo-inverse divides measurements (incl. noise) by them, so noise **explodes**. We need a prior. |
| 3 | [03_tikhonov_l2](03_tikhonov_l2.ipynb) | §3.6–3.9 | L2 · **Tikhonov** · direct solve | [λ sweep](figures/03_lambda_sweep.png), [bias–variance](figures/03_bias_variance.png) | A quadratic prior has a closed form — the regularized normal equations, = damped SVD / Wiener filtering. One knob λ trades noise for blur (peak **16.9 dB** on the hard 20-view problem), but **cannot recover edges**. |
| 4 | [04_sparsity_tv_fista](04_sparsity_tv_fista.ipynb) | §3.10–3.16 | L2 · **TV** · ISTA/FISTA | [TV vs Tikhonov](figures/04_tv_vs_tikhonov.png), [ISTA vs FISTA](figures/04_ista_vs_fista.png) | A sparsity prior recovers **sharp edges** where L2 blurs (**21.4 dB**); FISTA's momentum converges far faster than ISTA. |
| 5 | [05_pnp_red](05_pnp_red.ipynb) | §4.8 | L2 · **PnP (DRUNet)** + DPIR · PGD/HQS | [comparison](figures/05_comparison.png) | Keep the physics **D**, swap the hand-crafted prior for a **pretrained denoiser** → a big jump with no per-problem training (PnP **24.2 dB**, DPIR **25.1 dB**). |

The core arc on the fixed 40-view CT problem (σ = 0.02 noise, Shepp–Logan phantom):

> FBP baseline **≈ 15 dB** (streaky) → Tikhonov **19.4** → TV **21.4** → PnP **24.2** → DPIR **25.1 dB**
>
> (Notebooks 2–3 stress a harsher **20-view** version of the same operator: with no prior the pseudo-inverse collapses to ~8 dB, and Tikhonov lifts it to ~16.9.)

Each reconstruction figure reports **PSNR** in its title, so "visibly better" is always a number.

## Layout

- `01_forward_operators.ipynb … 05_pnp_red.ipynb` — the tutorials, runnable top-to-bottom.
- [`tutorial_common.py`](tutorial_common.py) — shared helpers imported by every notebook
  (`import tutorial_common as tc`): device, seed, the deck palette, the one shared object
  `tc.load_object()`, the fixed CT physics `tc.ct_physics()`, `tc.psnr/ssim/title_psnr`,
  and figure helpers `tc.show_images` / `tc.show_curves`. Read it once; it keeps all five
  notebooks looking and behaving like one family.
- [`figures/`](figures/) — pre-rendered copies of the key figures (drop straight into slides).
- [`requirements.txt`](requirements.txt) — pinned environment.
- [`NOTES.md`](NOTES.md) — `deepinv 0.4.1` API surprises and version workarounds hit while building these.
