# NOTES — deepinv 0.4.1 API surprises & version workarounds

Everything below was found by **running code** against the installed `deepinv` while
building these tutorials. Recorded so the next person doesn't rediscover them.

## Version / install

- **The version label was stale.** The repo is an *editable* install
  (`pip install -e .`). `deepinv.__version__` initially reported **0.3.6** (frozen
  editable metadata) while the checkout's `pyproject.toml` and PyPI both say **0.4.1**.
  Refreshed with `pip install -e . --no-deps` so `__version__` correctly prints `0.4.1`.
  The tutorials are pinned to **0.4.1** (= the current PyPI release, what
  `pip install deepinv` gives students) rather than to in-development `main`.
- Built/tested on `torch==2.9.1`, `numpy==2.2.6`, Python 3.11, **CPU**.

## Physics

- `generate_shepp_logan` lives at **`deepinv.utils.phantoms`** (not `deepinv.utils`),
  and returns a bare `(N, N)` tensor in `[0, 1]` — reshape to `(1, 1, N, N)` yourself.
- **`Tomography`**: always pass `normalize=True` explicitly (the default `None` warns).
  With `normalize=True`, `compute_sqnorm ≈ 1`, so a gradient `stepsize ≈ 1` works.
  - **Angles are in DEGREES.** `angles=N` → `linspace(0, 180, N+1)[:-1]`. A radian
    range silently produces a near-degenerate wedge.
  - `A_dagger(y)` defaults to a **slow least-squares solve** (~3 s @128px). Use
    **`phys.fbp(y)`** for fast filtered back-projection; `A_dagger(y, fbp=True) == fbp(y)`.
    `A_adjoint` is the raw (unfiltered) back-projection, not a reconstruction.
  - `condition_number(x)` is **slow** (~9 s @128px/20-angle, ~74 s @180-angle) and is a
    power-iteration *estimate* that is **not cleanly monotone** in the number of angles.
    For the ill-posedness story we instead build the explicit operator at 32px and SVD it
    (a few seconds) — that gives the clean "singular values decay to ~0" + Picard picture.
- `phys.A(x)` is **clean/deterministic**; **`phys(x)` (`__call__`) applies the noise model**.
  Attach noise via `noise_model=dinv.physics.GaussianNoise(sigma=...)`.
- **MRI** input is a **2-channel real** tensor `(B, 2, H, W)` (real, imag) — *not* a
  complex tensor. Build it with `torch.cat([x, torch.zeros_like(x)], 1)`; display
  `sqrt(c0² + c1²)`. Masks come from
  `dinv.physics.generator.GaussianMaskGenerator(img_size, acceleration=4).step()["mask"]`.
- **Blur**: `BlurFFT` (circular) **preserves** size; plain `Blur` defaults to
  `padding="valid"` and **shrinks** the image. The kernel helper moved:
  use `dinv.physics.functional.gaussian_blur(...)` (`dinv.physics.blur.gaussian_blur`
  is deprecated since 0.4.1).
- **RandomPhaseRetrieval** needs a **complex** input (`x.to(torch.cfloat)`); output is
  real `|Bx|²` of shape `(B, m)`.

## Optimization

- `optim_builder(iteration=...)` accepts strings `"GD" | "PGD" | "ADMM" | "HQS" | "PDCP"
  | "DRS" | "MD" | "PMD"` — **no `"FISTA"`**. FISTA is hand-rolled (Nesterov momentum
  around the PGD step) in notebook 04; arguably clearer for teaching anyway.
- **`params_algo["lambda"]` must be a float**, not an int: the cost function calls
  `lambda.flatten(...)`, which a Python `int` lacks (`AttributeError`). Cast with `float()`.
- **TV prox uses `gamma=`, not `ths=`** (`prior.prox(u, gamma=lambd*stepsize)`); passing
  `ths=` silently lands in `**kwargs` and does nothing. TV inner iterations are set at
  construction: `TVPrior(n_it_max=20)`, not via `params_algo`. (TV prox is also slightly
  non-deterministic due to a random init in the inner solver — fine for convergence.)
- Metrics live at **`deepinv.loss.metric`** (also `dinv.metric`): `PSNR()(estimate, reference)`
  returns a shape-`(B,)` tensor — order is **(estimate, reference)**; `.item()` for a float.
- Reconstruction call: `model(y, physics)` → `xhat`; `model(y, physics, x_gt=x,
  compute_metrics=True)` → `(xhat, metrics)` with keys `['psnr', 'cost', 'residual']`
  (each a list-of-lists, one per batch element). Pass `verbose=False` to suppress tqdm bars.

## Plug-and-Play / DPIR

- **Grayscale DRUNet works**: `DRUNet(in_channels=1, out_channels=1, pretrained="download")`
  downloads valid 1-channel weights.
- In `optim_builder`, the denoiser noise level is **`params_algo["g_param"]`** (the direct
  `PGD`/`HQS` classes instead take `sigma_denoiser=` — don't mix them). For the CT problem,
  `custom_init` must return **`{"est": (x0, x0)}`** (a 2-tuple), with the adjoint rescaled
  by `scaling = π / (2·angles)` for a sane starting magnitude.
- **`DPIR` needs an explicit grayscale denoiser** for 1-channel images:
  `DPIR(sigma=σ, denoiser=DRUNet(in_channels=1, out_channels=1, pretrained="download"))`.
  The default 3-channel DRUNet **crashes** on a 1-channel input (it concatenates a σ-map
  → channel-count mismatch). `DPIR` is a fixed 8-iteration HQS preset; call `dpir(y, phys)`.
- Absolute PSNRs depend on image scale: the verified recipe used `load_example("SheppLogan.png")`
  in `[0, 0.439]` (PnP ~28, DPIR ~34.6 dB); on the `[0, 1]` hero phantom the numbers are
  lower (PnP 24.2, DPIR 25.1) but the ordering FBP < TV < PnP < DPIR holds. (RED with light
  tuning landed below TV on this phantom, so notebook 5 shows PnP + DPIR and mentions RED.)

## Plotting (`deepinv.utils.plot`) — read this before trusting a figure

- **Default `dpi=1200`** → enormous PNGs. The `tc.save_images` helper sets `dpi=150`.
- **Titles render through mathtext/LaTeX.** Wrap math in `$…$`; a raw `^`, `_`, `||`, or
  `\nabla` outside `$…$` raises "Missing $ inserted". Mathtext also mangles a literal
  `->` and **drops everything after a `%`**. Use `$…$` or plain words.
- **A broken title can SILENTLY drop the figure** — the cell still runs and `nbconvert`
  still exits 0, but no PNG is written. **Always verify `figures/` on disk**, not just the
  notebook exit code. (This is why the build is followed by a figure-presence + visual check.)
- `plot` packs panels tightly and does not widen for many panels, so multi-line PSNR titles
  collide. `tc.save_images` now **auto-scales width by panel count** (≈ 3.3·n) when no
  `figsize` is given; pass an explicit `figsize` to override.
- `plot_curves` expects `{name: list-of-lists}` (batch × n_iter) and writes
  `<save_dir>/curves.png`. For the bespoke spectrum/Picard and ISTA-vs-FISTA plots we use
  plain matplotlib with the deck palette instead.

## Running the notebooks programmatically

- Notebook tooling (`nbconvert`, `nbformat`, `jupytext`) was installed into the env;
  a kernel was registered: `python -m ipykernel install --user --name deepinv-lecture`.
- Execute headless with:
  `python -m nbconvert --to notebook --execute --inplace NN.ipynb
   --ExecutePreprocessor.kernel_name=deepinv-lecture --ExecutePreprocessor.timeout=900`.

## Tuning notes

- Regularization strengths were tuned for the `[0,1]` Shepp–Logan hero (they differ from a
  box phantom): Tikhonov peaks at **λ≈3e-3** (broad plateau, 19.4 dB); TV needs a **small
  λ≈1e-3 with ~300 iterations** to clearly beat Tikhonov (21.4 dB). PnP wants `stepsize≈2.0`
  (≥2.5 diverges) with `g_param≈0.01` over ~80 iterations (24.2 dB).
