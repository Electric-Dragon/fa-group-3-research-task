# C. elegans · Model 2 — Plan

Source: Demir et al., eLife 2020;9:e52781 ("Main Paper.pdf"), Materials and methods, Eqs. 45–47
and the parameter list below them. The reasoning behind each choice is in
[DESIGN_NOTES.md](DESIGN_NOTES.md).

> **Status (2026-10-01): built.** The paper's pattern-formation model runs in one interactive
> window and in a headless check script. A bacteria/swarming extension was built, did not produce
> swarming, and was removed (DESIGN_NOTES.md, Section 5). Open items are in Section 7.

## Goal

Reproduce what the paper has equations for, without inventing new mechanisms. One window: worms on
the left, oxygen on the right, a handful of controls.

Not in scope: swarming and bacteria fields, biofilm/PA14 lawns, lawn edges, notebooks, analysis
modules, graphs.

## 1. The model (the paper, exactly)

Each worm is an agent that moves at speed V(O) in a straight line and picks a new random heading at
rate τ. This random walk is the paper's starting point (Eqs. 3–4), and its density obeys

    ∂W/∂t = ∇·[ (V²/2τ) ∇W ] + ∇·[ (V/2τ)(∂V/∂O) W ∇O ]      (Eq. 45)
    ∂O/∂t = D_O ∇²O + f (O_am − O) − k_c W                    (Eq. 46)
    V(O)  = a O² + b O + c                                    (Eq. 47)

Agents never sense the gradient. Aerotaxis (β) emerges from speed varying with oxygen. Oxygen use by
the bacteria that worms gather is part of k_c W, as in the paper.

### Parameters (paper values)

| Symbol | Meaning | Paper (SI) | In code (mm, s) |
|---|---|---|---|
| a, b, c | speed parabola, O as a fraction | 1.89e-2, −3.98e-3, 2.25e-4 m/s | same, × 1e3 → mm/s |
| τ | tumbling rate | 0.5 /s | 0.5 /s |
| D_O | oxygen diffusion | 2e-9 m²/s | 2e-3 mm²/s |
| f | oxygen penetration (regeneration) | 0.65 /s | 0.65 /s |
| k_c | oxygen consumption by worms + bacteria | 7.3e-10 /s (W in worms/m²) | 7.3e-4 /s per worm/mm² |
| O_am | ambient oxygen | 0 – 0.21 | 0.01 – 0.21 (slider) |
| W | worm density | 1 – 90 worms/mm² | same |
| L | box side | 2 cm | 20 mm |

The PDF text drops minus signs; b < 0 is the only reading that gives a sensible curve (A1): 225 µm/s
at 0 %, minimum ≈ 16 µm/s at 10.5 %, 222 µm/s at 21 %, the npr-1-type U shape.

### Strains

- **npr-1**: the paper's parabola.
- **N2**: the paper gives no coefficients. The same parabola form is least-squares fitted to the
  Fig. 2b N2 points: a = 9.02e-3, b = −2.64e-3, c = 2.07e-4 (A4). The window labels it "fitted to
  Fig. 2b". Whether to keep it is an open question (Section 7).

## 2. Numerics (as built)

- **Box and grids:** periodic 20 mm box. Oxygen on a 512² grid (dx ≈ 39 µm, which resolves
  √(D_O/f) ≈ 55 µm). Worm density on a 128² grid (A11).
- **Time step:** dt = 0.05 s, fixed. Each step a worm moves ≤ 11 µm and tumbles with probability
  τ·dt = 0.025.
- **Worm density:** cloud-in-cell deposit, then a Gaussian blur with σ = 0.25 mm (A6). The oxygen grid
  reads it bilinearly.
- **Oxygen:** explicit update. The number of sub-steps is computed from the current D_O and f, so
  the update stays stable at any slider setting. O is kept ≥ 0 (A5).
- **Worms:** each worm reads O at its position by bilinear interpolation.
- **Start:** uniform random positions and headings; oxygen at ambient.
- **Worm count:** N = density × 400 mm², so 400 – 36,000 agents.
- **Engine:** Taichi 1.7.4 on Metal (`ti.gpu`), with `ti.cpu` as a fallback. It needs Python ≤ 3.12.
- **Speed:** about 110 µs per step at 16k worms on the M5 GPU, so 10 simulated minutes take ≈ 1.3 s.

## 3. Window (Taichi GGUI)

- **Left panel:** every worm drawn as a 0.5 mm dark segment along its heading; nothing subsampled.
- **Right panel:** oxygen heat map, 0–21 %, with a colour bar.

| Control | Values | Applies |
|---|---|---|
| Run / Pause (or Space), Reset | — | — |
| Strain | npr-1 / N2 | live |
| Density | 1 – 90 worms/mm² | on Reset |
| Ambient O₂ | 1, 3, 7, 10, 14, 17, 21 % | live |
| Regeneration f | ×0.1, 0.2, 0.5, 1, 2, 5, 10 of the paper value | live |
| Diffusion D_O | same steps | live |
| Absorption k_c | same steps | live |
| Steps / draw | 1, 2, 5 … 2000, 5000 | live, display speed only |

Readout: worm count and density, simulated time, mean / min O₂, steps per second.

## 4. Files

    Celegans-Model2/
      PLAN.md, DESIGN_NOTES.md, README.md
      requirements.txt     taichi==1.7.4, numpy, pytest
      model2/
        params.py          every constant, with its source; strains; instability criterion
        sim.py             Taichi fields and kernels: deposit, blur, oxygen, move
        app.py             window, controls, drawing, --screenshot mode
        __main__.py        python -m model2
      check.py             headless density × ambient grid → results/grid_<strain>.png
      tests/test_sim.py    numeric checks
      results/             grid images and a screenshot

## 5. Checks

| Check | How | Result |
|---|---|---|
| Speed parabola matches the paper's values | test | pass |
| Worm density sums to the worm count | test | pass |
| Worms stay in the box | test | pass |
| No worms: O relaxes to O_am | test | pass |
| Uniform worms: mean O = O_am − k_c W/f | test | pass |
| npr-1 patterns where the criterion predicts | `check.py` | pass: 21 %: dots at 40, stripes at 60, faint holes at 90 worms/mm²; 17 %: onset at 40 |
| Patterns over the range of Fig. 3c | `check.py` | **no**: nothing at 14 % and below |
| N2 patterns at high density (Fig. 2e) | `check.py --strain N2` | **no**: N2 never patterns |
| 1 % O₂ breaks patterns | `check.py` | uniform at 1 % (patterns don't form; breaking up existing ones not yet tested) |
| Coarsening over time (Fig. 4a–c) | run longer in the window | not yet checked |
| Live window | `python -m model2` | rendered headless only; not yet used interactively by Claude |

## 6. Build history

1. Engine, headless: done.
2. Window: done.
3. Bacteria extension: built, tried in three variants, removed.
4. Density × ambient grid check: done.

## 7. Open items

- **N2:** keep the fitted parabola (paper form, paper data, our coefficients), or drop N2 and keep
  only the paper's own curve?
- **Fig. 3c gap:** the paper's numbers, read as A1–A2, only give patterns at 17–21 %. Re-reading the
  paper's units and signs, or its supplementary material, is the next thing to try.
- **Coarsening and 1 % break-up:** quick checks still to run.
