# C. elegans · Model 2 — Design notes

These notes record why Model 2 looks the way it does: the paper's equations, the original
intuitions behind the model, what changed during the design discussion, and every assumption
made along the way. The as-built plan and check results are in [PLAN.md](PLAN.md).

**Current state (2026-10-01):** Model 2 simulates only the paper's pattern-formation model
(Eqs. 45–47, paper parameter values). A bacteria/swarming extension was built and then removed
(Section 5).

Source paper: Demir, E., Yaman, Y. I., Basaran, M., & Kocabas, A. (2020). *Dynamics of pattern
formation and emergence of swarming in Caenorhabditis elegans.* eLife 9:e52781.
("Main Paper.pdf" in the repo root.)

---

## 1. What the paper says (key points)

- Bacteria use oxygen. C. elegans seeks low oxygen as a way of finding bacteria (food). In a dense
  population, worms, bacteria and oxygen become tightly coupled.
- When thousands of worms feed together, worms gathering bacteria into aggregates and the
  resulting oxygen depletion create unstable conditions that trigger phase separation.
- Phase separation is driven by a sudden change in the animals' motility. Collective dynamics
  are "driven by slowing down response of animals".
- The sensitivity of the oxygen-sensing neurons controls the dynamics. It is captured entirely by
  the oxygen-dependent speed curve **V(O)**, which can be measured experimentally (Fig. 2b).
- V(O) sets a competition between dispersion and aerotaxis (Fig. 2c):
  - **low O₂ (< ~5 %)**: worms speed up; dispersion and *reversed* aerotaxis (β < 0) win, and the
    worms spread out.
  - **optimum O₂ (7–10 %)**: worms are almost stationary (D ≈ 0, β ≈ 0).
  - **high O₂**: aerotaxis dominates (β > 0) and the worms aggregate.
- npr-1 has a sharp aerotactic response; N2 has a weaker, flatter one and a broader oxygen
  preference.
  - N2 patterns barely change when ambient oxygen is scanned.
  - npr-1 forms patterns even at low density, while N2 forms them only at high density: density
    can make up for low sensitivity.
- Raising worm density or lowering ambient oxygen turns dots into stripes, then holes (Fig. 2d, 3c).
- Patterns coarsen over time and merge into one large cluster (Fig. 4a–c).
- **Swarming:**
  - Inside a swarm, the front edge has more bacteria than the back. Worms eat the bacteria, and
    bacteria diffuse in from the front.
  - The food profile has decay length λ ≈ 2–4 mm, with λ² = D_b/f_b. This length sets the swarm's
    width.
  - Small clusters have symmetric food and no gradient.
  - Worms move faster toward the back, matching the off-food response: without bacteria, animals
    move fast. The balance of eating and diffusion makes the swarm move across the lawn
    (centre-of-mass speed 0.1–2 µm/s, rising over about 200 min, Fig. 4 supplement).
- npr-1 does not cluster on very thick lawns and avoids their thickest regions. Biofilm lawns
  (B. subtilis, PA14) change the dynamics strongly, and on PA14 the V(O) curves change (Fig. 6).
- The paper names four essential factors:
  1. Water flow around moving worms gathers bacteria into aggregates (the first, critical step).
  2. Oxygen-dependent motility sets the balance between aerotaxis and dispersion.
  3. Population density can make up for low neuronal sensitivity.
  4. Bacterial consumption creates a gradient across an aggregate, which starts swarming.
     Bacterial diffusibility and biofilm formation also matter.

---

## 2. Original intuitions, and how each one held up

| Intuition | Outcome |
|---|---|
| A worm can be a simple agent with a fixed speed function of local O₂. | **Correct, and it is exactly the paper's model.** The paper starts from a random walk at speed V(O) with tumbling rate τ (Eqs. 3–4). |
| …plus a separate "sensitivity to O₂". | **Dropped.** In the paper, sensitivity is the *slope* of V(O): β = (V/2τ)·∂V/∂O. A separate gradient-following term would count the same effect twice. Agents never sense gradients; they gather where they move slowly. |
| The oxygen field should be adjustable: regeneration, diffusion, absorption, ambient. | **Correct.** These are the four terms of the paper's oxygen equation: f, D_O, k_c, O_am. |
| V(O) can be a quadratic or quartic. | **Quadratic is right.** The paper uses V = aO² + bO + c and gives the coefficients (Eq. 47). An earlier suggestion to use piecewise interpolation instead was retracted because it departs from the paper. A quartic is not needed. |
| Exact parameters don't need to be found immediately. | Most are in the paper's parameter list (Section 4). The N2 curve is not, so it is fitted (A4). |
| Sliders could be discrete values taken from the graph. | **Kept for ambient O₂.** It snaps to the measured points: 1, 3, 7, 10, 14, 17, 21 %. Local oxygen still varies continuously in the simulation. |
| Include both N2 and npr-1. | **Kept.** npr-1 uses the paper's parabola; N2 uses the same form fitted to Fig. 2b. |
| Optional bacteria field. | **Built as a labelled extension, then removed** (Section 5). The paper gives no bacteria equation, and the extension did not produce swarming. |
| Don't overcomplicate with many sliders and graphs. | **Kept.** One window, two panels, eight controls, a text readout, no graphs, no notebook, no analysis modules. |
| Model 1 looked grainy and the number of worms was hard to judge. | **Addressed.** Every worm is drawn, none subsampled, as a short segment along its heading, and the count and density are always shown. Model 1 drew about 3,500 of about 21,700 agents as 1.2-pixel dots at 40 % opacity. |
| Step size should be adjustable to speed up the visualisation, as in Model 1. | **Kept** as *Steps / draw* (how many physics steps between redraws). The physics time step stays fixed, so results don't depend on display speed. |
| Use an engine that takes advantage of Apple Silicon CPU + GPU. | **Taichi** on Metal (Section 7). |
| Use the paper's equations only. | **Done.** The model is the paper's Eqs. 45–47 with its parameter values. What remains outside the paper is listed in Section 6. |

---

## 3. Decisions from the discussion

1. **The scope is pattern formation only.** The first plan covered patterns and swarming, with a
   bacteria switch. After the first build (Section 11) the bacteria part was removed: the paper
   gives no equations for it, and the version built from its descriptions did not swarm.
2. **The model reproduces what the paper has, without inventing new mechanisms.**
3. **Engine: Taichi** rather than NumPy/Numba/MLX/PyTorch/JAX. It runs the same kernels on the GPU
   (Metal) or the CPU, and it has its own native window with sliders and GPU drawing. That removes
   Jupyter widgets and matplotlib PNG frames, which were a source of Model 1's slowness and
   clutter.
4. **Simplicity over coverage.** Biofilm/PA14 lawns, lawn edges, phase-diagram tooling and graphs
   are out of scope.
6. **Engine detail settled during the build:** worm density moved to a coarser grid for speed (A11).
5. **Real units** (mm and s in the code, oxygen as a fraction), so the paper's numbers can be used
   directly after a unit conversion.

---

## 4. Equations (the paper, Materials and methods)

**Worms (agents).** Each worm moves straight at speed V(O) and picks a uniformly random new
heading at rate τ. Its density W obeys (Eqs. 3–8, 45):

    J = −(V / 2τ) ∇(V W)                                   (flux, 2-D)
    ∂W/∂t = ∇·[ D_W ∇W ] + ∇·[ β W ∇O ]
    D_W = V² / 2τ                                           (motility-dependent dispersion)
    β   = (V / 2τ) · ∂V/∂O                                  (aerotactic coupling)

**Oxygen** (Eq. 46). Diffusion on the agar surface, uptake from the air, and consumption by worms
and the bacteria they carry:

    ∂O/∂t = D_O ∇²O + f (O_am − O) − k_c W

**Speed** (Eq. 47):

    V(O) = a O² + b O + c

**Uniform state and instability criterion** (main text). Patterns start when

    W_eq · β · k_c  >  f · D_W ,      with  O_eq = O_am − k_c W_eq / f

Density and aerotactic response favour patterns; dispersion works against them.

### Parameters (paper)

| Symbol | Meaning | Value |
|---|---|---|
| a, b, c | speed parabola (O as a fraction, V in m/s) | 1.89e-2, −3.98e-3, 2.25e-4 |
| τ | tumbling rate | 0.5 /s |
| D_O | oxygen diffusion | 2e-5 cm²/s = 2e-9 m²/s |
| f | oxygen penetration from the air | 0.65 /s |
| k_c | oxygen consumption by worms + bacteria | 7.3e-10 /s (W in worms/m²) |
| O_am | ambient oxygen | 0 – 0.21 |
| W | worm density, uniform stage | 1 – 90 worms/mm² |
| L | box side | 2 cm |

The paper says f and k_c were chosen to match the observed width of domain boundaries and time
dynamics. The others were measured.

The npr-1-type parabola gives 225 µm/s at 0 %, a minimum of ≈ 16 µm/s at 10.5 %, and 222 µm/s at
21 %.

### V(O) data read from Fig. 2b (µm/s, OP50 lawn)

| O₂ % | 1 | 3 | 7 | 10 | 14 | 17 | 21 | 30 |
|---|---|---|---|---|---|---|---|---|
| npr-1 (on food) | 168 | 70 | 20 | 18 | 18 | 55 | 148 | 155 |
| N2 (on food) | 193 | 137 | 37 | 35 | 33 | 32 | 39 | 50 |
| N2 off food | 157 | 176 | 163 | 176 | 188 | 193 | 177 | 180 |

All values are read by eye from the figure.

---

## 5. Bacteria extension (built, then removed)

The paper describes the role of bacteria in swarming (Section 1) but gives no equations for it. The
first build added the smallest equations that matched its statements, as a labelled switch:

    ∂B/∂t = D_b ∇²B − k_b W B                   B ∈ [0, 1]; 1 = full lawn, 0 = eaten out
    V     = B · V_strain(O) + (1 − B) · V_off    blend toward the off-food speed (175 µm/s)

Result: food was eaten, worms sped up everywhere as it ran low, and the patterns dissolved. No
swarm formed, including in three variants (Section 11). Because none of it came from the paper and
it did not reproduce the paper's swarming either, it was removed.

What the model keeps from the paper is the bacteria's role as **the oxygen sink**. Worms gather
bacteria, and the gathered bacteria use oxygen. The paper folds this into k_c W, and so does Model 2.

---

## 6. Assumptions (everything not taken directly from the paper)

| # | Assumption | Why |
|---|---|---|
| A1 | b is negative (and a, c positive) in the speed parabola. | The PDF text drops minus signs; this is the only choice that gives a U-shaped curve matching Fig. 2b. |
| A2 | k_c's units take W in worms/m². | This is the only reading that gives sensible oxygen levels (k_c W comparable to f·O_am). |
| A3 | The paper's parabola is the **npr-1** curve. | Its minimum at 10.5 % and steep rise toward 21 % match npr-1, not N2. The paper does not say. |
| A4 | N2 uses the same parabola form, least-squares fitted to the Fig. 2b N2 points: a = 9.02e-3, b = −2.64e-3, c = 2.07e-4 (minimum ≈ 15 µm/s at 14.6 %). | The paper gives no N2 coefficients. The fit is poor because the measured N2 curve is L-shaped. |
| A5 | Oxygen is clamped at ≥ 0. | With a linear sink, O_eq goes negative at high density and low ambient (e.g. 90 worms/mm² at 7 %). The paper does not say how this is handled. |
| A6 | Worm density on the grid uses cloud-in-cell deposit plus Gaussian smoothing, σ ≈ 0.25 mm. | The paper solves a continuum equation. At realistic densities there are about 0.03 worms per grid cell, so unsmoothed agent counts are mostly noise. This is a numerical choice. |
| A7 | Periodic square box, uniform random start. | Matches the paper's simulation setup ("uniformly distributed with additional noise"). |
| A8 | The tumbling rate is constant and the same for both strains. | It is constant in the paper. Strains differ only in V(O). |
| A9 | *(removed)* Bacteria extension: its equations, V_off, D_b and k_b. | Removed with the bacteria part (Section 5). |
| A10 | The ambient O₂ slider snaps to 1, 3, 7, 10, 14, 17, 21 %. | These are the measured points in Fig. 2b; this was the original intuition. |
| A11 | Worm density is computed on a coarser 128² grid (0.156 mm cells) and sampled bilinearly by the 512² oxygen grid. | Speed: the smoothing step went from 270 µs to 17 µs. The density is smoothed over σ = 0.25 mm anyway (A6), so the effective footprint barely changes. |

---

## 7. Engine and numerics

- **Taichi 1.7.4**, Metal GPU backend (`ti.gpu`), CPU fallback (`ti.cpu`). It needs Python ≤ 3.12
  (this machine has Homebrew's 3.12 alongside 3.14).
- **Grids:** oxygen on 512² over 2 cm (dx ≈ 39 µm, resolving √(D_O/f) ≈ 55 µm). Worm density on
  128² (A11).
- **Time step** dt = 0.05 s, fixed and hidden. Each step a worm moves ≤ 11 µm and tumbles with
  probability τ·dt = 0.025.
- **Oxygen diffusion** is explicit, with enough sub-steps for stability at any slider value.
- **Worm count** N = density × 400 mm², so 400 – 36,000 worms.
- **Measured speed** on this Apple M5: about 110 µs per step at 16k worms (GPU). That is 10 simulated
  minutes in about 1.3 s. Before the coarse density grid it was about 370 µs, mostly the blur.
- **Display speed** is set by Steps / draw. It changes how often the screen redraws, not the physics.

---

## 8. Interface

- **Left panel:** every worm drawn as a 0.5 mm dark segment along its heading.
- **Right panel:** oxygen heat map (0–21 %) with a colour bar.

| Control | Values | Applies |
|---|---|---|
| Run / Pause (or Space), Reset | — | — |
| Strain | npr-1 / N2 | live |
| Density | 1 – 90 worms/mm² | on Reset |
| Ambient O₂ | 1, 3, 7, 10, 14, 17, 21 % | live |
| Regeneration f | ×0.1, 0.2, 0.5, 1, 2, 5, 10 of the paper value | live |
| Diffusion D_O | same steps | live |
| Absorption k_c | same steps | live |
| Steps / draw | 1, 2, 5 … 2000, 5000 | live |

The multiplier and speed sliders are discrete steps: Taichi's window has no log-scale slider, and
discrete steps make ×1 easy to find again. Each slider shows a step number, with the real value on
the line below.

Readout: worm count, density, simulated time, mean / min O₂, steps per second.

---|---|---|
| Run / Pause, Reset | — | — |
| Strain | npr-1 / N2 | live |
| Worm density | 1 – 90 worms/mm² | on Reset |
| Ambient O₂ | 1, 3, 7, 10, 14, 17, 21 % | live |
| Regeneration f | ×0.1 – ×10 of the paper value | live |
| Diffusion D_O | ×0.1 – ×10 of the paper value | live |
| Absorption k_c | ×0.1 – ×10 of the paper value | live |
| Steps / draw | 1 – 5000 | live |

Readout: worm count, density, simulated time, mean / min O₂, steps per second.

---

## 9. Predictions vs results

The criterion in Section 4, evaluated with the paper's numbers (and A1–A4), against the simulation
(`check.py`, 10 simulated minutes):

| Ambient O₂ | npr-1 predicted | npr-1 simulated | N2 predicted | N2 simulated |
|---|---|---|---|---|
| 1 – 14 % | none | none | none | none |
| 17 % | ≈ 27 – 50 worms/mm² | onset at 40 | none | none |
| 21 % | ≈ 35 – 89 worms/mm² | dots 40, stripes 60, faint holes 90 | none | none |

- **The simulation agrees with the criterion.** So the agents reproduce the paper's equations; the
  smoothing (A6, A11) does not visibly shift where patterns start.
- **The npr-1 range is narrower than Fig. 3c**, which shows patterns at lower ambient oxygen too.
- **N2 never patterns**, while the experiments show N2 patterns at about 20 worms/mm².
- Both gaps come from the parameters, not the numerics. Suspects: the readings A1–A2, or the
  parabola form for N2 (A4).

## 10. Validation targets

| Target | Status |
|---|---|
| Uniform state: O settles to O_am − k_c W/f | **met** (test) |
| Strain contrast: npr-1 patterns at lower density than N2 (Fig. 2e–f) | **not met**: N2 never patterns |
| Pattern sequence: dots → stripes → holes as density rises (Fig. 2d) | **met** at 21 % |
| Same sequence as ambient O₂ falls (Fig. 3c) | **not met**: nothing below 17 % |
| Low oxygen: 1 % breaks patterns apart | partly: uniform at 1 %; breaking up existing patterns not yet tested |
| Coarsening: patterns merge over time (Fig. 4a–c) | not yet checked |

## 11. Results (2026-10-01)

**The paper's model.** `check.py`, 10 simulated minutes, density × ambient O₂ grid
(images in `results/`):

- **npr-1** forms patterns only where the paper's criterion predicts them:
  - At 21 %: dots at 40 worms/mm², stripes at 60, faint holes at 90. This is the paper's
    dots → stripes → holes sequence (Fig. 2d).
  - At 17 %: an early pattern at 40 worms/mm².
  - At 14 % and below: uniform at every density.
- **N2** (fitted parabola) forms no patterns anywhere, as predicted.
- So the simulation follows the paper's equations and criterion, but **not the wider range of
  patterns shown in Fig. 3c**, nor N2's experimental patterns at about 20 worms/mm² (Section 9).

**Bacteria ON (extension, since removed).** Food was eaten, and as it ran low, worms everywhere
moved toward the fast off-food speed, so the patterns gradually **dissolved** over a few simulated
hours. **No swarm formed.** Three variants were tried:

1. Faster bacterial dynamics with the same λ: the patterns dissolved sooner.
2. A circular lawn patch: under the paper's parabola, npr-1 at 21 % is barely slower on food
   (≈ 150 µm/s) than off it (175 µm/s), so worms didn't collect on the lawn. A constant D_b also
   smeared the lawn across the box.
3. Worm-mediated bacterial mixing: no swarm either.

In every variant, food depletion sped worms up before any cluster grew larger than λ, which is when
a front–back gradient could form. **Decision:** remove the bacteria part and reproduce only what the
paper has equations for.

**Performance** (Apple M5, Metal): about 110 µs per step at 16k worms, so 10 simulated minutes take
about 1.3 s and 4 hours take about 30 s.

## 12. Out of scope

- Swarming and any bacteria field (Section 5).
- Biofilm and PA14 lawns, and V(O) on PA14 (Fig. 6).
- Lawn edges, and worms arriving from outside the lawn.
- Very thick lawns (npr-1 avoidance).
- Oxygen-dependent reversal rates.
- Phase-diagram tooling, notebooks and graphs.

## 13. Open questions

- **N2:** keep the fitted parabola (A4: paper form, paper data, our coefficients), or drop N2 and
  keep only the paper's own curve?
- **Fig. 3c gap:** re-check the paper's units and signs (A1, A2), and its supplementary material,
  for parameters that give patterns below 17 %.
