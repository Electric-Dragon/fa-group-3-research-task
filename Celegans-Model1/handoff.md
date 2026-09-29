# Handoff — C. elegans Model 1: environment-controlled pattern explorer

**For:** Claude Code (implementation) · **Written:** 29 Sep 2026 · **Base:** `../Celegans-Model0-v0.1.0`

## 0. TL;DR

Build **Model 1** on top of the existing Model 0 engine. Agent parameters are
**frozen per "strain" preset**; the user controls only the **environment**
(ambient oxygen as scalar / spatial map / time schedule, replenishment map,
agent count added or removed at runtime). The package must (a) predict where
patterns should appear from linear stability, (b) simulate, (c) automatically
classify the resulting pattern (uniform / dots / stripes / holes), and
(d) produce density × ambient-O₂ phase diagrams with the predicted boundary
overlaid. Inverse control (steering an aggregate with a moving low-O₂ spot) is
the final, optional milestone.

Do **not** modify `Celegans-Model0-v0.1.0/`. It is a released teaching package
and serves as the regression baseline.

---

## 1. Scientific context (read before coding)

Source: Demir et al. 2020, eLife 9:e52781 ("Dynamics of pattern formation and
emergence of swarming in C. elegans"). Supporting: Gray et al. 2004, Nature 430:317
(O₂ preference 5–12 %, npr-1 vs N2).

The paper's continuum model:

```
∂W/∂t = ∇·[D_W ∇W] + ∇·[β W ∇O]                  (worm density)
∂O/∂t = D_O ∇²O + f (O_am − O) − k_c W            (oxygen)
D_W = V(O)² / (2τ),   β = V(O) V'(O) / (2τ)
```

Key facts that drive the design:

1. **The PDE is the mean-field limit of an agent rule** (paper, Methods p.12:
   "random walk with oxygen-dependent speed V(O) … change direction at a rate τ").
   So agents only need: sense local O → set speed V(O) → move → randomise heading.
   **No gradient sensing.** Aerotaxis (β) emerges from speed modulation.
2. **The paper's τ is a rate** (units 1/time), despite the symbol. In Model 0 it
   corresponds to `rotational_diffusion` D_r (ABP in 2D: D_W = V²/(2 D_r)).
3. Pattern formation requires O_eq to sit on the **rising** branch of V(O).
   Pattern type follows density: low → dots, medium → stripes, high → holes
   (paper Figs. 2d, 3c). Ambient O₂ and density partly substitute for each other.
4. Banding / concentration / swarming (paper Fig. 4d–f) need a bacteria field.
   **Out of scope for Model 1.**

## 2. What already exists (Model 0 v0.1.0)

Path: `../Celegans-Model0-v0.1.0/celegans/model0/`

| File | Contents |
|---|---|
| `model.py` | frozen `Parameters` dataclass; `speed()` (sigmoid, monotone); periodic CIC deposition + Gaussian kernel `density_grid()`; bilinear `sample_field()`; `_oxygen_step()` (explicit 5-pt diffusion, exact replenishment, positivity-preserving implicit saturating consumption `q·ρ·O/(K+O)`); `Simulation` with adaptive substeps, `step`, `reset`, `set_params`, `state`, `result`; `run()` |
| `scenarios.py` | `uniform`, `landscape` (fixed fields), `feedback`, `seeded` (dynamic); `uniform_oxygen()` analytic equilibrium |
| `analysis.py` | `diagnostics()` → heterogeneity H, mean/min O, mean speed, polar order; `x_profile()` |
| `demo.py` | ipywidgets demo, cooperative asyncio loop, sliders |
| `visualization.py` | PNG frames, plots |
| `tests/test_model0.py` | 9 invariant tests (mass conservation, diffusion Fourier damping, replenishment analytic, positivity, chunking/reset reproducibility, …) |

Conventions to keep: arrays indexed `[y, x]`, positions `(x, y)`; periodic
cell-centred grid; arbitrary model units with oxygen normalised so ambient
air = 1.0; immutable parameters via `dataclasses.replace`; RNG seeded per run;
drawing never consumes RNG; notebook starts paused.

Current limitations Model 1 must lift:
- `n_agents`, domain, `dt` are immutable inside a `Simulation` (`set_params` raises).
- `ambient` and `replenishment` are scalars only.
- No pattern classifier, no structure factor, no stability prediction, no sweeps.
- `np.add.at` in `density_grid` is slow for 10⁴–10⁵ agents.

Default Model 0 parameters (from `assets/reference_metrics.json`):
64×48 domain, 64×48 grid, 1800 agents, dt 0.1, v_min 0.06, v_max 2.0,
midpoint 0.55, width 0.045, D_r 0.25, D_O 1.0, k (replenishment) 0.06,
ambient 1.0, q 0.06, K 0.10, kernel σ 1.2.

**Sanity numbers I computed for these defaults** (use as a test oracle for §4.3):
O_eq ≈ 0.510, V ≈ 0.626, V' ≈ 8.91, D_W ≈ 0.784, β ≈ 11.16,
max linear growth rate ≈ 0.057 at k* ≈ 0.29 → **wavelength ≈ 21.6 model units**.
The 64×48 box holds only ~3×2 wavelengths — too small for reliable pattern
classification. Phase-diagram runs should use a larger box (see §6).

---

## 3. Target layout

```
Celegans-Model1/
├── handoff.md                    (this file)
├── README.md
├── VALIDATION.md
├── requirements.txt              (Model 0's + nothing new unless justified)
├── celegans/
│   ├── __init__.py               (exports model1 API; keep model0 importable)
│   ├── model0/                   (copied verbatim from Model 0, untouched — regression baseline)
│   └── model1/
│       ├── __init__.py
│       ├── strains.py            §4.1
│       ├── environment.py        §4.2
│       ├── engine.py             §4.4 (extends model0 numerics)
│       ├── stability.py          §4.3
│       ├── patterns.py           §4.5
│       ├── sweep.py              §4.6
│       ├── demo.py               §4.7
│       └── visualization.py
├── tests/
│   ├── test_model0.py            (copied; must still pass)
│   └── test_model1.py
├── scripts/
│   └── phase_diagram.py          (CLI batch sweep)
└── 01_environment_controlled_patterns.ipynb
```

Copy Model 0's `celegans/`, `tests/`, `requirements.txt` in as the starting
point, then add `model1/`. Reuse Model 0 functions by import where possible
(`sample_field`, `_oxygen_step` logic) rather than duplicating; refactor only
inside `model1/`.

---

## 4. Components

### 4.1 `strains.py` — frozen agent parameters

```python
@dataclass(frozen=True)
class Strain:
    name: str
    v_min: float
    v_max: float
    oxygen_midpoint: float     # normalised O (1.0 = ambient air ≈ 21 %)
    response_width: float
    rotational_diffusion: float   # = paper's "τ" (a rate)
    consumption: float            # q
    oxygen_half_saturation: float # K
    kernel_width: float
```

Presets (teaching presets, **not** fitted — say so in docstrings):
- `NPR1_LIKE`: steep response (Model 0 defaults are a reasonable start).
- `N2_LIKE`: much wider `response_width` and/or smaller `v_max − v_min`
  → weaker β → patterns only at high density.
- `CONSTANT_SPEED`: control (β = 0), must never pattern.

Tune the two biological presets **using `stability.py` first** so that on the
ambient range 0.05–1.0 and density range of the sweep, NPR1_LIKE has a broad
unstable window and N2_LIKE only a high-density one. Record the chosen numbers
and the reasoning in VALIDATION.md.

Optional (off by default): non-monotone V(O) with a hypoxic high-speed branch
at very low O (paper Fig. 2b). Implement as a separate speed-law option, not
by editing the sigmoid.

### 4.2 `environment.py` — what the user controls

```python
@dataclass(frozen=True)
class Environment:
    ambient: float | Callable = 1.0          # scalar, or (x_grid, y_grid, t) -> array
    replenishment: float | np.ndarray = 0.06 # scalar or [ny, nx] map (cover glass = low values)
    oxygen_diffusion: float = 1.0
```

Helpers:
- `uniform(level)`
- `linear_gradient(lo, hi, axis='x')` — note: on a periodic domain use a smooth
  periodic profile (cosine) or document the seam; prefer cosine.
- `spot(center, radius, level, background)` and `moving_spot(path(t), …)`
- `schedule([(t0, level0), (t1, level1), …])` — piecewise-constant steps
  (reproduces Gray et al. 2004 Fig. 4d: 21 % → 7 % → 21 %).
- Percentage helper: `pct(21) == 1.0`, `pct(7) ≈ 0.333`.

**Rule:** the environment acts only via the replenishment term
`f(x)·(O_am(x,t) − O)`. Never overwrite the O field directly — the agents'
feedback must remain.

Change `_oxygen_step` accordingly: exact replenishment
`c = A + (c − A)·exp(−f·h)` works element-wise with array `A`, `f`.
Evaluate `ambient(x, y, t)` once per substep (or once per outer turn — document
which; per outer turn is fine if dt ≤ schedule resolution).

### 4.3 `stability.py` — analytic prediction (build this early)

Linearise around the homogeneous state for the Model 0/1 equations
(ABP speed V(O), saturating consumption, Gaussian consumption kernel):

```
W  = N / area
O_eq solves  f (A − O) = q W O / (K + O)           (reuse model0 uniform_oxygen)
V, V' at O_eq;  D_W = V²/(2 D_r);  β = V V'/(2 D_r)
a = q O_eq/(K+O_eq)           # ∂consumption/∂W
b = q W K/(K+O_eq)²           # ∂consumption/∂O
g(k) = exp(−σ² k² / 2)        # kernel_width σ

M(k) = [[ −D_W k²,        −β W k²            ],
        [ −a g(k),        −(D_O k² + f + b)  ]]
λ(k) = largest real eigenvalue of M(k)
unstable  ⇔  max_k λ(k) > 0
         ⇔  ∃k: D_W (D_O k² + f + b) < β W a g(k)
```

API:
```python
growth_curve(strain, env, density, k) -> λ(k)
predict(strain, env, density) -> dict(unstable, k_star, wavelength, growth_rate, O_eq, D_W, beta)
predicted_map(strain, ambients, densities) -> 2D bool/float arrays
```

Tests: reproduce the oracle numbers in §2 (O_eq 0.510, wavelength ≈ 21.6,
λ_max ≈ 0.057, tolerance ~2 %); CONSTANT_SPEED → never unstable; ambient on
the flat upper plateau of V → stable.

Caveat to document: the drift-diffusion limit needs pattern wavelength ≫
persistence length V/D_r (≈ 2.5 at defaults vs 21.6 wavelength — OK, but check
per preset and flag cases where the ratio < ~5). Finite-N noise shifts the
observed boundary; expect agreement away from the boundary, fuzz near it.

### 4.4 `engine.py` — Simulation with runtime environment and population

Extend Model 0's `Simulation` (subclass or new class reusing its pieces):

- Constructor takes `strain`, `env`, `domain=(width, height, nx, ny)`,
  `n_agents`, `dt`, `seed`, `init='uniform'|'seeded'|'droplet'`.
- `set_environment(env)` — immediate, logged with time in `parameter_log`.
- `add_agents(n, mode='uniform'|'point', center=None, spread=None)`
  and `remove_agents(n)` (uniformly random) — use the engine RNG so runs stay
  reproducible; log each change.
- Performance: replace `np.add.at` with `np.bincount(flat_index, weights, minlength)`
  for CIC deposition; Gaussian via `scipy.ndimage.gaussian_filter(mode='wrap')`
  or FFT multiply (precompute kernel). Target: **20 000 agents on 192×192 grid
  ≥ 50 outer turns/s** on a laptop. Profile before optimising further. Consider
  Numba only if needed; do not add JAX.
- Keep Model 0's adaptive substep logic and positivity guarantees. Keep
  "drawing never consumes RNG".
- Recorded history: Model 0 diagnostics + pattern metrics from §4.5 every
  `record_every` turns.

### 4.5 `patterns.py` — automatic classification

On the density field smoothed at a fixed diagnostic length (Model 0 uses 2.0):

1. **Heterogeneity H** (existing).
2. **Structure factor** S(k) of δρ, radially averaged → peak k_peak,
   characteristic length L = 2π/k_peak. Compare with `stability.predict`.
3. **Dense-phase area fraction** φ = fraction of cells with ρ > threshold
   (threshold: Otsu on ρ, or mean ρ; pick one and document).
4. **Topology on the torus:** components of dense phase (n_c) and of dilute
   phase (n_h) using `scipy.ndimage.label` with periodic wrap handled
   (label, then union labels touching across opposite edges).
   Euler-type descriptor χ = n_c − n_h.
5. **Label rule** (initial; calibrate on runs and record thresholds):
   - `uniform` if H < H_min
   - `dots` if dense phase = many small components, φ ≲ 0.35
   - `holes` if dilute phase = many small components, φ ≳ 0.65
   - `stripes` / `labyrinth` otherwise (components percolate)
6. **Coarsening:** L(t) history; fit exponent on log-log over late times.

Return a dict; include `label`, all raw numbers, and the threshold settings used.

### 4.6 `sweep.py` + `scripts/phase_diagram.py`

- Grid over `ambient × density × seeds` for a given strain. Default 8 × 8 × 3.
- `multiprocessing` / `concurrent.futures` (process pool), one run per task,
  deterministic seeds `seed = hash((i_ambient, i_density, rep))`-style but
  stable (not Python `hash`).
- Density sweep = change `n_agents` at fixed domain.
- Run length: fixed physical time (≥ ~10 / λ_max for the slowest unstable case,
  capped); record final metrics + label.
- Output: `results/<strain>_<timestamp>/runs.csv` (+ `config.json`,
  optional `final_fields.npz`).
- Plot: phase diagram coloured by majority label, marker alpha = seed agreement,
  **predicted instability boundary from `stability.predicted_map` overlaid**.
- Resume support: skip rows already in CSV.

### 4.7 `demo.py` — interactive

Model 0's demo pattern, but sliders only for environment + population:
- ambient (%) slider; replenishment slider; "Add N" / "Remove N" buttons with
  mode dropdown; strain preset dropdown (Reset required to change strain).
- Environment presets: uniform, cosine gradient, spot, O₂ step schedule.
- Live panels: density, oxygen, S(k) with predicted k*, current label,
  and a small (ambient, density) phase map with the current point marked.
- Starts paused; only one active demo (keep Model 0 behaviour).

### 4.8 Notebook `01_environment_controlled_patterns.ipynb`

Sections: (1) scope & what is fixed vs controlled; (2) strain presets and their
V(O) curves; (3) predicted map; (4) live demo; (5) phase diagram for NPR1_LIKE
and N2_LIKE (load precomputed results in `assets/`, don't run the full sweep in
the notebook); (6) O₂ step experiment (dissolve/re-form); (7) hysteresis ramp;
(8) limitations. Clean outputs; Run All must finish in < ~2 min.

---

## 5. Experiments to support (in order)

1. **Predicted vs simulated phase diagram** (NPR1_LIKE, N2_LIKE, CONSTANT_SPEED).
2. **O₂ step response:** 1.0 → 0.33 → 1.0; measure time to dissolve / re-form.
3. **Hysteresis:** slow ramp of ambient up then down at fixed N; plot H and
   label vs ambient for both directions.
4. **Finite-N:** fixed density, scale domain + N together (N = 500 … 50 000);
   probability of patterning near the boundary vs N.
5. **Spatial gradient:** cosine O_am(x); density profile vs x vs predicted
   1/V(O) accumulation; pattern label vs local ambient.
6. *(Optional milestone M6)* **Inverse control:** moving low-O₂ spot steers one
   aggregate along a path; parameterise spot (speed, radius, depth), optimise
   with CMA-ES. Reference implementation style:
   `../../SwarmRules-v2.0.2/cmaes_search.py` and `search_common.py` (read-only;
   do not import across projects — copy patterns). Objective averaged over ≥ 2 seeds.

---

## 6. Numerical guidance

- Phase-diagram box: at least ~6–8 wavelengths per side. With default-like
  presets (λ ≈ 22), use ~160–192 units square. Keep grid spacing ≈ 1 unit
  (≤ kernel σ); keep density in agents/area comparable to Model 0 (≈ 0.59)
  → ~15 000–22 000 agents.
- Keep all Model 0 stability bounds; the explicit diffusion bound is not
  limiting at dx = 1, D_O = 1.
- Convergence check for VALIDATION.md: one unstable point, halve dt and halve dx
  (preserving physical size and density); labels and L must agree across 3 seeds.

## 7. Acceptance criteria

- [ ] `python -m unittest discover -s tests -v` passes, **including all 9 Model 0 tests unchanged**.
- [ ] New tests: stability oracle (§2 numbers); constant-speed never unstable;
      array ambient/replenishment reduce to scalar case bit-for-bit when uniform;
      add/remove agents keep density integral = N and are seed-reproducible;
      environment schedule switches at the right turn; periodic labelling
      counts a single stripe wrapping the torus as 1 component; classifier
      labels synthetic dot/stripe/hole images correctly.
- [ ] Performance target in §4.4 met (report actual numbers).
- [ ] Phase diagram for NPR1_LIKE: simulated labels agree with the predicted
      stable/unstable region for ≥ 90 % of grid points not adjacent to the
      predicted boundary. Report the number.
- [ ] Constant-speed control: `uniform` everywhere.
- [ ] Qualitative ordering dots → stripes → holes with increasing density
      appears for at least one ambient level. If not, report it honestly — do
      not tune the classifier to force it.
- [ ] VALIDATION.md written in Model 0's style: what was checked, numbers,
      and explicit limits of the claims.

## 8. Out of scope

Bacteria field, food depletion, banding, swarming translation, body mechanics,
excluded volume, alignment, reversals, gradient steering, fitting to real
worm data, changes to Model 0 or SwarmRules.

## 9. Pitfalls

- τ in the paper is a **rate**; do not convert it as a time.
- Do not clip O to hide instability — Model 0's implicit consumption already
  keeps O ≥ 0; keep that.
- Don't write to the O field to "set" the environment; go through O_am and f.
- Periodic domains + linear gradients create a seam; use cosine profiles.
- Area fraction / Euler thresholds are heuristics — document them and keep
  raw metrics in outputs so labels can be recomputed.
- High H alone ≠ phase separation (Model 0 README says this; keep the wording).
- Different preset → different wavelength → the fixed box may be too small;
  have `sweep.py` warn when domain < 6 × predicted wavelength.

## 10. Suggested milestones

| # | Deliverable | Done when |
|---|---|---|
| M1 | Project copy, `strains.py`, `environment.py`, array ambient/f in engine, runtime add/remove | Model 0 tests + env/population tests pass |
| M2 | `stability.py` + predicted maps; presets tuned | oracle test passes; preset maps plotted |
| M3 | Performance (bincount/FFT) | target met |
| M4 | `patterns.py` classifier | synthetic-image tests pass |
| M5 | `sweep.py`, phase diagrams, notebook, VALIDATION.md | acceptance criteria met |
| M6 | (optional) inverse control with CMA-ES | aggregate follows path in ≥ 2/3 seeds |

## 11. References

- Demir, Yaman, Basaran, Kocabas (2020). eLife 9:e52781. doi:10.7554/eLife.52781
- Gray et al. (2004). Oxygen sensation and social feeding mediated by a C. elegans guanylate cyclase homologue. Nature 430:317–322.
- Cates & Tailleur (2015). Motility-induced phase separation. Annu. Rev. CMP 6:219. arXiv:1406.3533
- Tailleur & Cates (2008). PRL 100:218103 (run-and-tumble with density/position-dependent speed).
- Liebchen & Löwen (2018). Synthetic chemotaxis and collective behavior in active matter. Acc. Chem. Res. 51:2982.
- Arlt et al. (2018). Painting with light-powered bacteria. Nat. Commun. 9:768 (speed-modulation density shaping — prior art for M6).
- Frangipane et al. (2018). Dynamic density shaping of photokinetic E. coli. eLife 7:e36608.
