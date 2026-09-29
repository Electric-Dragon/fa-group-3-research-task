# Model 1 validation — v1.0.0

Checked on 29 September 2026. These checks support the teaching demonstrations
and the claims made in the notebook. They do not establish biological
calibration, a thermodynamic phase diagram, or a measured convergence rate.

Numerical environment: Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1,
Matplotlib 3.11.2. Notebook and widget checks used IPython 9.17.1,
ipywidgets 8.1.9 and nbformat 5.11.1. Exact trajectories are not guaranteed
across dependency versions or platforms.

---

## 1. Model 0 is unchanged and still passes

`celegans/model0/` is a byte-identical copy of Model 0 v0.1.0, and
`tests/test_model0.py` is its test file unchanged. All nine Model 0 checks pass
inside this package:

```
python -m unittest discover -s tests -p 'test_model0.py' -v    ->  9 tests, OK
python -m unittest discover -s tests -v                        -> 57 tests, OK
```

Model 0's names are still re-exported at the top level (`ce.Parameters`,
`ce.run`, …), which is what lets its test file run untouched.

### Model 1's engine reproduces Model 0's trajectories

Model 1 replaces `np.add.at` with `np.bincount` in the density deposition and
computes the cloud-in-cell weights once per substep instead of twice. Both are
arithmetic reorganisations, not a change of method, and the claim is tested:
with the `MODEL0_DEFAULT` strain and a uniform environment on a 32 × 24 domain,
400 agents, 200 turns, seed 3:

| Quantity | Largest difference from Model 0 |
|---|---:|
| agent positions | 1.1e-14 |
| oxygen field | 3.3e-16 |
| final heterogeneity H | identical to 6 decimal places |

The residual is float summation order in the deposition, which `np.bincount`
and `np.add.at` do differently. A separate test asserts the two `density_grid`
implementations agree to 1e-12 on the same positions.

---

## 2. Linear stability against the documented oracle

`stability.py` was built before the presets were tuned and was checked against
the independently computed numbers in the handoff, for Model 0's defaults
(density 1800 / (64 × 48) = 0.586, ambient 1.0, f = 0.06):

| Quantity | Handoff oracle | Computed | Difference |
|---|---:|---:|---:|
| O_eq | 0.510 | 0.5101 | 0.02% |
| V(O_eq) | 0.626 | 0.6261 | 0.02% |
| V'(O_eq) | 8.91 | 8.909 | 0.01% |
| D_W | 0.784 | 0.7840 | 0.00% |
| beta | 11.16 | 11.156 | 0.04% |
| max growth rate | 0.057 | 0.05722 | 0.4% |
| k* | 0.29 | 0.2917 | 0.6% |
| wavelength | 21.6 | 21.54 | 0.3% |

The test asserts all of these to within 2%. Two further checks pin the
structure of the dispersion relation: lambda(0) = 0 to 1e-11 (agent number is
conserved, so a uniform shift is not a growing mode), and lambda(k) < 0 at
k = 5 and k = 20, where the Gaussian consumption footprint has killed the
coupling.

`equilibrium_oxygen` is checked to agree with Model 0's `uniform_oxygen` to 14
decimal places at three different populations, and its vectorised form is
checked element by element against its own scalar form.

### Controls

- `CONSTANT_SPEED` is unstable at no point of a 12 × 12 grid spanning ambient
  0.02–1.0 and density 0.01–20: beta is exactly 0.0 and the maximum growth rate
  is never positive. `critical_density` returns NaN for it.
- A strain whose midpoint sits far below the equilibrium oxygen — the "ambient
  is so high the response has saturated" case — has beta < 1e-3 and is stable.

---

## 3. Preset tuning

The two biological presets were tuned with `stability.py` only, before any
simulation was run, against the requirement in the handoff: over ambient
0.3–1.0 and the density range of the sweep, `NPR1_LIKE` should have a broad
unstable window and `N2_LIKE` only a high-density one.

**Lower edge of the unstable density window, by ambient level:**

| Ambient | NPR1_LIKE | N2_LIKE |
|---:|---:|---:|
| 0.5 | 0.18 | 0.33 |
| 0.7 | 0.21 | 0.41 |
| 0.9 | 0.34 | 0.57 |
| 1.0 | 0.42 | 0.66 |

`N2_LIKE` needs 1.5–1.9 times the density for the same ambient level, at every
level. A test asserts this ordering at four ambient levels.

### What was changed and why

**`NPR1_LIKE` widened `response_width` from Model 0's 0.045 to 0.09.** At 0.045
the response is so steep that V'(O) is appreciable only in a narrow oxygen
window, so the unstable region is a thin diagonal sliver in the
(ambient, density) plane: on a 8 × 8 grid over ambient 0.05–1.0 and density
0.15–1.2 only 17% of cells were unstable and **no single ambient level held
more than three unstable densities**, which is not enough to sweep density
across the band and look for a morphology sequence. At 0.09 the same grid gives
31% unstable and up to six unstable densities at one ambient. The response is
still steep — it rises over roughly 0.36 in normalised oxygen, about 7% O2.
Model 0's exact defaults are preserved as the separate `MODEL0_DEFAULT` strain,
which is what the oracle above refers to and what the regression test uses.

**`N2_LIKE` has both a smaller speed span (1.45 against 1.94) and a lower
midpoint (0.35 against 0.55).** The smaller span lowers beta directly. The
lower midpoint means the agents only reach the responsive part of their speed
law once consumption has pulled the local oxygen much further down, which
pushes the onset to higher density.

The midpoint shift is a modelling choice and it was not the first thing tried.
Weakening beta alone — a much wider `response_width` at the same midpoint —
also raises the onset density, and a search over v_min, v_max and
`response_width` at midpoint 0.55 found several candidates that do. They were
rejected because all of them sit so close to marginal that the **predicted
wavelength exceeds 45 model units everywhere they pattern** (42–302 across the
grid for the best of them). Classifying that honestly needs a domain of roughly
280 units a side and about 80,000 agents, which would have made the phase
diagram for `N2_LIKE` an order of magnitude more expensive than the one for
`NPR1_LIKE` and not comparable to it. The present `N2_LIKE` keeps the predicted
wavelength at 31–36 where it is solidly unstable, so one box serves both.

At Model 0's default operating point (density 0.586, ambient air) the three
presets sit as follows — same O_eq, different consequences:

| Strain | O_eq | beta | prediction |
|---|---:|---:|---|
| MODEL0_DEFAULT | 0.510 | 11.14 | unstable, wavelength 21.5, growth 0.0572 |
| NPR1_LIKE | 0.510 | 8.40 | unstable, wavelength 33.5, growth 0.0205 |
| N2_LIKE | 0.510 | 5.50 | stable |
| CONSTANT_SPEED | 0.510 | 0.00 | stable |

### Validity of the diffusive limit

The drift-diffusion description behind `stability.py` needs the pattern
wavelength to be much larger than the persistence length V / D_r. At Model 0's
defaults that ratio is 21.5 / 2.50 = 8.6. `predict()` returns it as
`wavelength_over_persistence` and sets `diffusive_limit_ok=False` below 5;
`sweep.plan()` prints a warning for every cell of a sweep where it is marginal,
before the sweep runs.

---

## 4. Environments

- **A constant array is bit-for-bit the scalar path.** `oxygen_step` with
  ambient and replenishment given as full arrays of a constant produces an
  oxygen field identical in every bit to the same step with scalars
  (`assert_array_equal`, not a tolerance). A 40-turn run built from an array
  environment matches the scalar-environment run exactly in both positions and
  oxygen. This is the guarantee that makes array environments safe: they are
  the same code path, not a parallel one.
- **A schedule switches on the right turn.** With dt = 0.1 and steps at
  t = 2.0 and t = 4.0, the ambient used during turn 19 is 1.0, during turn 20 is
  1/3, during turn 39 is 1/3 and during turn 40 is 1.0. Ambient is resolved once
  per outer turn at the time the turn begins, so turn index i covers
  [i·dt, (i+1)·dt) and is evaluated at i·dt.
- **The cosine gradient has no seam.** On a periodic domain a linear ramp would
  leave a jump; the profile used runs lo → hi → lo and the difference between
  the first and last column is under a quarter of the profile's own range,
  while both stated extremes are reached to within 0.02.
- **The environment never writes to the oxygen field.** In a run with an imposed
  low-oxygen spot at level 0.5, the oxygen at the spot centre settles strictly
  below 0.5 and strictly above 0 — the agents' consumption is still competing
  with the imposed level, which is the whole point of acting through
  `f · (O_am − O)` rather than by assignment.
- Invalid environments (ambient outside [0, 1], negative replenishment or
  diffusivity, non-finite values, an empty schedule, an unknown gradient axis)
  are rejected at construction.

**A low-oxygen spot does accumulate agents, and it does so slowly.** In a
96 x 96 domain with 6,000 agents and an imposed spot of radius 22 at ambient
0.15, the radial density profile relative to the domain mean develops as:

| Radius | t = 400 | t = 2000 | t = 6000 | local O | local V |
|---|---:|---:|---:|---:|---:|
| 0-12 | 1.01 | 1.03 | 1.37 | 0.02 | 0.066 |
| 12-24 | 1.28 | 1.70 | 2.52 | 0.02-0.04 | 0.066 |
| 24-30 | 1.96 | 2.74 | 2.83 | 0.17 | 0.088 |
| 30-40 (outside) | 0.71 | 1.02 | 0.68 | 0.65 | 1.51 |

The accumulation appears first as a **ring at the spot edge** and fills inwards
over thousands of time units. That is not an artefact: inside the spot V is
0.066 against 1.5 outside, so D_W = V^2 / 2 D_r is 0.0087 against 4.5, and the
time to diffuse across the 22-unit disc at the interior mobility is of order
22^2 / (4 D_W) ~ 14,000. The ratio the stationary rho ~ 1/V argument predicts
(about 20) is therefore nowhere near reached at t = 6000, and should not be.
Anyone reading the demo should expect a ring long before a filled disc.

---

## 5. Population changes at runtime

- After adding 250 agents uniformly, adding 120 in a point blob, and removing
  300, the integral of the density field over the domain equals the agent count
  to 9 decimal places at each step, and the position and heading arrays stay the
  right length.
- A scripted history — step 10, add 150 at a point, step 10, remove 200,
  step 10 — reproduces identical positions and oxygen when replayed from the
  same seed. Everything stochastic comes from the one engine generator.
- Every change is logged with its turn, model time and resulting population;
  the test checks the exact `(event, turn, n_agents)` triples.
- `remove_agents` leaves at least one agent; negative counts are rejected.

---

## 6. Pattern classifier

**Periodic topology.** A stripe spanning the domain is one component, not two,
in both orientations. A blob split across the seam is one component, and four
quarter-blobs in the four corners are one component. Two genuinely separate
blobs stay two.

**Synthetic images.** Idealised dot, stripe and hole fields are labelled
correctly, with both the Otsu and the mean threshold. Their area fractions come
out in the right order (dots < stripes < holes) and the Euler-type descriptor
chi = n_dense − n_dilute is positive for dots and negative for holes.

**Noise is not a pattern.** A real Poisson cloud of 5400 agents on a 96 × 96
domain, put through the actual deposition and smoothing, is labelled `uniform`,
and its measured heterogeneity is within 25% of the analytic shot-noise level
1 / (4 pi sigma^2 W). That formula gives 0.034 at Model 0's defaults, which is
where Model 0 measured 0.026–0.036 for its initial conditions — so the
classifier's "is this more than noise" test is calibrated against a quantity
Model 0 already reported.

**Structure factor.** An imposed stripe wavelength of 16, 24 or 32 model units
is recovered to within 10%. The residual is the radial bin width, which is one
fundamental wavenumber; bins are reported at the mean |k| of the modes they
hold rather than at the bin centre, which removed a systematic several-percent
bias.

### The threshold choice, and what it changes

The dense-phase area fraction needs a threshold, and the two candidates were the
mean density and Otsu's. **The default is Otsu's, and that choice was made after
seeing simulation output**, so it is set out here in full.

The mean is a systematically biased separator. In a phase-separated system the
two coexisting densities sit either side of the mean at a spacing fixed by the
lever rule, so thresholding at the mean drives phi towards 0.5 whenever the
density contrast is symmetric and misplaces the boundary whenever it is not. In
this model the contrast is strongly asymmetric at high density: voids are deeply
depleted while the dense phase is only modestly above average, so the mean falls
*inside* the dense phase.

Measured, on `NPR1_LIKE` runs at ambient 0.9–1.0 and density 1.0–1.28
(144 × 144 box, t = 5000–8000), with the two thresholds applied to the *same*
density fields:

| Ambient, density | phi (mean) | label (mean) | phi (Otsu) | label (Otsu) | n_dense | n_dilute |
|---|---:|---|---:|---|---:|---:|
| 1.0, 1.28 | 0.63 | stripes | 0.80 | holes | 2 | 40 |
| 1.0, 1.20 | 0.60 | stripes | 0.71 | holes | 1 | 30 |
| 0.9, 1.10 | 0.59 | stripes | 0.78 | holes | 4 | 28 |
| 0.9, 1.00 | 0.59 | stripes | 0.69 | holes | 1 | 24 |

The topology is not in doubt in any of these: one connected dense phase riddled
with 24–40 isolated voids, none of which percolate, is a hole morphology by any
reading. The mean threshold simply fails to register it. Both labels and both
area fractions are written to every sweep CSV (`label` / `area_fraction` from
Otsu, `label_mean` / `area_fraction_mean` from the mean), so the choice can be
audited from the data without rerunning anything, and the label rule itself —
the 0.35 and 0.65 area fractions, the 3x shot-noise heterogeneity gate — was
**not** adjusted.

---

## 7. Performance

Target from the handoff: 20,000 agents on a 192 × 192 grid at **at least 50
outer turns per second** on a laptop. Measured on an Apple M-series laptop,
single process, after a 20-turn warm-up, averaged over 200 turns, with history
recording off:

| Engine | Turns / second | Notes |
|---|---:|---|
| Model 1 | **283** | target exceeded by 5.7x |
| Model 0, same configuration | 215 | for reference only |

Deposition microbenchmark at the same size (20,000 agents, 192 × 192,
20 repetitions):

| Implementation | Time per call |
|---|---:|
| Model 0, `np.add.at` | 1.99 ms |
| Model 1, `np.bincount` | 1.34 ms |

**The `np.add.at` bottleneck the handoff anticipated is largely gone on NumPy
2.5.** It is 1.5x slower than `np.bincount` here, not the order of magnitude it
used to be, so the deposition change contributes less to the 1.3x overall
speed-up than reusing the cloud-in-cell weights between deposition and field
sampling does. A profile of the stepped loop puts the remaining time at roughly
a quarter in `_weights`, a quarter in deposition and its Gaussian filter, and a
quarter in the oxygen step. Numba was not needed and was not added; no new
dependency was introduced for any part of Model 1.

At the phase-diagram configuration (176 × 176 box, 9,000–37,000 agents,
12,000 turns) a single run takes 36–120 seconds, and the full 192-run sweep
completes in about 30 minutes on nine processes.

---

## 8. Phase diagrams: prediction against simulation

Configuration, identical for all three strains: a 176 x 176 box on a 176 x 176
grid (dx = 1), dt = 0.1, run time 1200 model time units (12,000 turns), density
swept 0.30-1.20 in eight steps by changing the agent count at fixed domain
(9,293 to 37,171 agents), ambient swept 0.30-1.00 in eight steps, three seeds
per cell (two for the control). Seeds come from a CRC of the cell coordinates,
so the sweep is repeatable. Raw rows are in `assets/sweep_<strain>.csv`.

### NPR1_LIKE

**Agreement with the linear-stability prediction, excluding cells adjacent to
the predicted boundary: 24 / 24 = 100%.** The acceptance criterion was 90%.

That number needs its denominator stated plainly. The unstable region is a
*diagonal* band in the (ambient, density) plane, so on an 8 x 8 grid a one-cell
margin around the predicted boundary removes **40 of the 64 cells**, and the
100% is over the 24 that remain. Across all 64 cells, including every boundary
cell, agreement is 54 / 64 = 84.4%.

What makes the excluded cells worth excluding is visible in the disagreements
themselves. All ten of them are boundary-adjacent, and nine are of the same
kind: predicted unstable, observed uniform, with a predicted growth rate so
small that the run never had time to show anything.

| Ambient | Density | Predicted | Observed | growth x run time | Predicted wavelength |
|---:|---:|---|---|---:|---:|
| 0.50 | 0.30 | unstable | uniform | 0.0 | 84 |
| 0.60 | 0.30 | unstable | uniform | 0.8 | 49 |
| 0.60 | 0.43 | unstable | uniform | 0.3 | 46 |
| 0.60 | 0.56 | unstable | uniform | 0.0 | 72 |
| 0.70 | 0.69 | unstable | uniform | 0.1 | 47 |
| 0.80 | 0.81 | unstable | uniform | 0.3 | 38 |
| 0.80 | 0.94 | unstable | uniform | 0.0 | 75 |
| 0.90 | 1.07 | unstable | uniform | 0.1 | 49 |
| 1.00 | 1.20 | unstable | uniform | 0.1 | 39 |
| 0.90 | 0.30 | **stable** | **dots** | -0.0 | - |

`growth x run time` is the number of e-foldings the linear prediction allows in
the whole run. Nine of the ten cells had **less than one**, and six had
essentially none. A linearly unstable state that is given less than one
e-folding is not expected to look different from a uniform one, so these are
not the prediction being wrong; they are the run being shorter than the physics
the prediction describes. Several also carry a predicted wavelength of 46-84
model units, which is 2-4 per box side — `sweep.plan()` flagged 21 cells on
exactly that ground before the sweep ran.

Read off the figure, the same fact is one sentence: **every patterned cell but
one falls inside the shaded predicted band**, and the cells that disagree are
uniform ones lying inside the band near its upper edge, where the growth rate
is going to zero.

The tenth disagreement is the interesting one and runs the other way: at ambient 0.90 and
density 0.30 the state is predicted (marginally) stable and the simulation
produced dots. That is the finite-N noise the prediction cannot describe,
appearing where it was expected to — just below a marginal boundary.

**Seed agreement: 0.99 mean, unanimous in 63 of 64 cells.** One cell split
2 : 1. The phase-diagram markers are drawn with opacity proportional to this,
so disputed cells are visibly washed out.

**Measured against predicted length scale.** Over the 18 patterned cells with a
finite predicted wavelength, the ratio of the measured structure-factor length
to the predicted wavelength has median 1.16 and range 0.45-1.55. Patterns are
systematically a little coarser than linear theory predicts, which is what
coarsening does: the prediction is for the fastest-growing mode at t = 0, and
the measurement is at t = 1200 after that mode has merged with its neighbours.

### The morphology sequence

The acceptance criterion asked for dots -> stripes -> holes with increasing
density at at least one ambient level. **It appears, at ambient 1.00:**

| Density | 0.30 | 0.43 | 0.56 | 0.69 | 0.81 | 0.94 | 1.07 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| ambient 1.00 | uniform | dots | dots | dots | stripes | stripes | **holes** | uniform |
| ambient 0.90 | dots | dots | dots | stripes | stripes | stripes | uniform | uniform |
| ambient 0.80 | dots | dots | stripes | stripes | uniform | uniform | uniform | uniform |
| ambient 0.70 | dots | stripes | stripes | uniform | uniform | uniform | uniform | uniform |

Two honest qualifications.

**Holes are rare here: one cell out of 64, three runs out of 192.** The reason
is structural, not a classifier artefact. Raising the density lowers the
equilibrium oxygen, which is what drives the dense phase to occupy more area —
but it also carries the agents off the rising branch of V(O), which kills beta.
In this model the second effect wins before the area fraction gets far past
0.7, so the band closes from above rather than continuing into a broad hole
regime. At ambient 1.00 the hole cell sits immediately below the upper edge of
the band, and the cell above it is uniform because it is past that edge.

**With the mean threshold instead of Otsu's, the hole cell reads `stripes` and
no cell in the entire sweep is labelled `holes`** (192 runs: 134 uniform,
27 dots, 31 stripes). This is the same effect measured in section 6, now on the
sweep itself, and it is why the threshold choice is documented rather than
buried. Both columns are in the CSV.

The sequence also runs the other way, and that is the paper's point: at fixed
density, *lowering* the ambient level moves a cell along the same sequence as
*raising* the density. The table above reads as a diagonal for that reason.

### N2_LIKE

**Agreement excluding boundary-adjacent cells: 19 / 19 = 100%** (45 cells
excluded; across all 64 cells, 93.8%). Seed agreement 0.98, unanimous in 62 of
64 cells. Measured-to-predicted length-scale ratio: median 1.18 over 20
patterned cells, matching NPR1_LIKE's 1.16.

The point of this strain is the contrast with NPR1_LIKE, and it survives
simulation. The lowest density at which each strain actually patterned:

| Ambient | NPR1 observed | NPR1 predicted | N2 observed | N2 predicted |
|---:|---:|---:|---:|---:|
| 0.70 | 0.30 | 0.21 | 0.43 | 0.41 |
| 0.80 | 0.30 | 0.27 | 0.56 | 0.48 |
| 0.90 | 0.30 | 0.34 | 0.56 | 0.57 |
| 1.00 | 0.43 | 0.42 | 0.69 | 0.66 |

Observed onsets sit within one grid step of the predicted ones (the density
grid spacing is 0.128), and **N2_LIKE needs more density than NPR1_LIKE at
every ambient level**, by 1.4 to 1.9 times. That ordering was imposed on the
presets through linear theory alone, before any simulation was run; the sweep
is an independent confirmation, not a restatement.

N2_LIKE produced no `holes` at all. Its unstable band is narrower, and it never
reaches the area fraction where a hole morphology appears before the band
closes from above.

### CONSTANT_SPEED (the control)

**`uniform` in all 64 cells, 128 of 128 runs, at every ambient level and every
density.** Agreement 64 / 64 = 100%, with no cells excluded (there is no
boundary to be adjacent to: beta is exactly zero, so nothing is predicted
unstable anywhere). Seed agreement 1.00 in every cell.

The strongest heterogeneity anywhere in the control sweep is **1.05 times the
Poisson shot-noise level** — that is, the densest-looking run is statistically
indistinguishable from a random cloud of the same density. This is the
acceptance criterion for the control, and it is the check that the patterning
seen in the other two strains is the oxygen-motility feedback and not an
artefact of deposition, smoothing, the classifier, or the periodic box.

Note what the control does *not* remove: these agents still consume oxygen, and
the oxygen field is still spatially structured. Only the feedback onto motility
is cut. So the control isolates beta specifically, not "oxygen dynamics".

### Summary of the three sweeps

| Strain | Runs | Labels found | Agreement (interior) | Agreement (all 64) | Seed agreement |
|---|---:|---|---:|---:|---:|
| NPR1_LIKE | 192 | uniform, dots, stripes, holes | 24/24 = 100% | 84.4% | 0.99 |
| N2_LIKE | 192 | uniform, dots, stripes | 19/19 = 100% | 93.8% | 0.98 |
| CONSTANT_SPEED | 128 | uniform only | 64/64 = 100% | 100% | 1.00 |

---

## 9. The experiments of handoff section 5

### 9.1 Oxygen step: 21% -> 7% -> 21%

Gray et al. (2004) Fig. 4d's protocol, run on a formed pattern at density 0.75
in a 128 x 128 box, three seeds, with each level held for 2500 model time units.
Run for two speed laws.

| Strain | H at end of air | H at end of hypoxia | Time to reach `uniform` | Time to re-form |
|---|---:|---:|---:|---:|
| NPR1_LIKE (monotone) | 1.33 - 1.38 | 0.52 - 0.54 | **never, within 2500** | 0 (never lost) |
| HYPOXIC_BRANCH (non-monotone) | 0.66 - 0.67 | 0.021 - 0.022 | **90 - 110** | 90 |

**The monotone speed law cannot reproduce the experiment, and the reason is
structural rather than a matter of run length.** Dropping the ambient level to
7% does remove the driving force — beta collapses, which is what the phase
diagram predicts — but it also puts *every* agent on the lower plateau of V(O),
at v_min. At the resulting O_eq the speed is 0.068 against 1.5 in air, so
D_W = V^2 / 2D_r falls from 4.5 to 0.0093, a factor of 480. Erasing a pattern of
wavelength 26 is diffusive and needs of order (26/2pi)^2 / D_W ~ 1900 time
units; the measured decay over 2500 units is from 1.33 to 0.53, exactly the
partial relaxation that timescale implies. **Hypoxia freezes the pattern rather
than dispersing it.**

Adding the extra high-speed branch at very low oxygen that Demir et al. (2020)
Fig. 2b reports changes this completely. `HYPOXIC_BRANCH` removes the driving
force *and* keeps the agents fast (V = 1.12 at O = 0.02, so D_W = 2.5, 296 times
the monotone value), and its pattern collapses to the noise floor in about 100
time units and re-forms in about 90.

This was the first version of the experiment's finding, not a tuned one: the
original run used a 400-unit hold and reported "never dissolves", which looked
like a bug and turned out to be the physics. It is worth stating plainly because
it is a limitation of Model 0's speed law inherited by Model 1's default
presets, and the notebook and README both say the monotone law does not model
the low-oxygen branch.

### 9.2 Hysteresis

Ambient ramped 0.40 -> 1.00 and back over 2000 time units at fixed population
(density 0.60, 128 x 128 box, three seeds each direction), after settling at the
starting level.

| | Patterned over | H at ambient 0.50 |
|---|---|---:|
| up ramp | ambient 0.82 - 1.00 | 0.035 |
| down ramp | ambient 0.40 - 1.00 | 0.834 |

The loop is large — a factor of 24 in H at the same ambient level — but **it
should not be read as bistability.** The predicted onset at this density is
ambient 0.60, and the up-ramp does not pattern until 0.82; the predicted growth
rate at onset is 0.0000 and only reaches 0.0027 by ambient 0.80, so the ramp
crosses the boundary far faster than anything can grow. The down-ramp persists
below the onset for the mirror-image reason measured in 9.1: once the ambient is
low, D_W collapses and the pattern cannot relax. Both arms of the loop are
kinetic, and a longer ramp would narrow it. This experiment on its own cannot
separate a genuine first-order transition from a slow approach to a unique
state, and nothing here claims it does.

### 9.3 Finite N

Density and ambient fixed (0.50, air), box and agent count scaled together,
five seeds each. **Every N from 500 to 20,000 patterned in every seed
(probability 1.0 throughout).**

That is a null result for the question asked, and the reason is that the
operating point was not close enough to the boundary: at density 0.50 the
predicted growth rate times the run length is 12.4, i.e. twelve e-foldings, so
noise has nothing to decide. It also exposes a design tension in the experiment
itself. At fixed density the box side scales as sqrt(N), while the predicted
wavelength is fixed by (ambient, density) alone — 49 units here — so N = 500
gives a box of 32 units, **0.65 of one wavelength**. Such a box cannot hold the
predicted pattern at all; what it does instead is condense into a single
box-filling blob, which the classifier correctly calls `dots`. Below roughly two
wavelengths per side the failure mode is geometric, not stochastic, and the two
cannot be separated in this design.

The top count was capped at 20,000 rather than the 50,000 the handoff suggests.
At 50,000 and this density the box is 316 x 316, and with the adaptive
substepping that engages once the pattern forms those five runs cost more than
the rest of the experiment suite combined; 500 to 20,000 is still a 40x range in
N and a 6x range in box side.

### 9.4 Spatial gradient

A cosine ambient profile from 0.35 to 1.00 across x, density 0.60, 176 x 176
box, t = 1500. The eight vertical slabs are labelled by their own local ambient
level:

| Slab local ambient | 0.38 | 0.55 | 0.80 | 0.97 | 0.97 | 0.80 | 0.55 | 0.38 |
|---|---|---|---|---|---|---|---|---|
| label | uniform | uniform | dots | dots | dots | dots | uniform | uniform |

The label tracks the local ambient exactly as the phase diagram says it should:
patterning in the oxygen-rich middle, uniform at the poor edges. That is the
spatial version of the phase diagram and it works.

The comparison against the stationary drift prediction rho ~ 1 / V(O) does
**not** work, and the reason is again the timescale:

| Region | correlation with 1/V(O) | local D_W | time to equilibrate across half the box |
|---|---:|---:|---:|
| patterned slabs (ambient 0.80 - 0.97) | **+0.87** | 0.10 - 0.83 | 2,300 - 19,000 |
| uniform slabs (ambient 0.38 - 0.55) | **-0.63** | 0.010 - 0.016 | 124,000 - 187,000 |
| whole profile | +0.53 | | |

The run is 1500 time units. In the oxygen-rich half that is comparable to the
equilibration time and the agreement is good; in the oxygen-poor half it is
smaller by a factor of 125 and the profile has barely begun to form, so the
correlation there is not merely weak but negative. Model 0 reported a
correlation of 0.99 for the same comparison, but on a **fixed** oxygen landscape
whose speeds never collapsed. This is not a contradiction of that result; it is
the same result in a regime where the feedback has driven the local mobility to
near zero.

### 9.5 A single cause behind three results

Sections 9.1, 9.2 and 9.4 are three observations of one thing: **in this model
the relaxation time diverges as ambient oxygen falls**, because a monotone
V(O) sends every agent to v_min and D_W ~ V^2. Any experiment that takes the
system to low ambient oxygen and then asks it to rearrange will be
rate-limited by that, not by the thermodynamics the phase diagram describes.
The phase diagram itself is unaffected, because it starts from a uniform state
in the oxygen-rich regime and only has to grow a pattern, not erase one.

---

## 10. Resolution convergence

One solidly unstable point (NPR1_LIKE, ambient 1.0, density 0.70, predicted
wavelength 26.9), three seeds at each of three resolutions. Physical size,
density, consumption footprint and diagnostic length are all held fixed.

| Variant | dt | dx | Labels (3 seeds) | Measured L | H |
|---|---:|---:|---|---:|---:|
| base | 0.1 | 1.0 | stripes, dots, dots | 44.2 +/- 9.4 | 1.194 |
| half dt | 0.05 | 1.0 | dots, dots, stripes | 49.6 +/- 9.4 | 1.241 |
| half dx | 0.1 | 0.5 | dots, dots, dots | 49.6 +/- 9.4 | 1.243 |

H agrees across resolutions to within 4%, and the measured length scale to
within 12%, which is inside the seed-to-seed scatter of +/- 9.4 in every case.

The labels do **not** agree in every seed: base and half-dt each give two `dots`
and one `stripes`, half-dx gives three `dots`. That disagreement is not a
resolution effect — it appears at the base resolution too — but a seed-level
ambiguity at an operating point that sits near the dots/stripes crossover, where
the area fraction is close to the 0.35 threshold. The honest reading is that
label agreement across resolutions holds to within one seed out of three at this
point, and would be firmer at an operating point further from a threshold.

Measured L (44-50) is well above the predicted wavelength (26.9). That is
coarsening, not disagreement: the prediction is for the fastest-growing mode at
t = 0 and the measurement is at t = 1200, by which time modes have merged. The
same ratio appears across the whole phase diagram (median 1.16 for NPR1_LIKE).

This is a robustness check, not a measured convergence rate. Three seeds at
three resolutions cannot establish an order of accuracy.

