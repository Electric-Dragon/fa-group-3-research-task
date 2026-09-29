# C. elegans · Model 1

**Environment-controlled pattern formation · v1.0.0**

Model 0 asked whether oxygen-mediated motility can make agents aggregate.
Model 1 asks a sharper question: **given a strain whose parameters you cannot
touch, what can you do to it with the environment alone?**

The agent parameters are frozen into named **strains**. The only controls are
the ones an experimenter actually has: ambient oxygen (a level, a map, or a
schedule), how fast the gas is exchanged (a replenishment map — a cover glass
is a patch of small values), and how many animals are on the plate. Those can
all be changed while the simulation is running.

The package does four things with that setup:

1. **Predicts** where patterns should appear, from linear stability of the
   homogeneous state — before any simulation runs.
2. **Simulates**, on Model 0's numerics, with the environment and the
   population free to change mid-run.
3. **Classifies** what actually formed — uniform, dots, stripes or holes —
   automatically, from the density field.
4. **Sweeps** density against ambient oxygen into a phase diagram, with the
   predicted boundary drawn over the simulated labels.

Like Model 0 this is a teaching model in arbitrary units. It is not a fitted
reproduction of Demir et al. (2020), and the strain presets are named after
real strains because they reproduce a qualitative contrast, not because any
number in them was measured in a worm.

## Start here

Use Python 3.10 or newer. From the folder containing this README:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `01_environment_controlled_patterns.ipynb` and run the cells in order.
Interactive demos start **paused**; click Start or Advance frame. Static
reference figures are included, so the argument is readable without a live
kernel, and Run All does not start an endless simulation.

```python
from celegans import model1 as m1

env = m1.environment.uniform(m1.environment.pct(21))     # 21% O2 = 1.0

# Predict first.
print(m1.predict('NPR1_LIKE', env, density=0.7)['wavelength'])

# Then simulate.
sim = m1.Simulation('NPR1_LIKE', env, domain=(176, 176, 176, 176), n_agents=21700)
sim.step(12000)
print(sim.result().patterns['label'])

# Then change the world underneath it.
sim.set_environment(m1.environment.uniform(m1.environment.pct(7)))
sim.add_agents(5000, mode='point', center=(88, 88), spread=6.0)
sim.step(4000)
```

Model 0 is still importable at the top level and is unchanged:

```python
import celegans as ce
result = ce.run("feedback", seed=1, turns=3000)      # Model 0, verbatim
```

## What is fixed and what you control

| Fixed by the strain | Controlled by you, live |
|---|---|
| `v_min`, `v_max`, `oxygen_midpoint`, `response_width` | ambient oxygen `O_am`: scalar, map, or schedule |
| `rotational_diffusion` (the paper's "tau", a **rate**) | replenishment `f`: scalar or map (cover glass) |
| `consumption` q, `oxygen_half_saturation` K | oxygen diffusivity `D_O` |
| `kernel_width` (consumption footprint) | agent count, added or removed at any turn |

Presets: `NPR1_LIKE` (steep response, patterns readily), `N2_LIKE` (weaker
aerotaxis, patterns only at high density), `CONSTANT_SPEED` (control, beta = 0,
must never pattern), `MODEL0_DEFAULT` (Model 0's defaults, the regression
anchor), `HYPOXIC_BRANCH` (optional non-monotone speed law, off the main path).

**The environment never writes to the oxygen field.** It acts only through the
replenishment term `f(x) * (O_am(x, t) - O)`, so the agents' own consumption
always competes with it. A "low-oxygen spot" is a place where the gas being
exchanged against is poor, not a place where oxygen is deleted — which is why
the aggregate inside a spot still depresses the oxygen below the imposed level.

## Prediction and measurement are separate

`stability.py` linearises the mean-field limit of the agent rule and returns
whether a ripple grows and at what wavelength. It never runs a simulation.
`patterns.py` measures what a density field looks like and never consults the
prediction. The phase diagram puts one on top of the other; that overlay is the
result, and it is only meaningful because neither half saw the other.

Linear theory predicts *whether*, not *what*: dots, stripes and holes are all
the same instability, distinguished by how much area the dense phase occupies.
That is measured, not predicted.

## What Model 1 does not include

Bacteria field, food depletion, banding, swarming translation, body mechanics,
excluded volume, alignment, reversals, explicit oxygen-gradient steering, and
any fit to real worm data. Agents sense only the local oxygen level and set
their speed from it; the aerotactic drift `beta` is an emergent consequence of
speed modulation, not a steering rule.

As in Model 0: **a high heterogeneity score alone does not establish phase
coexistence or swarming.** The classifier's thresholds are heuristics, and every
raw number behind a label is returned with it so labels can be recomputed.

One consequence of the monotone speed law is worth knowing before you run
anything. At low ambient oxygen every agent sits at `v_min`, so `D_W = V²/2D_r`
collapses — by a factor of 480 between air and 7% O₂ at the default preset. The
model therefore **freezes** at low oxygen rather than dispersing, and three of
the experiments in `VALIDATION.md` are rate-limited by that rather than by the
physics they were meant to probe. The optional `HYPOXIC_BRANCH` strain, which
adds the high-speed low-oxygen branch of Demir et al. Fig. 2b, does disperse.

## Files and responsibilities

| File | Role |
|---|---|
| `01_environment_controlled_patterns.ipynb` | Scope, presets, prediction, demo, phase diagrams, experiments, limits |
| `celegans/__init__.py` | Re-exports Model 0 unchanged; exposes `model1` |
| `celegans/model0/` | Model 0 v0.1.0, copied verbatim, untouched — the regression baseline |
| `celegans/model1/strains.py` | Frozen agent parameters and the speed laws |
| `celegans/model1/environment.py` | Ambient/replenishment fields, schedules, spots, gradients |
| `celegans/model1/engine.py` | Simulation with a live environment and live population |
| `celegans/model1/stability.py` | Dispersion relation, predicted maps, instability boundaries |
| `celegans/model1/patterns.py` | Structure factor, periodic topology, classifier |
| `celegans/model1/sweep.py` | Parallel (ambient x density x seed) sweeps with resume |
| `celegans/model1/demo.py` | Environment-only interactive controls |
| `celegans/model1/visualization.py` | Maps, dispersion curves, phase diagrams |
| `celegans/model1/control.py` | Inverse control: steering an aggregate, and the CMA-ES that tunes it |
| `scripts/phase_diagram.py` | Command-line batch sweep |
| `scripts/experiments.py` | The oxygen step, hysteresis, finite-N, gradient and convergence runs |
| `scripts/steer.py` | Inverse-control search and its figure |
| `scripts/build_notebook.py` | Regenerates the notebook from source, outputs always clean |
| `assets/` | Precomputed sweep results and reference figures used by the notebook |
| `tests/test_model0.py` | Model 0's nine invariants, unchanged |
| `tests/test_model1.py` | Stability oracle, array environments, population, topology, classifier |
| `VALIDATION.md` | What was checked for this release, with numbers and limits |

## Reproducibility

Everything stochastic — initial conditions, heading noise, and the positions and
headings of agents added at runtime — comes from one generator seeded per run,
so a scripted population history replays exactly. Drawing never consumes RNG
state. `sim.parameter_log` records every environment swap and population change
with its turn and model time; a clean run has one entry.

Sweep seeds come from a CRC of the cell coordinates, not Python's `hash`, which
is salted per process and would make sweeps unrepeatable. Sweeps resume by
skipping rows already in `runs.csv`.

Exact numerical reproducibility is intended within the same package and
dependency versions, not across all platforms.

## Numerics

Model 0's, unchanged in substance. Periodic cell-centred grid, arrays indexed
`[y, x]` and positions `(x, y)`. Cloud-in-cell deposition conserves agent count;
a normalised periodic Gaussian spreads consumption over a fixed model length;
field sampling is bilinear. Each substep senses oxygen, turns and moves agents,
then advances oxygen using the beginning-of-substep density. Oxygen diffusion is
explicit under a conservative stability bound, replenishment is exact, and the
saturating sink is a positive implicit local solve — so **oxygen is never
clipped** to hide an unstable solve.

Two changes. Deposition uses `np.bincount` on the flattened cell index instead
of `np.add.at`, and the cloud-in-cell weights are computed once per substep and
reused for both deposition and sampling. Both are arithmetic reorganisations:
Model 1 with `MODEL0_DEFAULT` and a uniform environment reproduces Model 0's
trajectories, agreeing to 1.1e-14 on positions and 3.3e-16 on the oxygen field
after 200 turns (the test asserts 1e-10 and 1e-12). The residual is float
summation order, not a different method.

Ambient is resolved **once per outer turn**, at the time the turn begins.
Substeps exist for numerical stability, not to resolve a protocol, so a schedule
resolves changes to within `dt`.

Stability bounds are not proof of convergence. Compare ensembles at smaller `dt`
and finer grids, preserving physical size and density; `VALIDATION.md` reports
one such comparison.

```bash
python -m unittest discover -s tests -v
python scripts/phase_diagram.py --strain NPR1_LIKE --time 1200 --seeds 3
python scripts/experiments.py --all
python scripts/steer.py --search
python scripts/build_notebook.py
```

If widgets do not appear, check that the selected kernel has `ipywidgets`
installed, restart the kernel, and rerun the setup cell. Use `m1.run(...)` and
`m1.plot(...)` for static work. Close or pause demos before running longer
headless experiments; creating another demo pauses the previously running one.

## Sources

- Demir, E., Yaman, Y. I., Basaran, M. & Kocabas, A. (2020).
  *Dynamics of pattern formation and emergence of swarming in Caenorhabditis elegans.*
  eLife 9:e52781. https://doi.org/10.7554/eLife.52781
- Gray, J. M. et al. (2004). *Oxygen sensation and social feeding mediated by a
  C. elegans guanylate cyclase homologue.* Nature 430:317–322.
  https://doi.org/10.1038/nature02714
- Cates, M. E. & Tailleur, J. (2015). *Motility-Induced Phase Separation.*
  Annual Review of Condensed Matter Physics 6:219–244.
  https://arxiv.org/abs/1406.3533
- Tailleur, J. & Cates, M. E. (2008). *Statistical mechanics of interacting
  run-and-tumble bacteria.* Phys. Rev. Lett. 100:218103.
- Liebchen, B. & Löwen, H. (2018). *Synthetic chemotaxis and collective behavior
  in active matter.* Acc. Chem. Res. 51:2982.
- Arlt, J. et al. (2018). *Painting with light-powered bacteria.*
  Nat. Commun. 9:768.
- Frangipane, G. et al. (2018). *Dynamic density shaping of photokinetic
  E. coli.* eLife 7:e36608.

Original simulation figures are generated by this package; no paper figures or
measured datasets are redistributed. Model 0 v0.1.0 is included unmodified as a
regression baseline and is not re-released.
