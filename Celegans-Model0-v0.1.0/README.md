# C. elegans · Model 0

**Oxygen-mediated motility and aggregation · v0.1.0**

Can overlapping agents aggregate because their consumption changes oxygen,
and oxygen changes how fast they move? This project isolates that feedback.
It is a teaching model with arbitrary units, not a fitted reproduction of the
experiments in Demir et al. (2020).

## Start here

Use Python 3.10 or newer. Extract the ZIP, then open a terminal in the
`Celegans-Model0` folder (the folder containing this README).

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `01_oxygen_mediated_aggregation.ipynb`, select the Python environment in
which you installed the dependencies, and run the cells in order. VS Code with
the Jupyter extension also works. A virtual environment is recommended.
Interactive demos start **paused**; click Start or Advance frame. Static
reference figures are included, so the scientific narrative remains readable
without a live kernel. The notebook contains finite examples as well as widgets;
Run All does not start an endless simulation.

Both interactive and headless experiments use the same engine:

```python
import celegans as ce

demo = ce.Demo("feedback", seed=1)
demo.show()

result = ce.run("feedback", seed=1, turns=3000)
print(result.history["heterogeneity"][-1])
```

## What Model 0 includes

- Persistent point agents with independent angular Brownian motion.
- A positive, increasing sigmoid speed response to local oxygen.
- A periodic oxygen field with diffusion, replenishment and saturating consumption.
- A normalized Gaussian consumption footprint, with density measured in agents/area.
- Fixed-field controls, coupled feedback, and a seeded density perturbation.

Agents overlap freely. The model has no body mechanics, alignment, reversal
state, explicit oxygen-gradient steering, bacteria field, food depletion or
excluded volume. Heading marks in the plots are not worm bodies. Speed-dependent
residence time can produce a coarse-grained drift without a steering rule.

The monotone speed law is deliberately restricted in meaning: the paper reports
increased movement again at very low oxygen. Our law does not model that branch.
The effective consumption parameter is not a measured worm respiration rate;
the paper emphasizes bacteria-associated oxygen depletion. Neither aggregation
nor a high density-variance score alone establishes phase coexistence or swarming.

## Files and responsibilities

| File | Role |
|---|---|
| `01_oxygen_mediated_aggregation.ipynb` | Questions, model, live controls, experiments, interpretation |
| `celegans/__init__.py` | Compact public API |
| `celegans/model0/model.py` | Parameters, agent and oxygen dynamics, state, results |
| `celegans/model0/scenarios.py` | Initial conditions and fixed-field protocols |
| `celegans/model0/demo.py` | Interactive controls and cooperative animation loop |
| `celegans/model0/visualization.py` | Maps, heading marks, diagnostic plots |
| `celegans/model0/analysis.py` | Density heterogeneity, polar order and x profiles |
| `assets/` | Reproducible simulation reference figures and comparison data |
| `tests/test_model0.py` | Numerical invariants and reproducibility checks |
| `VALIDATION.md` | Checks performed for this release and their limits |

The structure follows the supplied SwarmRules v2.0.2 reference. Its files and
simulation models have not been modified or incorporated.

## Controls and reproducibility

Physical sliders apply immediately and retain the current state. Reset starts
again with the current parameters and chosen seed. For a clean comparison, change
one parameter and Reset. The complete parameter history is in
`result.parameter_log`; clean runs have one entry. `Steps / draw` affects only
viewing cadence; it does not change the time step. The simulation seed controls
both initial conditions and heading noise. Exact numerical reproducibility is
intended within the same package/dependency versions, not across all platforms.

`ce.Parameters` is immutable; use `dataclasses.replace` to form a new parameter
set. `Simulation.state` is an independent read-only snapshot. `Simulation.result()`
also includes diagnostic arrays and parameters. Use `sim.step(n)` for explicit
control and `sim.reset(seed=2)` to restart. Restarting is not a resume-from-snapshot
operation. The current API does not serialize RNG state for exact continuation.

## Numerics

The grid is cell-centered and periodic. Cloud-in-cell deposition conserves agent
count; a normalized periodic Gaussian spreads consumption over a fixed model
length. Field sampling is bilinear. Each internal step senses oxygen, turns and
moves agents, then advances oxygen using the **beginning-of-step density**.
Oxygen diffusion is explicit under a conservative stability bound; replenishment
is exact; consumption uses a positive implicit local solve. This is a first-order
split method. Adaptive internal steps also limit displacement, angular variance
and reaction rates. Stability is not proof of convergence: compare ensembles at
smaller `dt` and finer grids, preserving physical lengths and population density.

Run the included checks with:

```bash
python -m unittest discover -s tests -v
```

If widgets do not appear, check that the selected kernel has `ipywidgets`
installed, restart the kernel, and rerun the setup cell. Use `ce.run(...)` and
`ce.plot(...)` for static work. Close or pause demos before running longer
headless experiments. Creating another demo pauses the previously running one.

## Sources

- Demir, E., Yaman, Y. I., Basaran, M. & Kocabas, A. (2020).
  *Dynamics of pattern formation and emergence of swarming in Caenorhabditis elegans.*
  eLife 9:e52781. https://doi.org/10.7554/eLife.52781
- Cates, M. E. & Tailleur, J. (2015; arXiv 2014).
  *Motility-Induced Phase Separation.* Annual Review of Condensed Matter Physics
  6:219–244. https://arxiv.org/abs/1406.3533
  https://doi.org/10.1146/annurev-conmatphys-031214-014710

Original simulation figures are generated by this package; no paper figures or
measured datasets are redistributed. See the notebook for the distinction between
published evidence, derived consequences of our rules, and teaching assumptions.
