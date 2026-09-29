# Model 0 validation — v0.1.0

Checked on 29 September 2026. These checks support the teaching demonstrations;
they do not establish biological calibration or a thermodynamic phase diagram.

## Numerical invariants

All nine checks in `tests/test_model0.py` passed:

- cloud-in-cell plus Gaussian density conserves agent count, including periodic shifts;
- periodic diffusion conserves oxygen mass without sources or sinks and damps a
  Fourier mode at the expected discrete rate;
- replenishment agrees with its analytic exponential solution;
- a strong saturating sink remains finite, nonnegative, and depletes rather than creates oxygen;
- grouping turns into different calls, and collecting intermediate results, leaves trajectories unchanged;
- reset reproduces initial conditions and subsequent trajectories at a fixed seed;
- a prescribed landscape remains fixed, and a uniform no-consumption field remains uniform;
- positions remain periodic and oxygen remains bounded in a diffusion/substep stress test;
- invalid numerical and physical parameter values are rejected.

The reset, snapshot and chunking checks are combined in one automated test;
the list above describes the properties covered, not a one-to-one list of methods.

## Mechanism and resolution checks

At model time 300, using seeds 1, 2 and 3, the default full coupling produced much
larger density heterogeneity than the controls. H uses the fixed observation
length of 2 model units. The initial default H was 0.026–0.036.

| Experiment | Grid | dt | Final H, mean ± sample SD (3 seeds) |
|---|---|---:|---:|
| Full feedback, defaults | 64 × 48 | 0.1 | 0.858 ± 0.046 |
| Full feedback, smaller time step | 64 × 48 | 0.05 | 0.786 ± 0.068 |
| Full feedback, finer grid | 128 × 96 | 0.05 | 0.810 ± 0.094 |
| Constant-speed control | 64 × 48 | 0.1 | 0.0258 ± 0.0033 |
| No-consumption control | 64 × 48 | 0.1 | 0.0258 ± 0.0033 |

The refined cases preserve physical domain size, population, consumption
footprint and diagnostic length. All comparisons use the same physical final
time. This is a qualitative robustness check, **not a demonstrated convergence
rate or a precision error bar**. More seeds and refinements would be needed for
quantitative parameter inference or instability thresholds. The standard
deviation describes variability among three runs, not a confidence interval.

The fixed-landscape experiment also reproduced the expected location of
accumulation. In a 12,000-turn run with the notebook's landscape parameters,
the 16-bin density profile had correlation approximately 0.99 with the
inverse-speed stationary target. This finite-run comparison does not establish
complete equilibration, and the analytic continuous-field target is sampled on
a finite grid in the implementation.

`assets/reference_metrics.json` records the exact defaults, per-seed values,
and numerical library versions. The included PNG figures use default parameters,
seed 1 and 3000 turns. The no-consumption and constant-speed control curves
nearly coincide because the no-consumption field stays at ambient, where the
default sigmoid speed is almost v_max.

## Notebook and interface checks

- Notebook format validated with nbformat 5.10.4.
- All nine code cells ran in order in an in-process IPython session, including
  the Matplotlib inline setup, finite examples and widget construction.
- Actual ipywidgets 8.1.7 controls were exercised in an asyncio loop: Start,
  Pause, completion at the stopping turn, Reset, changed seed, changed physical
  sliders, speed-response toggle, Advance frame and disabled fixed-field controls.
- Starting/creating another demo pauses the previous active demo.
- The two included reference figures were rendered and visually inspected.

A separate Jupyter kernel could not start in the validation environment because
its socket/interface operations were restricted. The checks above therefore do
not claim a live JupyterLab or VS Code browser/frontend test. The delivered
notebook has clean outputs and static reference images; its widget panels need
a local running kernel with the listed dependencies.

Numerical environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8. Notebook/interface checks used IPython 8.37.0 and
ipywidgets 8.1.7. Exact trajectories across dependency versions or platforms
are not guaranteed.
