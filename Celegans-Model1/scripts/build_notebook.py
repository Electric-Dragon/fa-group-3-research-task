#!/usr/bin/env python3
"""Regenerate 01_environment_controlled_patterns.ipynb from source.

The notebook is kept as a build product so its prose and code stay in one
reviewable place and its outputs are always clean. Run from the project root:

    python scripts/build_notebook.py

Then execute it in Jupyter, or check it headlessly with nbclient.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t.strip('\n')))
def code(t): C.append(nbf.v4.new_code_cell(t.strip('\n')))

md(r"""
# C. elegans · Model 1 — environment-controlled pattern formation

**What this notebook is about.** Model 0 showed that oxygen-mediated motility can
make agents aggregate. This notebook asks the question an experimenter actually
faces: *the animals are what they are — what can you do to them with the
environment alone?*

Three things are new relative to Model 0.

1. **Agent parameters are frozen into named strains.** You cannot edit them.
2. **The environment is the control surface**, and it is live: ambient oxygen
   (a level, a map, or a schedule), the replenishment rate (a cover glass is a
   patch of low values), and the number of animals on the plate.
3. **The package predicts before it simulates.** Linear stability of the
   homogeneous state says where patterns *should* appear. An independent
   classifier says what *did* appear. Putting one on top of the other is the
   result.

Units are arbitrary model units with oxygen normalised so ambient air = 1.0.
This is a teaching model, not a fit to Demir et al. (2020).
""")

code(r"""
%matplotlib inline
import numpy as np
from pathlib import Path
from IPython.display import Image, display

from celegans import model1 as m1
from celegans.model1 import environment as envmod

ASSETS = Path('assets')
np.set_printoptions(precision=3, suppress=True)
print('strains:', list(m1.PRESETS))
print('21% O2 ->', envmod.pct(21), '   7% O2 ->', round(float(envmod.pct(7)), 3))
""")

md(r"""
## 1. What is fixed and what you control

| Fixed by the strain | Yours, live |
|---|---|
| `v_min`, `v_max`, `oxygen_midpoint`, `response_width` | ambient oxygen `O_am`: scalar, map, or schedule |
| `rotational_diffusion` — the paper's "tau", which is a **rate** | replenishment `f`: scalar or map |
| `consumption` q, `oxygen_half_saturation` K | oxygen diffusivity `D_O` |
| `kernel_width` (consumption footprint) | agent count, added or removed at any turn |

**The environment never writes to the oxygen field.** It acts only through the
replenishment term `f(x) · (O_am(x, t) − O)`, so the agents' own consumption
always competes with it. That is why an aggregate sitting inside an imposed
low-oxygen spot still pulls the local oxygen *below* the imposed level — the
feedback that makes patterns is still running.

Agents have no gradient sensing. They read the local oxygen level, set their
speed, move, and forget their heading at rate `D_r`. The aerotactic drift
`β = V V′ / (2 D_r)` is a consequence of speed modulation, not a steering rule.
""")

code(r"""
for name, strain in m1.PRESETS.items():
    print(f'{name:16s} v {strain.v_min:.2f}-{strain.v_max:.2f}  '
          f'midpoint {strain.oxygen_midpoint:.2f}  width {strain.response_width:.3f}  '
          f'D_r {strain.rotational_diffusion}  law {strain.speed_law}')
""")

md(r"""
## 2. The strain presets and their speed laws

`NPR1_LIKE` is steep: a small drop in oxygen costs a lot of speed, so `V′` — and
with it `β` — is large. `N2_LIKE` has a smaller speed span *and* a midpoint far
lower, so its agents only reach the responsive part of their curve once
consumption has pulled the oxygen much further down. `CONSTANT_SPEED` is the
control: `V′ = 0` everywhere, so `β = 0` and no oxygen-driven instability exists
at any density or ambient level.

The dots on the curves mark where the agents actually sit at Model 0's default
density in air. **Where you sit on the curve is what decides β**, and the curve
alone does not show that.
""")

code(r"""
density = 0.586                      # Model 0's default, agents per model area
env_air = envmod.uniform(envmod.pct(21))
O_eq = {name: m1.equilibrium_oxygen(name, 1.0, 0.06, density) for name in
        ('NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED')}
print({k: round(v, 3) for k, v in O_eq.items()})
m1.plot_speed_laws(['NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED'], mark=O_eq);
""")

md(r"""
## 3. The prediction, before any simulation

Linearising the mean-field limit around the homogeneous state gives a growth
rate for every wavenumber:

$$\lambda(k) = \text{largest eigenvalue of }
\begin{pmatrix} -D_W k^2 & -\beta W k^2 \\ -a\,g(k) & -(D_O k^2 + f + b)\end{pmatrix}$$

with $D_W = V^2/2D_r$, $\beta = V V'/2D_r$, $a = \partial(\text{sink})/\partial W$,
$b = \partial(\text{sink})/\partial O$, and $g(k) = e^{-\sigma^2k^2/2}$ the Fourier
transform of the consumption footprint. The state is unstable when
$\max_k \lambda(k) > 0$, which needs $\beta > 0$ — the agents must be on the
**rising** branch of `V(O)`.

Note $\lambda(0) = 0$ exactly: agent number is conserved, so a uniform shift is
not a growing mode.
""")

code(r"""
for W in (0.30, 0.60, 0.90, 1.20):
    p = m1.predict('NPR1_LIKE', env_air, W)
    print(f'W = {W:.2f}  O_eq = {p["O_eq"]:.3f}  beta = {p["beta"]:6.2f}  '
          + (f'UNSTABLE  wavelength {p["wavelength"]:5.1f}  '
             f'growth {p["growth_rate"]:.4f}' if p['unstable'] else 'stable'))
m1.plot_growth_curve('NPR1_LIKE', env_air, [0.30, 0.60, 0.90, 1.20]);
""")

md(r"""
### The predicted map

Ambient oxygen and density partly substitute for each other, which is why the
unstable region is a diagonal band rather than a threshold in either variable
alone: raising the density lowers the equilibrium oxygen, and so does lowering
the ambient level. Push the density too far and the equilibrium falls off the
*bottom* of the rising branch — so the band has an upper edge as well as a
lower one.
""")

code(r"""
ambients, densities = np.linspace(0.30, 1.00, 40), np.linspace(0.15, 1.60, 40)
predicted = {name: m1.predicted_map(name, ambients, densities)
             for name in ('NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED')}
for name, grid in predicted.items():
    print(f'{name:15s} unstable in {100 * grid["unstable"].mean():4.1f}% of the grid')
m1.visualization.plot_predicted_map('NPR1_LIKE', predicted['NPR1_LIKE']);
""")

code(r"""
# The unstable density window at three ambient levels, for all three presets.
for name in ('NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED'):
    windows = []
    for ambient in (0.5, 0.7, 1.0):
        lo, hi = m1.unstable_density_range(name, ambient)
        windows.append('never' if not np.isfinite(lo) else f'{lo:.2f}-{hi:.2f}')
    print(f'{name:15s} unstable density window at ambient '
          f'0.5 / 0.7 / 1.0:  ' + '   '.join(windows))
""")

md(r"""
## 4. The live demo

Sliders for the **environment** and buttons for the **population** — nothing
else. Changing the strain resets the run, because a strain is not something an
experiment changes mid-plate.

The demo starts **paused**. Click *Start*, or *Advance frame* for a finite step.
The right-hand panel shows where you are in the predicted phase diagram; the
structure factor shows the measured length scale against the predicted `k*`.

Try: start in air, let dots form, then drag ambient down to 7% and watch them
dissolve. Or add 3000 agents as a point blob and watch the aggregate seed.
""")

code(r"""
demo = m1.Demo('NPR1_LIKE', domain=(96.0, 96.0, 96, 96), n_agents=6000, seed=1)
demo.show()
""")

md(r"""
## 5. Phase diagrams: prediction against simulation

The sweeps below are **precomputed** — each is 192 runs of a 176 × 176 box for
1200 model time units, which is far too long for a notebook. Regenerate them
with:

```bash
python scripts/phase_diagram.py --strain NPR1_LIKE --time 1200 --seeds 3
```

Marker colour is the majority label across three seeds; opacity is the share of
seeds that agreed, so a washed-out marker is a cell the seeds disputed. The
lines are the predicted boundary from `stability.py`, computed without reference
to anything that was simulated.
""")

code(r"""
from celegans.model1.sweep import load_runs, majority_labels, agreement_with_prediction

summaries, checks = {}, {}
for name in ('NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED'):
    path = ASSETS / f'sweep_{name}.csv'
    if not path.exists():
        print(f'{name}: no precomputed sweep in assets/ — run scripts/phase_diagram.py')
        continue
    rows = load_runs(path)
    summaries[name] = majority_labels(rows)
    checks[name] = agreement_with_prediction(rows)
    labels, counts = np.unique(summaries[name]['label'], return_counts=True)
    print(f'{name:15s} {dict(zip(labels, counts))}   '
          f'agreement away from the boundary: '
          f'{checks[name]["agree"]}/{checks[name]["total"]} '
          f'({100 * checks[name]["fraction"]:.0f}%)')
""")

code(r"""
if summaries:
    m1.visualization.plot_phase_comparison(summaries);
""")

md(r"""
### Reading the diagram

- **Away from the predicted boundary the two agree closely.** That is the claim
  the sweep supports, and the number printed above is how closely.
- **On the boundary they blur.** Linear theory is infinite, deterministic and
  instantaneous; the simulation is finite-N, noisy and stopped at a finite time.
  Near the boundary the predicted growth rate goes to zero, so a cell that is
  formally unstable may not have grown anything yet at t = 1200 — and the
  predicted wavelength diverges, so what does grow may not fit in the box.
  `sweep.plan()` flags exactly those cells before the sweep runs.
- **`N2_LIKE` needs more density for the same effect**, at every ambient level.
- **`CONSTANT_SPEED` is uniform everywhere**, which is the control working.
""")

code(r"""
# Density ordering at the ambient level where the unstable band is widest.
if 'NPR1_LIKE' in summaries:
    s = summaries['NPR1_LIKE']
    j = int(np.argmax((s['label'] != 'uniform').sum(axis=0)))
    print(f'ambient = {s["ambient"][0, j]:.2f}')
    for i in range(s['label'].shape[0]):
        print(f'  W = {s["density"][i, j]:.2f}  {s["label"][i, j]:8s}  '
              f'phi = {s["area_fraction"][i, j]:.2f}  '
              f'L = {s["length_scale"][i, j]:5.1f}  '
              f'(predicted {"unstable" if s["predicted"][i, j] else "stable"}, '
              f'wavelength {s["predicted_wavelength"][i, j]:.0f})')
""")

md(r"""
## 6. An oxygen step: 21% → 7% → 21%

The protocol of Gray et al. (2004) Fig. 4d, run on a formed pattern. Dropping the
ambient level moves the equilibrium oxygen down off the rising branch of `V(O)`
and `β` collapses — so the aggregate should disperse, and restoring air should
bring it back. Nothing about the animals changes.

**It does not disperse, and the reason is worth more than the experiment.** With
the monotone speed law, low ambient oxygen does remove the driving force, but it
also puts *every* agent at `v_min`. The speed falls from 1.5 to 0.068, so
`D_W = V²/2D_r` falls by a factor of 480, and erasing a pattern is a diffusive
process. Hypoxia **freezes** the pattern instead of dispersing it: over 2500 time
units H only falls from 1.33 to 0.53.

The right panel adds `HYPOXIC_BRANCH` — the same strain plus the extra
high-speed branch at very low oxygen that Demir et al. (2020) Fig. 2b reports.
That removes the driving force *and* keeps the agents fast (296× the mobility at
2% O₂), and its pattern collapses to the noise floor in about 100 time units and
re-forms in about 90.

So reproducing the dispersal that Gray et al. observed **requires** the
non-monotone branch. A monotone V(O) cannot do it in principle, not merely in
practice — which is a concrete limitation of Model 0's speed law, inherited here.
""")

code(r"""
display(Image(filename=str(ASSETS / 'step_response.png'))) if (ASSETS / 'step_response.png').exists() else print('run: python scripts/experiments.py step')
""")

md(r"""
## 7. Hysteresis: ramp the ambient up, then down

At a fixed population, sweep the ambient level slowly across the predicted
boundary in both directions. Any gap between the two curves is a lag, and this
experiment on its own **cannot** separate a genuine bistability from a slow
approach to a unique steady state: the ramp is finite, and so is the time the
pattern needs to coarsen or to break up. Read the gap as "the state depends on
history over the timescale of this ramp", not as a first-order transition.
""")

code(r"""
display(Image(filename=str(ASSETS / 'hysteresis.png'))) if (ASSETS / 'hysteresis.png').exists() else print('run: python scripts/experiments.py hysteresis')
""")

md(r"""
## 8. A spatial gradient, and finite-N

**Left:** a cosine ambient profile across x. A periodic domain has no seam-free
linear ramp, so the profile runs low → high → low; half the domain carries each
branch. The label measured slab by slab tracks the local ambient level exactly as
the phase diagram says — patterning in the oxygen-rich middle, uniform at the
poor edges. That part works.

The comparison against the stationary drift target `ρ ~ 1/V(O)` works only where
there has been time for it to: the correlation is **+0.87 in the oxygen-rich
slabs and −0.63 in the poor ones**. Same cause as §6 — at the poor edges
`D_W ≈ 0.01`, so equilibrating the profile across half the box would take about
187,000 time units against a run of 1,500. Model 0 got 0.99 for this comparison
on a *fixed* landscape whose speeds never collapsed; this is the same physics in
a regime where the feedback has driven the mobility to nearly zero.

**Right:** at fixed density, scaling the box and the agent count together leaves
the prediction unchanged but shrinks the relative noise. Note the design tension
this experiment has: the box side grows as √N while the predicted wavelength is
fixed, so the smallest systems hold well under one wavelength and fail for
geometric rather than stochastic reasons. The figure reports both.
""")

code(r"""
for name in ('gradient.png', 'finite_n.png'):
    path = ASSETS / name
    display(Image(filename=str(path))) if path.exists() else print(f'run: python scripts/experiments.py')
""")

md(r"""
## 9. Inverse control: steering an aggregate

Everything so far is forward — *this environment does that to the animals*. The
inverse question is: given a path you want an aggregate to follow, what
environment makes it follow? The actuator is the weakest one available: a disc
where the gas being exchanged against is poor, moving along the path. It never
touches an agent, never writes to the oxygen field, and carries no information
about where the agents currently are. It is **open loop**.

Three numbers are searched with CMA-ES — the spot's radius, its oxygen depth,
and how fast it travels — each candidate scored by the mean tracking error over
at least two seeds, so a parameter set that only works for one initial
condition cannot win. They genuinely trade off: a deep narrow spot holds an
aggregate tightly but is easy to outrun, and a fast spot just leaves it behind.

```bash
python scripts/steer.py --search --generations 10 --processes 9
```

The prior art for the idea in synthetic active matter is Arlt et al. (2018) and
Frangipane et al. (2018), who shape bacterial density with patterned light.
""")

code(r"""
path = ASSETS / 'steering.png'
display(Image(filename=str(path))) if path.exists() else print('run: python scripts/steer.py --search')
""")

md(r"""
## 10. Limitations

**Of the prediction.** It is a drift-diffusion limit, so it needs the pattern
wavelength to be much larger than the persistence length `V / D_r`;
`predict()` reports the ratio and flags it below 5. It is linear, so it says
where a ripple starts to grow, not what it grows into. It is infinite-N and
deterministic; §8 shows what finite N does to it.

**Of the classifier.** The label thresholds — dense-phase area fraction 0.35 and
0.65, heterogeneity 3× the shot-noise level — are heuristics. They are returned
with every label, alongside the raw metrics, so labels can be recomputed without
rerunning anything. `stripes` also covers labyrinths; the classifier does not
separate them.

**Of the model.** As in Model 0: agents overlap freely, there is no body
mechanics, excluded volume, alignment, reversal state, bacteria field or food
depletion, and the speed law is monotone (the hypoxic high-speed branch of the
paper's Fig. 2b is available as `HYPOXIC_BRANCH` but is off the main path).
**A high heterogeneity score alone does not establish phase coexistence or
swarming.** Banding and swarming translation need a bacteria field and are out
of scope.

**Of the presets.** `NPR1_LIKE` and `N2_LIKE` are teaching presets tuned with
`stability.py` to give a clear contrast, not fits to strain data. `VALIDATION.md`
records the numbers, the reasoning, and what the sweeps did and did not show —
including which of the expected morphologies did **not** appear.
""")

nb['cells'] = C
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'},
})
out = Path(__file__).resolve().parents[1] / '01_environment_controlled_patterns.ipynb'
out.write_text(nbf.writes(nb))
print('wrote notebook with', len(C), 'cells')
