#!/usr/bin/env python3
"""Render the pattern-formation time-lapse: four outcomes, twelve timepoints each.

    python scripts/timelapse_figure.py

Runs three conditions that sit at different densities along the NPR1_LIKE phase
diagram at ambient air, plus the constant-speed control, and lays their
coarse-grained density out as a grid of snapshots. The three NPR1_LIKE rows use
the same strain and the same air; only the number of animals differs, which is
the point -- morphology is set by density, not by anything about the agents.

Writes assets/timelapse.png and assets/timelapse.json (the latter feeds the
animated version in the write-up).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from celegans.model1.visualization import NAVY, LABEL_COLORS

ASSETS = Path(__file__).resolve().parents[1] / 'assets'
SIDE, CELLS, OUT = 128.0, 128, 64
TIMES = [0, 50, 100, 200, 350, 500, 750, 1000, 1400, 1800, 2200, 2600]
SHOWN = [0, 2, 4, 6, 8, 11]          # columns drawn in the static figure

CASES = [
    ('dots', 'NPR1_LIKE', 1.0, 0.56, 'low density'),
    ('stripes', 'NPR1_LIKE', 1.0, 0.81, 'medium density'),
    ('holes', 'NPR1_LIKE', 1.0, 1.07, 'high density'),
    ('control', 'CONSTANT_SPEED', 1.0, 0.81, 'beta = 0 control'),
]


def _run(case):
    """One condition, sampled at every time in TIMES. Picklable for the pool."""
    name, strain, ambient, density, _ = case
    from celegans.model1.engine import Simulation, density_grid, DIAGNOSTIC_LENGTH
    from celegans.model1 import environment as envmod
    from celegans.model1.patterns import classify
    n = int(round(density * SIDE * SIDE))
    sim = Simulation(strain, envmod.uniform(ambient), domain=(SIDE, SIDE, CELLS, CELLS),
                     n_agents=n, dt=0.1, seed=3, record_every=10**9,
                     record_patterns=False)
    fields, labels, hetero = [], [], []
    for t in TIMES:
        sim.run_until(float(t))
        rho = density_grid(sim.state.positions, sim.params, smoothing=DIAGNOSTIC_LENGTH)
        metrics = classify(rho, sim.params.dx, sim.params.dy)
        labels.append(metrics['label'])
        hetero.append(round(float(metrics['heterogeneity']), 4))
        k = CELLS // OUT
        fields.append((rho / rho.mean()).reshape(OUT, k, OUT, k).mean(axis=(1, 3)))
    return dict(name=name, strain=strain, ambient=ambient, density=density,
                n_agents=n, labels=labels, heterogeneity=hetero,
                fields=np.stack(fields))


def main(processes=None):
    with ProcessPoolExecutor(max_workers=processes or 4) as pool:
        results = list(pool.map(_run, CASES))

    rows, cols = len(results), len(SHOWN)
    fig = Figure(figsize=(1.75 * cols + 2.1, 1.75 * rows + 1.0),
                 constrained_layout=True, facecolor='white')
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols)
    for r, (result, case) in enumerate(zip(results, CASES)):
        for c, i in enumerate(SHOWN):
            ax = axes[r, c]
            ax.imshow(result['fields'][i], origin='lower', vmin=0, vmax=3,
                      cmap='magma', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f't = {TIMES[i]}', fontsize=8.5, color=NAVY, pad=3)
            # A band across the bottom of each panel naming the classifier's call.
            label = result['labels'][i]
            ax.add_patch(Rectangle((0, 0), OUT, OUT * 0.085,
                                   color=LABEL_COLORS[label], zorder=5))
            ax.text(OUT / 2, OUT * 0.042, label, ha='center', va='center',
                    fontsize=7.5, zorder=6,
                    color='white' if label in ('stripes', 'holes') else '#222')
        axes[r, 0].set_ylabel(f'{case[0]}\n{case[4]}\nW = {case[3]:.2f}',
                              fontsize=9, color=NAVY, rotation=0,
                              ha='right', va='center', labelpad=50)
    fig.suptitle('Pattern formation over time  ·  same strain, same air, different density'
                 '  ·  band under each frame = the classifier\'s label',
                 fontsize=12.5, color=NAVY, weight='bold')
    fig.savefig(ASSETS / 'timelapse.png', dpi=150)
    print(f'wrote {ASSETS / "timelapse.png"}')

    # Quantised frames for the animated version: relative density clipped to
    # [0, 3] and scaled to a byte, which is all the colour ramp resolves.
    payload = dict(grid=OUT, side=SIDE, times=TIMES, cases=[
        dict(name=r['name'], strain=r['strain'], ambient=r['ambient'],
             density=r['density'], n_agents=r['n_agents'], labels=r['labels'],
             heterogeneity=r['heterogeneity'],
             frames=[(np.clip(f / 3.0, 0, 1) * 255).round().astype(np.uint8).ravel().tolist()
                     for f in r['fields']])
        for r in results])
    (ASSETS / 'timelapse.json').write_text(json.dumps(payload))
    print(f'wrote {ASSETS / "timelapse.json"}')
    for r in results:
        print(f'  {r["name"]:9s} N={r["n_agents"]:6d} {r["labels"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
