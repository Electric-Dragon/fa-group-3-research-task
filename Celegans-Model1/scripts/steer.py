#!/usr/bin/env python3
"""Inverse control (milestone M6): steer an aggregate with a moving low-O2 spot.

    python scripts/steer.py --search --generations 10 --processes 9
    python scripts/steer.py --radius 14 --level 0.12 --speed 0.08 --seeds 1 2 3

`--search` runs CMA-ES over (radius, level, speed), scoring each candidate by
the mean tracking error over at least two seeds. Without it, the given
parameters are simply evaluated and plotted. Results go to assets/.

The controller is open loop: the spot follows its prescribed path and never
sees where the agents are.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from celegans.model1 import control
from celegans.model1.engine import density_grid, DIAGNOSTIC_LENGTH
from celegans.model1.visualization import NAVY, TEAL, ORANGE

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def figure_for(runs, task, params, search_history=None):
    n = 2 + (search_history is not None)
    fig = Figure(figsize=(5.2 * n, 4.6), constrained_layout=True, facecolor='white')
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n)

    best = runs[0]
    rho = density_grid(np.asarray(best['state'].positions), best['params_object'],
                       smoothing=DIAGNOSTIC_LENGTH)
    axes[0].imshow(rho / rho.mean(), origin='lower', vmin=0, vmax=6, cmap='magma',
                   extent=(0, task.side, 0, task.side), interpolation='nearest')
    track = best['track']
    axes[0].plot(track[:, 3], track[:, 4], color='white', lw=1.4, ls='--',
                 label='commanded path')
    axes[0].plot(track[:, 1], track[:, 2], color=ORANGE, lw=1.8, label='aggregate')
    axes[0].legend(frameon=False, fontsize=8, labelcolor='white', loc='lower left')
    axes[0].set(xlabel='x [model length]', ylabel='y [model length]')
    axes[0].set_title(f'seed {best["seed"]}: final density and tracks',
                      loc='left', color=NAVY, weight='bold', fontsize=10)

    for run, color in zip(runs, (TEAL, ORANGE, '#7d4a8f', '#4a7f3f')):
        axes[1].plot(run['track'][:, 0], run['track'][:, 5], color=color, lw=1.8,
                     label=f'seed {run["seed"]}')
    axes[1].axhline(params['radius'], color=NAVY, ls=':', lw=1.4,
                    label='spot radius')
    axes[1].set(xlabel='Time [model time]', ylabel='Distance to commanded point',
                ylim=(0, None))
    axes[1].set_title('Tracking error', loc='left', color=NAVY, weight='bold', fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.18)
    axes[1].spines[['top', 'right']].set_visible(False)

    if search_history is not None:
        gens = [h['generation'] for h in search_history]
        axes[2].plot(gens, [h['best'] for h in search_history], color=TEAL, lw=2,
                     label='best so far')
        axes[2].plot(gens, [h['median'] for h in search_history], color='0.6', lw=1.5,
                     label='generation median')
        axes[2].set(xlabel='CMA-ES generation', ylabel='Mean tracking error')
        axes[2].set_title('Search progress', loc='left', color=NAVY, weight='bold',
                          fontsize=10)
        axes[2].legend(frameon=False, fontsize=8)
        axes[2].grid(alpha=0.18)
        axes[2].spines[['top', 'right']].set_visible(False)

    fig.suptitle('Inverse control: a moving low-oxygen spot steers an aggregate  ·  '
                 + ', '.join(f'{k} {v:.3f}' for k, v in params.items()),
                 fontsize=12.5, color=NAVY, weight='bold')
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--population', type=int, default=8)
    parser.add_argument('--search-seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--radius', type=float, default=14.0)
    parser.add_argument('--level', type=float, default=0.12)
    parser.add_argument('--speed', type=float, default=0.08)
    parser.add_argument('--side', type=float, default=120.0)
    parser.add_argument('--processes', type=int, default=None)
    args = parser.parse_args(argv)

    task = control.SteeringTask(side=args.side, cells=int(args.side))
    params = dict(radius=args.radius, level=args.level, speed=args.speed)
    history = None
    if args.search:
        print(f'CMA-ES over {control.PARAMETER_ORDER}, '
              f'{args.generations} generations x {args.population} candidates x '
              f'{len(args.search_seeds)} seeds')
        outcome = control.search(task, seeds=tuple(args.search_seeds),
                                 generations=args.generations,
                                 population=args.population,
                                 processes=args.processes, start=params)
        params, history = outcome['params'], outcome['history']
        print('best parameters:', {k: round(v, 4) for k, v in params.items()},
              f'  error {outcome["value"]:.2f}')

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        runs = list(pool.map(_verify, [(params, seed, task) for seed in args.seeds]))

    radius = params['radius']
    held = [r for r in runs if r['error'] < radius]
    print(f'\ntracking error by seed: '
          + ', '.join(f'seed {r["seed"]}: {r["error"]:.2f}' for r in runs))
    print(f'{len(held)}/{len(runs)} seeds kept the aggregate within one spot '
          f'radius ({radius:.1f}) on average')

    ASSETS.mkdir(exist_ok=True)
    figure_for(runs, task, params, history).savefig(ASSETS / 'steering.png', dpi=150)
    np.savez_compressed(
        ASSETS / 'steering.npz',
        **{f'track_{r["seed"]}': r['track'] for r in runs},
        params=json.dumps(params),
        history=json.dumps(history or []),
        summary=json.dumps([dict(seed=r['seed'], error=r['error'],
                                 final_error=r['final_error'],
                                 worst_error=r['worst_error']) for r in runs]))
    print(f'wrote {ASSETS / "steering.png"}')
    return 0


def _verify(payload):
    params, seed, task = payload
    return control.trial(params, seed, task, record=True)


if __name__ == '__main__':
    raise SystemExit(main())
