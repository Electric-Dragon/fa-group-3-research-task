#!/usr/bin/env python3
"""Batch phase-diagram sweep for one strain. Run from the project root:

    python scripts/phase_diagram.py --strain NPR1_LIKE
    python scripts/phase_diagram.py --strain N2_LIKE --time 1200 --seeds 3
    python scripts/phase_diagram.py --strain CONSTANT_SPEED --seeds 2 --out results/control

Writes results/<strain>_<timestamp>/runs.csv plus config.json, then a phase
diagram PNG with the predicted instability boundary drawn over the simulated
labels. Rerunning the same output directory resumes: rows already in runs.csv
are skipped, so an interrupted sweep continues where it stopped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from celegans import model1 as m1
from celegans.model1.sweep import (SweepConfig, sweep_phase_diagram, load_runs,
                                   majority_labels, agreement_with_prediction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--strain', default='NPR1_LIKE', choices=list(m1.PRESETS))
    parser.add_argument('--ambient-min', type=float, default=0.30)
    parser.add_argument('--ambient-max', type=float, default=1.00)
    parser.add_argument('--density-min', type=float, default=0.30)
    parser.add_argument('--density-max', type=float, default=1.20)
    parser.add_argument('--n-ambient', type=int, default=8)
    parser.add_argument('--n-density', type=int, default=8)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--side', type=float, default=176.0,
                        help='square domain side in model length units')
    parser.add_argument('--dx', type=float, default=1.0, help='grid spacing')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--time', type=float, default=1200.0,
                        help='model time per run')
    parser.add_argument('--replenishment', type=float, default=0.06)
    parser.add_argument('--processes', type=int, default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--save-fields', action='store_true')
    parser.add_argument('--plot-only', action='store_true',
                        help='re-render figures from an existing --out directory')
    args = parser.parse_args(argv)

    cells = int(round(args.side / args.dx))
    config = SweepConfig(
        strain=args.strain,
        ambients=tuple(np.round(np.linspace(args.ambient_min, args.ambient_max,
                                            args.n_ambient), 4)),
        densities=tuple(np.round(np.linspace(args.density_min, args.density_max,
                                             args.n_density), 4)),
        seeds=args.seeds, width=args.side, height=args.side, nx=cells, ny=cells,
        dt=args.dt, run_time=args.time, replenishment=args.replenishment,
        save_fields=args.save_fields)

    if args.plot_only:
        out_dir = Path(args.out)
        rows = load_runs(out_dir)
    else:
        out_dir, rows = sweep_phase_diagram(config, args.out,
                                            processes=args.processes,
                                            resume=not args.no_resume)
    summary = majority_labels(rows)
    check = agreement_with_prediction(rows)

    figure = m1.plot_phase_diagram(
        summary, config.strain, replenishment=config.replenishment,
        title=(f'{config.strain} · {config.width:.0f} x {config.height:.0f} box · '
               f't = {config.run_time:.0f} · {config.seeds} seeds'))
    figure.savefig(out_dir / 'phase_diagram.png', dpi=150)
    m1.visualization.plot_metric_map(
        summary, 'heterogeneity', cmap='magma',
        label='Heterogeneity H', title=f'{config.strain}: measured H'
    ).savefig(out_dir / 'heterogeneity_map.png', dpi=150)

    print()
    print(f'results in {out_dir}')
    print(f'labels found: {sorted(set(summary["label"].ravel()))}')
    print(f'agreement with prediction, excluding cells adjacent to the boundary: '
          f'{check["agree"]}/{check["total"]}'
          + (f' = {100 * check["fraction"]:.1f}%' if check['total'] else ''))
    disagreeing = np.argwhere(check['interior'] & (check['predicted'] != check['observed']))
    for i, j in disagreeing:
        print(f'  disagree: ambient {summary["ambient"][i, j]:.3g}, '
              f'density {summary["density"][i, j]:.3g}: '
              f'predicted {"unstable" if check["predicted"][i, j] else "stable"}, '
              f'observed {summary["label"][i, j]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
