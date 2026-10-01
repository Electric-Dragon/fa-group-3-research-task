"""Headless check: density x ambient grid for one strain, saved as a montage.

    python check.py                 # npr-1, 10 simulated minutes, as in Fig. 3c
    python check.py --strain N2 --minutes 20

Each tile is the smoothed worm density, darker = more worms (all tiles share a scale).
The printed table gives density contrast (std/mean; ~0.1 uniform, >~0.5 patterned)
next to the paper's instability criterion.
"""
import argparse
import os
import time

import numpy as np
import taichi as ti

from model2 import params as P, sim

DENSITIES = (10, 20, 40, 60, 90)
AMBIENTS = (0.01, 0.07, 0.14, 0.17, 0.21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strain', default='npr-1', choices=[s.name for s in P.STRAINS])
    ap.add_argument('--minutes', type=float, default=10.0)
    ap.add_argument('--arch', default='gpu')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    strain = next(s for s in P.STRAINS if s.name == args.strain)
    sim.init(args.arch)
    s = sim.Simulation(strain=strain)
    steps = int(args.minutes * 60 / P.DT)

    tile = 256
    montage = np.ones((len(AMBIENTS) * (tile + 8), len(DENSITIES) * (tile + 8)), np.float32)
    print(f'{args.strain}, {args.minutes:g} min')
    print('density  ambient  contrast  criterion')
    t0 = time.time()
    for r, density in enumerate(DENSITIES):
        for c, ambient in enumerate(AMBIENTS):
            s.density, s.ambient = density, ambient
            s.reset(seed=r * 10 + c)
            s.step(steps)
            w = s.W.to_numpy() / density
            small = np.kron(w, np.ones((tile // len(w),) * 2))
            y, x = (len(AMBIENTS) - 1 - c) * (tile + 8), r * (tile + 8)
            montage[y:y + tile, x:x + tile] = 1 - np.clip(small / 4, 0, 1)
            crit = 'unstable' if P.unstable(strain, density, ambient) else '-'
            print(f'{density:7d}  {ambient * 100:6.0f}%  {s.contrast():8.2f}  {crit}')
    print(f'{time.time() - t0:.0f} s wall')

    os.makedirs(args.out, exist_ok=True)
    name = f'{args.out}/grid_{args.strain}.png'
    # imwrite takes [x, y] with y up: columns = density, rows = ambient (top = highest)
    ti.tools.imwrite((montage.T[:, ::-1] * 255).astype(np.uint8), name)
    print('saved', name, '(columns: density', DENSITIES, '/mm2; rows top->bottom: ambient',
          [f'{a * 100:g}%' for a in reversed(AMBIENTS)], ')')


if __name__ == '__main__':
    main()
