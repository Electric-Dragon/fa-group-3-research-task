#!/usr/bin/env python3
"""The Model 1 experiments of handoff section 5, saved as assets for the notebook.

    python scripts/experiments.py --all
    python scripts/experiments.py step hysteresis
    python scripts/experiments.py gradient --processes 6

Each experiment writes a .npz of raw arrays and a .png into assets/, so the
notebook can present them without running anything long. Every run goes through
the ordinary engine; nothing here is a special-cased code path.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from celegans import model1 as m1
from celegans.model1 import environment as envmod
from celegans.model1.engine import Simulation, density_grid, DIAGNOSTIC_LENGTH
from celegans.model1.patterns import classify
from celegans.model1.visualization import NAVY, TEAL, ORANGE, PLUM, LABEL_COLORS

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def _figure(size):
    fig = Figure(figsize=size, constrained_layout=True, facecolor='white')
    FigureCanvasAgg(fig)
    return fig


def _tidy(ax):
    ax.grid(alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)


# -- 2. oxygen step response --------------------------------------------

def _step_run(args):
    """21% -> 7% -> 21%, the Gray et al. (2004) Fig. 4d protocol."""
    seed, strain, side, density, hold, dt = args
    n = int(round(density * side * side))
    env = envmod.schedule([(0.0, envmod.pct(21)),
                           (hold, envmod.pct(7)),
                           (2 * hold, envmod.pct(21))])
    sim = Simulation(strain, env, domain=(side, side, int(side), int(side)),
                     n_agents=n, dt=dt, seed=seed, record_every=100,
                     record_patterns=True)
    sim.run_until(3 * hold)
    h = sim.result().history
    return dict(time=h['time'], heterogeneity=h['heterogeneity'],
                label=np.array([str(x) for x in h['label']]),
                mean_oxygen=h['mean_oxygen'], mean_ambient=h['mean_ambient'],
                area_fraction=h['area_fraction'], seed=seed, hold=hold,
                strain=strain, density=density, n_agents=n)


def step_response(processes=None, seeds=(1, 2, 3), side=128.0, density=0.75,
                  hold=2500.0, dt=0.1):
    """Dissolve and re-form a pattern by changing only the ambient oxygen.

    Run for two speed laws. NPR1_LIKE is monotone, so at 7% O2 every agent sits
    at v_min and the pattern loses its mobility as well as its driving force.
    HYPOXIC_BRANCH adds the extra high-speed branch at very low oxygen that the
    paper reports (Fig. 2b); it removes the driving force *and* keeps the agents
    fast. The contrast is the point of the experiment: only the second one
    actually disperses.

    `hold` is long (2500) because the relevant timescale is diffusive. At 7% O2
    the monotone strain has D_W ~ 0.009, so erasing a pattern of wavelength 26
    needs of order (26/2pi)^2 / D_W ~ 1900 model time units. A shorter hold does
    not measure dissolution, it measures the start of it.
    """
    strains = ('NPR1_LIKE', 'HYPOXIC_BRANCH')
    jobs = [(seed, strain, side, density, hold, dt)
            for strain in strains for seed in seeds]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        runs = list(pool.map(_step_run, jobs))

    fig = _figure((12.5, 4.6))
    axes = fig.subplots(1, 3)
    colors = {'NPR1_LIKE': TEAL, 'HYPOXIC_BRANCH': ORANGE}
    for ax, strain in zip(axes[:2], strains):
        for run in [r for r in runs if r['strain'] == strain]:
            ax.plot(run['time'], run['heterogeneity'], color=colors[strain], lw=1.6,
                    alpha=0.85, label=f'seed {run["seed"]}')
        ax.axvspan(hold, 2 * hold, color='#dfe6ec', zorder=0)
        ax.set(xlabel='Time [model time]', ylabel='Heterogeneity H', ylim=(0, None),
               title=f'{strain}')
        ax.legend(frameon=False, fontsize=8)
        _tidy(ax)
    reference = runs[0]
    axes[2].plot(reference['time'], reference['mean_ambient'], color='0.4', ls='--',
                 lw=1.6, label='imposed ambient')
    for strain in strains:
        run = [r for r in runs if r['strain'] == strain][0]
        axes[2].plot(run['time'], run['mean_oxygen'], color=colors[strain], lw=1.8,
                     label=f'mean O, {strain}')
    axes[2].axvspan(hold, 2 * hold, color='#dfe6ec', zorder=0)
    axes[2].set(xlabel='Time [model time]', ylabel='Oxygen / reference',
                title='Field follows the protocol')
    axes[2].legend(frameon=False, fontsize=8)
    _tidy(axes[2])
    fig.suptitle(f'O2 step 21% -> 7% -> 21% (hold {hold:.0f})  ·  W = {density:.2f}  ·  '
                 'only the hypoxic branch disperses',
                 fontsize=13, color=NAVY, weight='bold')
    fig.savefig(ASSETS / 'step_response.png', dpi=150)

    summary = []
    for run in runs:
        summary.append(dict(strain=run['strain'], seed=int(run['seed']),
                            dissolve_time=_transition(run, hold, 2 * hold, want=False),
                            reform_time=_transition(run, 2 * hold, 3 * hold, want=True),
                            H_before=float(run['heterogeneity'][
                                np.argmin(np.abs(run['time'] - hold))]),
                            H_at_end_of_hypoxia=float(run['heterogeneity'][
                                np.argmin(np.abs(run['time'] - 2 * hold))])))
    np.savez_compressed(
        ASSETS / 'step_response.npz',
        **{f'{r["strain"]}_{k}_{r["seed"]}': v for r in runs for k, v in r.items()
           if isinstance(v, np.ndarray)},
        summary=json.dumps(summary))
    for row in summary:
        print(f'  {row["strain"]:15s} seed {row["seed"]}: '
              f'H {row["H_before"]:.3f} -> {row["H_at_end_of_hypoxia"]:.3f}, '
              f'dissolve {row["dissolve_time"]:.0f}, reform {row["reform_time"]:.0f}')
    return summary


def _transition(run, start, end, *, want):
    """First time in [start, end) where 'patterned' becomes `want`, relative to start."""
    mask = (run['time'] >= start) & (run['time'] < end)
    times, labels = run['time'][mask], run['label'][mask]
    patterned = labels != 'uniform'
    hits = np.nonzero(patterned == want)[0]
    return float(times[hits[0]] - start) if hits.size else float('nan')


# -- 3. hysteresis -------------------------------------------------------

def _hysteresis_run(args):
    seed, direction = args
    # Density chosen so the ramp crosses the *lower* instability boundary, near
    # ambient 0.6, where the predicted growth rate rises quickly on the unstable
    # side (0.021 at ambient 1.0, growth time ~47 against a ramp of 2000). A
    # crossing of the upper boundary instead would have growth rates two orders
    # of magnitude smaller on both sides, and the experiment would measure only
    # how slow it is.
    side, density, dt = 128.0, 0.60, 0.1
    lo, hi, duration = 0.40, 1.00, 2000.0
    n = int(round(density * side * side))
    start, end = (lo, hi) if direction == 'up' else (hi, lo)
    env = envmod.ramp(start, end, duration)
    sim = Simulation('NPR1_LIKE', env, domain=(side, side, int(side), int(side)),
                     n_agents=n, dt=dt, seed=seed, record_every=100,
                     record_patterns=True)
    # Settle at the starting level before the ramp begins.
    sim.set_environment(envmod.uniform(start))
    sim.run_until(600.0)
    sim.set_environment(envmod.ramp(start, end, duration))
    base = sim.time
    sim.run_until(base + duration)
    h = sim.result().history
    keep = h['time'] >= base
    return dict(direction=direction, seed=seed,
                ambient=h['mean_ambient'][keep], heterogeneity=h['heterogeneity'][keep],
                label=np.array([str(x) for x in h['label']])[keep],
                density=density, n_agents=n)


def hysteresis(processes=None, seeds=(1, 2, 3)):
    jobs = [(s, d) for d in ('up', 'down') for s in seeds]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        runs = list(pool.map(_hysteresis_run, jobs))
    fig = _figure((9.5, 4.6))
    axes = fig.subplots(1, 2)
    for run in runs:
        color = ORANGE if run['direction'] == 'up' else TEAL
        axes[0].plot(run['ambient'], run['heterogeneity'], color=color, lw=1.6,
                     alpha=0.85,
                     label=f'{run["direction"]} ramp' if run['seed'] == seeds[0] else None)
        patterned = (run['label'] != 'uniform').astype(float)
        offset = 0.04 if run['direction'] == 'up' else -0.04
        axes[1].plot(run['ambient'], patterned + offset, color=color, lw=1.4, alpha=0.7,
                     label=f'{run["direction"]} ramp' if run['seed'] == seeds[0] else None)
    ambients = np.linspace(0.40, 1.0, 40)
    low, high = m1.boundary_curve('NPR1_LIKE', ambients)
    inside = (runs[0]['density'] >= low) & (runs[0]['density'] <= high)
    for ax in axes:
        if inside.any():
            ax.axvspan(ambients[inside].min(), ambients[inside].max(),
                       color='#dfe6ec', zorder=0, label='predicted unstable')
        ax.set_xlabel('Ambient oxygen / air')
        _tidy(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set(ylabel='Heterogeneity H', title='H along the ramp')
    axes[1].set(ylabel='patterned (1) / uniform (0)', title='Label along the ramp',
                yticks=[0, 1])
    fig.suptitle(f'Hysteresis: slow ambient ramp at fixed N '
                 f'(W = {runs[0]["density"]:.2f}, NPR1_LIKE)',
                 fontsize=13, color=NAVY, weight='bold')
    fig.savefig(ASSETS / 'hysteresis.png', dpi=150)
    np.savez_compressed(ASSETS / 'hysteresis.npz',
                        **{f'{r["direction"]}_{k}_{r["seed"]}': v
                           for r in runs for k, v in r.items()
                           if isinstance(v, np.ndarray)})
    print('hysteresis: saved', len(runs), 'runs')
    return runs


# -- 4. finite N ---------------------------------------------------------

def _finite_run(args):
    n_agents, seed, density, ambient, run_time = args
    side = float(np.sqrt(n_agents / density))
    cells = max(32, int(round(side)))
    sim = Simulation('NPR1_LIKE', envmod.uniform(ambient),
                     domain=(side, side, cells, cells), n_agents=int(n_agents),
                     dt=0.1, seed=seed, record_every=10**9, record_patterns=False)
    sim.run_until(run_time)
    rho = density_grid(sim.state.positions, sim.params, smoothing=DIAGNOSTIC_LENGTH)
    metrics = classify(rho, sim.params.dx, sim.params.dy)
    return dict(n_agents=int(n_agents), seed=int(seed), side=side,
                label=metrics['label'], heterogeneity=metrics['heterogeneity'],
                ratio=metrics['heterogeneity_ratio'])


def finite_n(processes=None, seeds=(1, 2, 3, 4, 5), run_time=1200.0):
    """Fixed density near the predicted boundary; scale the box and N together."""
    density, ambient = 0.50, 1.0
    # Top count capped at 20,000 rather than the 50,000 first tried. The box
    # scales with N at fixed density, so a 50,000-agent run is a 316 x 316 grid
    # and, once the pattern forms and the adaptive substepping tightens, takes
    # far longer than the rest of the sweep put together. 500 -> 20,000 is still
    # a 40x range in N and a 6x range in box side, which is what the question
    # needs; see VALIDATION.md.
    counts = (500, 1500, 5000, 12000, 20000)
    jobs = [(n, s, density, ambient, run_time) for n in counts for s in seeds]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        rows = list(pool.map(_finite_run, jobs))
    fractions, sides = [], []
    for n in counts:
        group = [r for r in rows if r['n_agents'] == n]
        fractions.append(np.mean([r['label'] != 'uniform' for r in group]))
        sides.append(group[0]['side'])
    prediction = m1.predict('NPR1_LIKE', envmod.uniform(ambient), density)
    fig = _figure((9.5, 4.4))
    axes = fig.subplots(1, 2)
    axes[0].semilogx(counts, fractions, 'o-', color=TEAL, lw=2, ms=8)
    axes[0].set(xlabel='Number of agents N', ylabel='Fraction of seeds that patterned',
                ylim=(-0.05, 1.05),
                title=f'Near the boundary (W = {density:.2f}, ambient {ambient:.2f})')
    for n, side in zip(counts, sides):
        group = [r for r in rows if r['n_agents'] == n]
        axes[1].semilogx([n] * len(group), [r['ratio'] for r in group], 'o',
                         color=ORANGE, alpha=0.7)
    axes[1].axhline(3.0, color=NAVY, ls='--', lw=1.4, label='classifier threshold')
    axes[1].set(xlabel='Number of agents N', ylabel='H / shot-noise H',
                title='Heterogeneity against its own noise floor')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        _tidy(ax)
    lam = prediction['wavelength'] if prediction['unstable'] else float('nan')
    fig.suptitle(f'Finite-N: density fixed, box and N scaled together '
                 f'(predicted wavelength {lam:.0f}, boxes '
                 f'{sides[0]:.0f}-{sides[-1]:.0f})',
                 fontsize=12.5, color=NAVY, weight='bold')
    fig.savefig(ASSETS / 'finite_n.png', dpi=150)
    np.savez_compressed(ASSETS / 'finite_n.npz',
                        counts=np.array(counts), fractions=np.array(fractions),
                        sides=np.array(sides),
                        rows=json.dumps(rows))
    print('finite N:', dict(zip(counts, np.round(fractions, 2))))
    return rows


# -- 5. spatial gradient -------------------------------------------------

def gradient(processes=None, seed=1, side=176.0, density=0.60, run_time=1500.0):
    """Cosine ambient across x: does density follow 1/V(O) and does the label
    track the local ambient level?"""
    cells = int(side)
    n = int(round(density * side * side))
    env = envmod.linear_gradient(0.35, 1.00, 'x', width=side, height=side,
                                 nx=cells, ny=cells)
    sim = Simulation('NPR1_LIKE', env, domain=(side, side, cells, cells),
                     n_agents=n, dt=0.1, seed=seed, record_every=10**9,
                     record_patterns=False)
    sim.run_until(run_time)
    p = sim.params
    rho = density_grid(sim.state.positions, p, smoothing=DIAGNOSTIC_LENGTH)
    profile = rho.mean(axis=0)
    oxygen = np.asarray(sim.state.oxygen).mean(axis=0)
    ambient = np.asarray(sim.ambient_field).mean(axis=0)
    x = (np.arange(cells) + 0.5) * p.dx
    # Stationary target of a speed-modulated random walk: rho ~ 1 / V(O).
    target = 1.0 / sim.strain.speed(oxygen)
    target = target / target.mean()
    measured = profile / profile.mean()
    correlation = float(np.corrcoef(measured, target)[0, 1])

    # Label in vertical slabs, each classified on its own local ambient.
    slabs, labels, local = 8, [], []
    for i in range(slabs):
        lo, hi = i * cells // slabs, (i + 1) * cells // slabs
        labels.append(classify(rho[:, lo:hi], p.dx, p.dy)['label'])
        local.append(float(ambient[lo:hi].mean()))

    fig = _figure((11.5, 4.6))
    axes = fig.subplots(1, 3, gridspec_kw={'width_ratios': [1.15, 1, 1]})
    im = axes[0].imshow(rho / rho.mean(), origin='lower', vmin=0, vmax=4, cmap='magma',
                        extent=(0, side, 0, side), interpolation='nearest')
    fig.colorbar(im, ax=axes[0], fraction=0.046, label='Density / mean')
    axes[0].set(xlabel='x [model length]', ylabel='y [model length]')
    axes[0].set_title('Density under a cosine ambient', loc='left',
                      color=NAVY, weight='bold', fontsize=10)
    axes[1].plot(x, measured, color=TEAL, lw=2, label='measured density / mean')
    axes[1].plot(x, target, color=ORANGE, lw=1.8, ls='--', label='1 / V(O), normalised')
    axes[1].set(xlabel='x [model length]', ylabel='Relative density',
                title=f'Accumulation where it is slow (r = {correlation:.2f})')
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(x, ambient, color='0.4', ls='--', lw=1.6, label='imposed O_am')
    axes[2].plot(x, oxygen, color=TEAL, lw=2, label='actual O')
    for i, (lab, amb) in enumerate(zip(labels, local)):
        lo = (i + 0.5) * side / slabs
        axes[2].plot([lo], [0.06], marker='s', ms=11, color=LABEL_COLORS[lab])
    axes[2].set(xlabel='x [model length]', ylabel='Oxygen / reference', ylim=(0, 1.05),
                title='Local ambient, local oxygen, local label')
    axes[2].legend(frameon=False, fontsize=8, loc='upper left')
    for ax in axes[1:]:
        _tidy(ax)
    fig.suptitle(f'Spatial gradient · NPR1_LIKE · W = {density:.2f} · t = {run_time:.0f}',
                 fontsize=13, color=NAVY, weight='bold')
    fig.savefig(ASSETS / 'gradient.png', dpi=150)
    np.savez_compressed(ASSETS / 'gradient.npz', x=x, measured=measured, target=target,
                        oxygen=oxygen, ambient=ambient, density_field=rho,
                        labels=np.array(labels), local_ambient=np.array(local),
                        correlation=correlation)
    print(f'gradient: correlation with 1/V(O) = {correlation:.3f}; '
          f'slab labels = {labels}')
    return correlation, labels


# -- 6. resolution convergence ------------------------------------------

def _convergence_run(args):
    label, seed, dt, cells, side, density, ambient, run_time = args
    n = int(round(density * side * side))
    sim = Simulation('NPR1_LIKE', envmod.uniform(ambient),
                     domain=(side, side, cells, cells), n_agents=n, dt=dt,
                     seed=seed, record_every=10 ** 9, record_patterns=False)
    sim.run_until(run_time)
    rho = density_grid(sim.state.positions, sim.params, smoothing=DIAGNOSTIC_LENGTH)
    m = classify(rho, sim.params.dx, sim.params.dy)
    return dict(variant=label, seed=seed, dt=dt, dx=side / cells,
                label=m['label'], length_scale=m['length_scale'],
                heterogeneity=m['heterogeneity'], area_fraction=m['area_fraction'])


def convergence(processes=None, seeds=(1, 2, 3)):
    """One solidly unstable point at three resolutions, three seeds each.

    Physical size, density, consumption footprint and diagnostic length are all
    held fixed; only dt and dx change. This is a robustness check, not a
    measured convergence rate.
    """
    side, density, ambient, run_time = 128.0, 0.70, 1.0, 1200.0
    variants = [('base (dt 0.1, dx 1.0)', 0.1, 128),
                ('half dt (0.05)', 0.05, 128),
                ('half dx (0.5)', 0.1, 256)]
    jobs = [(name, seed, dt, cells, side, density, ambient, run_time)
            for name, dt, cells in variants for seed in seeds]
    with ProcessPoolExecutor(max_workers=processes) as pool:
        rows = list(pool.map(_convergence_run, jobs))
    prediction = m1.predict('NPR1_LIKE', envmod.uniform(ambient), density)
    print(f'predicted: unstable={prediction["unstable"]} '
          f'wavelength={prediction["wavelength"]:.1f}')
    for name, _, _ in variants:
        group = [r for r in rows if r['variant'] == name]
        lengths = [r['length_scale'] for r in group]
        print(f'  {name:22s} labels {[r["label"] for r in group]}  '
              f'L = {np.mean(lengths):.1f} +/- {np.std(lengths, ddof=1):.1f}  '
              f'H = {np.mean([r["heterogeneity"] for r in group]):.3f}')
    np.savez_compressed(ASSETS / 'convergence.npz', rows=json.dumps(rows),
                        prediction=json.dumps({k: float(v) if isinstance(v, (int, float)) else v
                                               for k, v in prediction.items()}))
    return rows


EXPERIMENTS = dict(step=step_response, hysteresis=hysteresis,
                   finite=finite_n, gradient=gradient,
                   convergence=convergence)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*', choices=list(EXPERIMENTS) + [],
                        help='experiments to run')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--processes', type=int, default=None)
    args = parser.parse_args(argv)
    names = list(EXPERIMENTS) if args.all or not args.names else args.names
    ASSETS.mkdir(exist_ok=True)
    for name in names:
        started = time.perf_counter()
        print(f'--- {name} ---')
        EXPERIMENTS[name](processes=args.processes)
        print(f'    {time.perf_counter() - started:.0f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
