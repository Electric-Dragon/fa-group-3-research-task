"""Batch sweeps over (ambient oxygen) x (density) x (seed), and their bookkeeping.

One run per task, one process per core. Seeds are derived from a CRC of the
task's coordinates rather than from Python's `hash`, which is salted per
process and would make sweeps unrepeatable across runs.

Density is swept by changing the agent count at a fixed domain, so every run
shares the same box and the same grid and the measured length scales are
directly comparable. The box has to hold several wavelengths of whatever it is
measuring: `plan` computes the predicted wavelength for every cell of the grid
and warns for any cell where the domain is under six of them, because a pattern
that barely fits cannot be classified honestly.

Results go to `results/<strain>_<timestamp>/runs.csv` next to a `config.json`.
Reruns with `resume=True` skip rows already present in the CSV, so an
interrupted sweep continues where it stopped.
"""
from __future__ import annotations

import csv
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np

from .strains import get_strain
from . import environment as env_module
from .stability import predict

#: Columns written to runs.csv, in order.
FIELDS = ('strain', 'i_ambient', 'i_density', 'rep', 'seed', 'ambient', 'density',
          'n_agents', 'time', 'turns', 'label', 'heterogeneity',
          'shot_noise_heterogeneity', 'heterogeneity_ratio', 'contrast',
          'k_peak', 'length_scale', 'area_fraction', 'n_dense', 'n_dilute',
          'euler', 'dense_percolates', 'dilute_percolates',
          'label_mean', 'area_fraction_mean',
          'mean_oxygen', 'min_oxygen', 'mean_speed', 'polar_order',
          'predicted_unstable', 'predicted_wavelength', 'predicted_growth_rate',
          'predicted_O_eq', 'wavelengths_per_box', 'seconds')


@dataclass(frozen=True)
class SweepConfig:
    """Everything needed to reproduce a phase diagram.

    `run_time` is model time, not turns, so refining dt does not shorten the
    physical experiment. It should be at least about 10 / growth_rate for the
    slowest unstable cell, but growth_rate goes to zero at the boundary, so it
    is capped instead and cells near the boundary are honestly under-run --
    which is one of the reasons agreement is only claimed away from it.
    """
    strain: str = 'NPR1_LIKE'
    ambients: tuple = tuple(np.round(np.linspace(0.30, 1.00, 8), 4))
    densities: tuple = tuple(np.round(np.linspace(0.30, 1.20, 8), 4))
    seeds: int = 3
    width: float = 176.0
    height: float = 176.0
    nx: int = 176
    ny: int = 176
    dt: float = 0.1
    run_time: float = 600.0
    replenishment: float = 0.06
    oxygen_diffusion: float = 1.0
    init: str = 'uniform'
    record_every: int = 200
    seed_salt: int = 0
    save_fields: bool = False

    @property
    def area(self):
        return self.width * self.height

    def tasks(self):
        """Every (i_ambient, i_density, rep) cell with its derived seed."""
        for j, ambient in enumerate(self.ambients):
            for i, density in enumerate(self.densities):
                for rep in range(self.seeds):
                    yield dict(i_ambient=j, i_density=i, rep=rep,
                               ambient=float(ambient), density=float(density),
                               seed=self.seed_for(j, i, rep),
                               n_agents=max(1, int(round(float(density) * self.area))))

    def seed_for(self, j, i, rep):
        """Stable across processes, machines and Python versions, unlike hash()."""
        key = f'{self.strain}|{self.seed_salt}|{j}|{i}|{rep}'.encode()
        return int(zlib.crc32(key) & 0x7FFFFFFF)


def plan(config):
    """Predicted wavelength for every cell, plus the cells the box is too small for.

    Returns (predictions, warnings). Each warning names a cell where the domain
    holds fewer than six predicted wavelengths, which is the point below which
    a measured length scale stops meaning much.
    """
    strain = get_strain(config.strain)
    side = min(config.width, config.height)
    predictions, warnings = {}, []
    for j, ambient in enumerate(config.ambients):
        for i, density in enumerate(config.densities):
            p = predict(strain, (float(ambient), config.replenishment), float(density))
            per_box = side / p['wavelength'] if p['unstable'] else np.inf
            p['wavelengths_per_box'] = float(per_box)
            predictions[(j, i)] = p
            if p['unstable'] and per_box < 6:
                warnings.append(
                    f'ambient {ambient:.3g}, density {density:.3g}: predicted '
                    f'wavelength {p["wavelength"]:.0f} gives only {per_box:.1f} '
                    f'per box side ({side:.0f}); classification is unreliable')
            if p['unstable'] and not p['diffusive_limit_ok']:
                warnings.append(
                    f'ambient {ambient:.3g}, density {density:.3g}: wavelength is '
                    f'only {p["wavelength_over_persistence"]:.1f} persistence '
                    f'lengths; the drift-diffusion prediction is marginal here')
    return predictions, warnings


def run_one(task, config):
    """One simulation to completion, returned as a flat row. Picklable."""
    from .engine import Simulation
    started = time.perf_counter()
    strain = get_strain(config.strain)
    env = env_module.uniform(task['ambient'], config.replenishment,
                             config.oxygen_diffusion)
    sim = Simulation(strain, env,
                     domain=(config.width, config.height, config.nx, config.ny),
                     n_agents=task['n_agents'], dt=config.dt, seed=task['seed'],
                     init=config.init, record_every=config.record_every,
                     record_patterns=False)
    sim.run_until(config.run_time)
    final = sim.diagnostics(patterns=False)
    from .patterns import classify
    from .engine import density_grid, DIAGNOSTIC_LENGTH
    rho = density_grid(sim.state.positions, sim.params, smoothing=DIAGNOSTIC_LENGTH)
    metrics = classify(rho, sim.params.dx, sim.params.dy, smoothing=DIAGNOSTIC_LENGTH)
    # Also record the label the mean threshold would have given, so the
    # threshold choice can be audited from the CSV without rerunning anything.
    by_mean = classify(rho, sim.params.dx, sim.params.dy,
                       smoothing=DIAGNOSTIC_LENGTH, threshold='mean')
    p = predict(strain, (task['ambient'], config.replenishment), task['density'])
    side = min(config.width, config.height)
    row = dict(
        strain=config.strain, i_ambient=task['i_ambient'], i_density=task['i_density'],
        rep=task['rep'], seed=task['seed'], ambient=task['ambient'],
        density=task['density'], n_agents=task['n_agents'],
        time=sim.time, turns=sim.turn,
        mean_oxygen=final['mean_oxygen'], min_oxygen=final['min_oxygen'],
        mean_speed=final['mean_speed'], polar_order=final['polar_order'],
        predicted_unstable=int(p['unstable']),
        predicted_wavelength=p['wavelength'],
        predicted_growth_rate=p['growth_rate'], predicted_O_eq=p['O_eq'],
        wavelengths_per_box=(side / p['wavelength']) if p['unstable'] else float('inf'),
        seconds=time.perf_counter() - started)
    for key in ('label', 'heterogeneity', 'shot_noise_heterogeneity',
                'heterogeneity_ratio', 'contrast', 'k_peak', 'length_scale',
                'area_fraction', 'n_dense', 'n_dilute', 'euler'):
        row[key] = metrics[key]
    row['dense_percolates'] = int(metrics['dense_percolates'])
    row['dilute_percolates'] = int(metrics['dilute_percolates'])
    row['label_mean'] = by_mean['label']
    row['area_fraction_mean'] = by_mean['area_fraction']
    fields = None
    if config.save_fields:
        fields = dict(oxygen=np.asarray(sim.state.oxygen), density=rho)
    return row, fields


def _worker(payload):
    task, config = payload
    return run_one(task, config)


def sweep_phase_diagram(config=None, out_dir=None, *, processes=None, resume=True,
                        progress=True):
    """Run the whole grid and write results/<strain>_<timestamp>/runs.csv.

    Returns (out_dir, rows). Rows already in the CSV are skipped when
    `resume=True`, matched on (i_ambient, i_density, rep).
    """
    config = config or SweepConfig()
    out_dir = Path(out_dir) if out_dir else Path('results') / (
        f'{config.strain}_{time.strftime("%Y%m%d-%H%M%S")}')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'runs.csv'

    predictions, warnings = plan(config)
    (out_dir / 'config.json').write_text(json.dumps(
        dict(config=asdict(config), warnings=warnings), indent=2, default=float))
    if progress and warnings:
        print(f'{len(warnings)} cells flagged by plan():')
        for line in warnings[:8]:
            print('  !', line)
        if len(warnings) > 8:
            print(f'  ... and {len(warnings) - 8} more (see config.json)')

    done = set()
    existing = []
    if resume and csv_path.exists():
        existing = load_runs(csv_path)
        done = {(int(r['i_ambient']), int(r['i_density']), int(r['rep']))
                for r in existing}
        if progress and done:
            print(f'resuming: {len(done)} rows already present')
    tasks = [t for t in config.tasks()
             if (t['i_ambient'], t['i_density'], t['rep']) not in done]
    if progress:
        print(f'{len(tasks)} runs to go on {processes or os.cpu_count()} processes')

    write_header = not csv_path.exists()
    rows, fields_out = list(existing), {}
    if tasks:
        with open(csv_path, 'a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=processes) as pool:
                payloads = [(t, config) for t in tasks]
                for n, (row, fields) in enumerate(pool.map(_worker, payloads), 1):
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
                    if fields is not None:
                        key = f'{row["i_ambient"]}_{row["i_density"]}_{row["rep"]}'
                        fields_out[f'oxygen_{key}'] = fields['oxygen']
                        fields_out[f'density_{key}'] = fields['density']
                    if progress and (n % 10 == 0 or n == len(tasks)):
                        rate = n / (time.perf_counter() - started)
                        left = (len(tasks) - n) / max(rate, 1e-9)
                        print(f'  {n}/{len(tasks)} runs  '
                              f'{rate * 60:.1f}/min  ~{left / 60:.1f} min left')
    if fields_out:
        np.savez_compressed(out_dir / 'final_fields.npz', **fields_out)
    return out_dir, rows


def load_runs(path):
    """Read runs.csv back with numeric columns as floats and label as a string."""
    path = Path(path)
    if path.is_dir():
        path = path / 'runs.csv'
    rows = []
    with open(path, newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if key in ('strain', 'label', 'label_mean'):
                    row[key] = value
                elif value in ('', None):
                    row[key] = float('nan')
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def majority_labels(rows, config=None):
    """Collapse seeds into a per-cell majority label and an agreement fraction.

    Returns a dict of 2D arrays indexed [i_density, i_ambient]: `label` (object),
    `agreement` (majority share of seeds), `predicted` (bool), plus the mean of
    the numeric metrics. Ties go to the label that is rarer overall, so a cell
    split between 'uniform' and 'dots' is reported as 'dots' rather than being
    silently absorbed into the background -- and `agreement` says it was a tie.
    """
    if not rows:
        raise ValueError('no rows to summarise')
    ja = int(max(r['i_ambient'] for r in rows)) + 1
    id_ = int(max(r['i_density'] for r in rows)) + 1
    shape = (id_, ja)
    out = dict(label=np.full(shape, '', dtype=object),
               agreement=np.zeros(shape),
               predicted=np.zeros(shape, dtype=bool),
               ambient=np.full(shape, np.nan), density=np.full(shape, np.nan),
               n_seeds=np.zeros(shape, dtype=int))
    numeric = ('heterogeneity', 'heterogeneity_ratio', 'length_scale',
               'area_fraction', 'predicted_wavelength', 'predicted_growth_rate',
               'wavelengths_per_box', 'k_peak')
    for key in numeric:
        out[key] = np.full(shape, np.nan)
    overall = {}
    for r in rows:
        overall[r['label']] = overall.get(r['label'], 0) + 1
    buckets = {}
    for r in rows:
        buckets.setdefault((int(r['i_density']), int(r['i_ambient'])), []).append(r)
    for (i, j), group in buckets.items():
        counts = {}
        for r in group:
            counts[r['label']] = counts.get(r['label'], 0) + 1
        best = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == best]
        label = min(tied, key=lambda lab: overall.get(lab, 0))
        out['label'][i, j] = label
        out['agreement'][i, j] = best / len(group)
        out['n_seeds'][i, j] = len(group)
        out['predicted'][i, j] = bool(group[0]['predicted_unstable'])
        out['ambient'][i, j] = group[0]['ambient']
        out['density'][i, j] = group[0]['density']
        for key in numeric:
            values = [r[key] for r in group if np.isfinite(r.get(key, np.nan))]
            if values:
                out[key][i, j] = float(np.mean(values))
    return out


def agreement_with_prediction(rows, *, boundary_margin=1):
    """How often the simulated label agrees with the linear-stability prediction.

    'Patterned' means any label other than 'uniform'. Cells adjacent to the
    predicted boundary are excluded, because that is exactly where finite-N
    noise, finite run time and a finite box are all expected to disagree with an
    infinite, deterministic, linear prediction -- see the acceptance criterion.

    Returns a dict with the counts, the fraction, and the excluded-cell mask.
    """
    summary = majority_labels(rows)
    predicted = summary['predicted']
    observed = summary['label'] != 'uniform'
    near = _dilate(_boundary_cells(predicted), boundary_margin)
    interior = ~near & (summary['n_seeds'] > 0)
    agree = (predicted == observed) & interior
    total = int(interior.sum())
    return dict(agree=int(agree.sum()), total=total,
                fraction=float(agree.sum() / total) if total else float('nan'),
                interior=interior, predicted=predicted, observed=observed,
                summary=summary)


def _boundary_cells(predicted):
    """Cells whose predicted stability differs from an in-grid neighbour.

    Neighbours are taken without wrapping: the (ambient, density) grid is a
    parameter plane, not a torus, so its opposite edges are not adjacent.
    """
    mask = np.zeros_like(predicted, dtype=bool)
    mask[:-1] |= predicted[:-1] != predicted[1:]
    mask[1:] |= predicted[1:] != predicted[:-1]
    mask[:, :-1] |= predicted[:, :-1] != predicted[:, 1:]
    mask[:, 1:] |= predicted[:, 1:] != predicted[:, :-1]
    return mask


def _dilate(mask, radius):
    """Grow a boolean mask by `radius` cells in both directions, non-periodically."""
    out = mask.copy()
    for _ in range(int(radius)):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out
