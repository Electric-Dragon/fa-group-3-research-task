"""Model 1's simulation engine: Model 0's numerics, with a live environment.

Three things change relative to Model 0's `Simulation`.

1. **The environment may be a field and may move.** Ambient oxygen and the
   replenishment rate can be [ny, nx] arrays, and ambient can be a function of
   time. They enter exactly where Model 0's scalars did, in the exact
   replenishment update `c = A + (c - A) exp(-f h)`, which is elementwise, so a
   uniform array reproduces the scalar path bit for bit.
2. **The population may change at runtime.** `add_agents` and `remove_agents`
   draw from the engine RNG, so a run with a scripted population history is
   still reproducible from its seed.
3. **Deposition is faster.** Model 0's `np.add.at` is replaced by `np.bincount`
   on the flattened cell index. The cloud-in-cell weights are computed once per
   substep and reused for both deposition and field sampling, which Model 0
   computed twice.

Everything else is deliberately unchanged: the adaptive substep bounds, the
positivity-preserving implicit consumption solve, periodic cell-centred grids,
arrays indexed [y, x] with positions (x, y), and the rule that drawing never
consumes RNG state.

Ambient is evaluated **once per outer turn**, at the time the turn begins, not
once per substep. Substeps exist for numerical stability, not to resolve the
protocol, and re-evaluating a spatial field several times per turn was the
single largest cost in profiling. A schedule therefore resolves changes to
within dt; `Simulation` refuses a schedule whose steps are finer than that.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
import numpy as np
from scipy.ndimage import gaussian_filter

from ..model0.model import Parameters, State, _weights
from .strains import Strain, get_strain
from .environment import Environment, cell_centers, uniform
from .stability import equilibrium_oxygen

INIT_MODES = ('uniform', 'seeded', 'droplet')
DIAGNOSTIC_LENGTH = 2.0  # same fixed observation length Model 0 reports H at


# -- fields --------------------------------------------------------------

def density_grid(positions, params, smoothing=None, *, weights=None):
    """Agents per area, integrating to the agent count over the domain.

    Same convention and result as Model 0's `density_grid`, but deposited with
    `np.bincount` instead of `np.add.at`. Floating-point summation order
    differs, so values agree to rounding, not to the last bit.
    """
    ny, nx = params.ny, params.nx
    weights = _weights(positions, params) if weights is None else weights
    flat = np.zeros(ny * nx)
    for y, x, w in weights:
        flat += np.bincount(y * nx + x, weights=w, minlength=ny * nx)
    rho = flat.reshape(ny, nx) / (params.dx * params.dy)
    sigma = params.kernel_width if smoothing is None else float(smoothing)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('smoothing must be finite and nonnegative')
    if sigma:
        rho = gaussian_filter(rho, (sigma / params.dy, sigma / params.dx), mode='wrap')
    return rho


def sample_field(field, positions, params, *, weights=None):
    """Bilinear periodic interpolation, optionally reusing cloud-in-cell weights."""
    weights = _weights(positions, params) if weights is None else weights
    return sum(w * field[y, x] for y, x, w in weights)


def oxygen_step(oxygen, rho, p, h, ambient, replenishment):
    """One positivity-preserving split step, first order in h.

    Identical to Model 0's `_oxygen_step` except that `ambient` and
    `replenishment` are supplied by the caller and may be arrays. Every
    operation is elementwise, so passing a constant array gives bit-for-bit the
    scalar result. Diffusion is the explicit five-point Laplacian under its
    stability bound; replenishment is exact; the saturating sink is a local
    backward-Euler quadratic solve, so oxygen cannot be overspent and never
    needs clipping.
    """
    lap = ((np.roll(oxygen, 1, 1) + np.roll(oxygen, -1, 1) - 2 * oxygen) / p.dx**2
           + (np.roll(oxygen, 1, 0) + np.roll(oxygen, -1, 0) - 2 * oxygen) / p.dy**2)
    c = oxygen + h * p.oxygen_diffusion * lap
    c = ambient + (c - ambient) * np.exp(-replenishment * h)
    if p.consumption:
        k = p.oxygen_half_saturation
        b = k + h * p.consumption * rho - c
        root = np.sqrt(b * b + 4 * k * c)
        out = np.empty_like(c)
        positive = b >= 0
        # Algebraically equivalent branches; each avoids cancellation on its side.
        out[positive] = 2 * k * c[positive] / (root[positive] + b[positive])
        out[~positive] = (root[~positive] - b[~positive]) / 2
        c = out
    oxygen[:] = c


def _snapshot(array):
    result = np.asarray(array).copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class Result:
    """A run's endpoint, its sampled history, and everything needed to repeat it."""
    state: State
    history: dict
    params: Parameters
    strain: Strain
    environment: Environment
    seed: int
    init: str
    parameter_log: tuple
    patterns: dict = field(default_factory=dict)

    @property
    def positions(self):
        return self.state.positions

    @property
    def headings(self):
        return self.state.headings

    @property
    def oxygen(self):
        return self.state.oxygen

    @property
    def density(self):
        return self.params.n_agents / (self.params.width * self.params.height)


class Simulation:
    """Frozen agents, live environment, live population.

    Parameters
    ----------
    strain : `Strain` or preset name. Immutable for the life of the run.
    env : `Environment`; defaults to uniform air. Swap with `set_environment`.
    domain : (width, height, nx, ny).
    n_agents : starting population; changes with `add_agents`/`remove_agents`.
    dt : outer turn length. Substeps subdivide it as stability requires.
    seed : RNG seed. Everything stochastic, including population changes, is
        drawn from this one generator.
    init : 'uniform', 'seeded' (a fifth of the agents in a central Gaussian) or
        'droplet' (all agents in a central disc).
    record_every : outer turns between history samples.
    record_patterns : also run the `patterns` classifier at each history sample.
        It costs an FFT and a connected-component labelling, so it is worth
        turning off for large sweeps that only need the endpoint.
    """

    def __init__(self, strain='NPR1_LIKE', env=None, *, domain=(64.0, 48.0, 64, 48),
                 n_agents=1800, dt=0.1, seed=1, init='uniform', record_every=20,
                 record_patterns=True):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError('record_every must be a positive integer')
        if init not in INIT_MODES:
            raise ValueError(f'init must be one of {INIT_MODES}')
        self.strain = get_strain(strain)
        self.environment = env if env is not None else uniform()
        if not isinstance(self.environment, Environment):
            raise ValueError('env must be an Environment')
        width, height, nx, ny = domain
        self.params = self.strain.to_parameters(
            width=float(width), height=float(height), nx=int(nx), ny=int(ny),
            n_agents=int(n_agents), dt=float(dt),
            oxygen_diffusion=self.environment.oxygen_diffusion,
            replenishment=float(np.mean(np.asarray(self.environment.replenishment))),
            ambient=_representative_ambient(self.environment,
                                            float(width), float(height),
                                            int(nx), int(ny)))
        self.seed = int(seed)
        self.init = init
        self.record_every = record_every
        self.record_patterns = bool(record_patterns)
        self._grid = cell_centers(self.params.width, self.params.height,
                                  self.params.nx, self.params.ny)
        self.reset()

    # -- setup -----------------------------------------------------------
    def reset(self, *, seed=None):
        """Restart from initial conditions with the current strain and environment."""
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError('seed must be a nonnegative integer')
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        p = self.params
        self._positions = self._draw_positions(p.n_agents, self.init)
        self._headings = self.rng.uniform(0, 2 * np.pi, p.n_agents)
        self.time, self.turn = 0.0, 0
        self._refresh_environment()
        density = p.n_agents / (p.width * p.height)
        self._oxygen = np.broadcast_to(
            equilibrium_oxygen(self.strain, self._ambient, self._replenishment, density),
            (p.ny, p.nx)).astype(float).copy()
        self._records = []
        self.parameter_log = [self._log_entry('reset')]
        self._record()
        return self.state

    def _draw_positions(self, n, mode, center=None, spread=None):
        p = self.params
        center = (p.width / 2, p.height / 2) if center is None else tuple(center)
        if mode == 'uniform':
            return self.rng.uniform(size=(n, 2)) * (p.width, p.height)
        if mode == 'point':
            sigma = 0.02 * min(p.width, p.height) if spread is None else float(spread)
            return (self.rng.normal(size=(n, 2)) * sigma + center) % (p.width, p.height)
        if mode == 'seeded':
            positions = self.rng.uniform(size=(n, 2)) * (p.width, p.height)
            m = n // 5
            positions[:m] = (self.rng.normal(size=(m, 2)) * (0.08 * p.width, 0.08 * p.height)
                             + center) % (p.width, p.height)
            return positions
        if mode == 'droplet':
            radius = 0.18 * min(p.width, p.height) if spread is None else float(spread)
            r = radius * np.sqrt(self.rng.uniform(size=n))
            theta = self.rng.uniform(0, 2 * np.pi, n)
            return (np.column_stack((center[0] + r * np.cos(theta),
                                     center[1] + r * np.sin(theta)))
                    % (p.width, p.height))
        raise ValueError(f'unknown placement mode {mode!r}')

    def _refresh_environment(self):
        """Resolve the environment onto the grid for the turn that is starting."""
        self._ambient = self.environment.ambient_field(*self._grid, self.time)
        self._replenishment = self.environment.replenishment_field(
            (self.params.ny, self.params.nx))

    def _log_entry(self, event, **extra):
        return dict(event=event, time=self.time, turn=self.turn,
                    n_agents=self.params.n_agents,
                    environment=self.environment.label,
                    params=asdict(self.params), **extra)

    # -- user controls ---------------------------------------------------
    def set_environment(self, env):
        """Swap the environment immediately, mid-run. Logged with the turn and time."""
        if not isinstance(env, Environment):
            raise ValueError('env must be an Environment')
        self.environment = env
        self.params = replace(
            self.params, oxygen_diffusion=env.oxygen_diffusion,
            replenishment=float(np.mean(np.asarray(env.replenishment))),
            ambient=env.representative(self._grid, self.time)[0])
        self._refresh_environment()
        self.parameter_log.append(self._log_entry('set_environment'))
        return self.environment

    def add_agents(self, n, mode='uniform', center=None, spread=None):
        """Introduce n agents with fresh random headings. Logged.

        mode 'uniform' spreads them over the domain; 'point' drops them in a
        Gaussian blob of standard deviation `spread` at `center`. Positions and
        headings come from the engine RNG, so the run stays reproducible.
        """
        n = _positive_count(n, 'n')
        if n == 0:
            return self.params.n_agents
        placement = 'uniform' if mode == 'uniform' else mode
        new_positions = self._draw_positions(n, placement, center, spread)
        new_headings = self.rng.uniform(0, 2 * np.pi, n)
        self._positions = np.vstack((self._positions, new_positions))
        self._headings = np.concatenate((self._headings, new_headings))
        self.params = replace(self.params, n_agents=self.params.n_agents + n)
        self.parameter_log.append(self._log_entry('add_agents', added=n, mode=mode))
        return self.params.n_agents

    def remove_agents(self, n):
        """Remove n uniformly chosen agents. Logged. At least one agent remains."""
        n = _positive_count(n, 'n')
        current = self.params.n_agents
        n = min(n, current - 1)
        if n <= 0:
            return current
        keep = np.sort(self.rng.choice(current, size=current - n, replace=False))
        self._positions = self._positions[keep].copy()
        self._headings = self._headings[keep].copy()
        self.params = replace(self.params, n_agents=current - n)
        self.parameter_log.append(self._log_entry('remove_agents', removed=n))
        return self.params.n_agents

    def set_population(self, n, mode='uniform', center=None, spread=None):
        """Add or remove agents so the population becomes exactly n."""
        n = _positive_count(n, 'n')
        if n < 1:
            raise ValueError('population must stay at least 1')
        delta = n - self.params.n_agents
        if delta > 0:
            return self.add_agents(delta, mode, center, spread)
        if delta < 0:
            return self.remove_agents(-delta)
        return n

    # -- state and history -----------------------------------------------
    @property
    def state(self):
        return State(_snapshot(self._positions), _snapshot(self._headings),
                     _snapshot(self._oxygen), self.time, self.turn)

    @property
    def density(self):
        return self.params.n_agents / (self.params.width * self.params.height)

    @property
    def ambient_field(self):
        """The ambient map currently in force; a scalar stays a scalar."""
        return self._ambient

    @property
    def replenishment_field(self):
        return self._replenishment

    def diagnostics(self, *, patterns=None):
        """Model 0's diagnostics, evaluated with this strain's speed law.

        The optional pattern metrics come from `patterns.classify`, computed on
        the density field smoothed at the same fixed diagnostic length, so H and
        the classifier always describe the same field.
        """
        p = self.params
        rho = density_grid(self._positions, p, smoothing=DIAGNOSTIC_LENGTH)
        relative = rho / rho.mean()
        sensed = sample_field(self._oxygen, self._positions, p)
        record = dict(
            turn=self.turn, time=self.time, n_agents=p.n_agents,
            density=self.density,
            heterogeneity=float(np.mean((relative - 1) ** 2)),
            mean_oxygen=float(self._oxygen.mean()),
            min_oxygen=float(self._oxygen.min()),
            mean_speed=float(self.strain.speed(sensed).mean()),
            polar_order=float(np.abs(np.mean(np.exp(1j * self._headings)))),
            mean_ambient=float(np.mean(self._ambient)))
        want = self.record_patterns if patterns is None else patterns
        if want:
            from .patterns import classify
            metrics = classify(rho, p.dx, p.dy)
            record['label'] = metrics['label']
            for key in ('k_peak', 'length_scale', 'area_fraction',
                        'n_dense', 'n_dilute', 'euler', 'contrast',
                        'shot_noise_heterogeneity', 'heterogeneity_ratio'):
                record[key] = metrics[key]
        return record

    def _record(self):
        self._records.append(self.diagnostics())

    # -- time stepping ---------------------------------------------------
    def step(self, n=1):
        """Advance n outer turns of length dt.

        Chunking does not change the trajectory: the substep schedule depends
        only on the state, and the environment is resolved at the start of each
        outer turn regardless of how the turns were grouped into calls.
        """
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError('n must be a nonnegative integer')
        p = self.params
        for _ in range(n):
            if self.environment.time_varying:
                self._refresh_environment()
            p = self.params
            f_max = float(np.max(self._replenishment))
            remaining = p.dt
            while remaining > p.dt * 1e-12:
                weights = _weights(self._positions, p)
                rho = density_grid(self._positions, p, weights=weights)
                rate = max(self.strain.max_speed / (0.3 * min(p.dx, p.dy)),
                           p.rotational_diffusion / 0.1,
                           2.5 * p.oxygen_diffusion * (p.dx**-2 + p.dy**-2),
                           f_max / 0.2,
                           p.consumption * rho.max() / (0.2 * p.oxygen_half_saturation))
                h = min(remaining, 1 / rate)
                sensed = sample_field(self._oxygen, self._positions, p, weights=weights)
                self._headings[:] = (self._headings
                                     + np.sqrt(2 * p.rotational_diffusion * h)
                                     * self.rng.normal(size=p.n_agents)) % (2 * np.pi)
                directions = np.column_stack((np.cos(self._headings), np.sin(self._headings)))
                self._positions[:] = (self._positions
                                      + h * self.strain.speed(sensed)[:, None] * directions
                                      ) % (p.width, p.height)
                oxygen_step(self._oxygen, rho, p, h, self._ambient, self._replenishment)
                remaining -= h
            self.turn += 1
            self.time += p.dt
            if self.turn % self.record_every == 0:
                self._record()
        return self.state

    def run_until(self, time):
        """Advance to at least the given model time."""
        turns = int(np.ceil((float(time) - self.time) / self.params.dt))
        return self.step(max(turns, 0))

    def result(self, *, patterns=None):
        """Snapshot, sampled history and the full parameter/environment log.

        The endpoint is always included even if it is not on the recording grid.
        History keys that only some samples carry (the pattern label, when
        `record_patterns` was toggled mid-run) are dropped rather than padded.
        """
        records = list(self._records)
        if not records or records[-1]['turn'] != self.turn:
            records.append(self.diagnostics(patterns=patterns))
        shared = set(records[0])
        for r in records:
            shared &= set(r)
        history = {}
        for key in shared:
            values = [r[key] for r in records]
            history[key] = np.array(values, dtype=object if key == 'label' else float)
        final = records[-1]
        patterns_out = {k: final[k] for k in final if k in (
            'label', 'k_peak', 'length_scale', 'area_fraction', 'n_dense',
            'n_dilute', 'euler', 'contrast', 'shot_noise_heterogeneity',
            'heterogeneity_ratio')}
        return Result(self.state, history, self.params, self.strain,
                      self.environment, self.seed, self.init,
                      tuple(copy.deepcopy(self.parameter_log)), patterns_out)


def _representative_ambient(env, width, height, nx, ny):
    """Mean ambient over the grid, used only to fill the Model 0 `Parameters`.

    The engine's own oxygen step always uses the resolved field, never this.
    """
    return env.representative(cell_centers(width, height, nx, ny), 0.0)[0]


def _positive_count(n, name):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(n)


def run(strain='NPR1_LIKE', env=None, *, turns=3000, **kwargs):
    """Finite headless run through exactly the interactive engine."""
    simulation = Simulation(strain, env, **kwargs)
    simulation.step(turns)
    return simulation.result()
