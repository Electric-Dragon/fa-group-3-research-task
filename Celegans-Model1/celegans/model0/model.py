"""Model 0: overlapping active points coupled to an oxygen field.

All quantities use arbitrary model units, not calibrated biological units.
Grid entries are cell-centered. Arrays are indexed [y, x]; positions are (x, y).
There is no pair force, alignment torque, or oxygen-gradient steering.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit


@dataclass(frozen=True)
class Parameters:
    """Physical parameters, domain, and numerical resolution.

    Oxygen is normalized to a fixed reference concentration (not re-normalized
    when ambient changes). q has units oxygen * area / (agent * time).
    kernel_width is a physical consumption footprint, not a worm body radius.
    """
    width: float = 64.0
    height: float = 48.0
    nx: int = 64
    ny: int = 48
    n_agents: int = 1800
    dt: float = 0.1
    v_min: float = 0.06
    v_max: float = 2.0
    oxygen_midpoint: float = 0.55
    response_width: float = 0.045
    rotational_diffusion: float = 0.25
    oxygen_diffusion: float = 1.0
    replenishment: float = 0.06
    ambient: float = 1.0
    consumption: float = 0.06
    oxygen_half_saturation: float = 0.10
    kernel_width: float = 1.2
    speed_response: bool = True

    def __post_init__(self):
        for name in ('nx', 'ny', 'n_agents'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
            if value < (4 if name != 'n_agents' else 1):
                raise ValueError(f'{name} is too small')
        positive = ('width', 'height', 'dt', 'v_min', 'v_max',
                    'response_width', 'oxygen_half_saturation', 'kernel_width')
        nonnegative = ('rotational_diffusion', 'oxygen_diffusion',
                       'replenishment', 'consumption')
        for name in positive + nonnegative + ('ambient', 'oxygen_midpoint'):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
            if name in positive and value <= 0:
                raise ValueError(f'{name} must be positive')
            if name in nonnegative and value < 0:
                raise ValueError(f'{name} must be nonnegative')
        if self.v_min > self.v_max:
            raise ValueError('v_min must not exceed v_max')
        if not 0 <= self.ambient <= 1 or not 0 <= self.oxygen_midpoint <= 1:
            raise ValueError('ambient and oxygen_midpoint must lie in [0, 1]')
        if not isinstance(self.speed_response, (bool, np.bool_)):
            raise ValueError('speed_response must be boolean')

    @property
    def dx(self):
        return self.width / self.nx

    @property
    def dy(self):
        return self.height / self.ny


def speed(oxygen, params: Parameters):
    """Positive monotone toy law; disabling response gives constant v_max."""
    oxygen = np.asarray(oxygen, dtype=float)
    if not params.speed_response:
        return np.full_like(oxygen, params.v_max)
    return params.v_min + (params.v_max - params.v_min) * expit(
        (oxygen - params.oxygen_midpoint) / params.response_width)


def _weights(positions, params):
    """Periodic cloud-in-cell indices and weights at cell centers."""
    xy = positions / (params.dx, params.dy) - 0.5
    base = np.floor(xy).astype(np.int64)
    fraction = xy - base
    x, y = base[:, 0], base[:, 1]
    fx, fy = fraction[:, 0], fraction[:, 1]
    return [((y + j) % params.ny, (x + i) % params.nx,
             (fx if i else 1 - fx) * (fy if j else 1 - fy))
            for j in (0, 1) for i in (0, 1)]


def sample_field(field, positions, params):
    """Bilinear, periodic interpolation; same grid convention as deposition."""
    return sum(w * field[y, x] for y, x, w in _weights(positions, params))


def density_grid(positions, params: Parameters, smoothing=None):
    """Agents per area; integral over the domain equals the agent count.

    Deposit by cloud-in-cell, then convolve with a normalized periodic Gaussian.
    smoothing is the Gaussian standard deviation in model length units.
    """
    rho = np.zeros((params.ny, params.nx), dtype=float)
    for y, x, weight in _weights(positions, params):
        np.add.at(rho, (y, x), weight)
    rho /= params.dx * params.dy
    sigma = params.kernel_width if smoothing is None else float(smoothing)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError('smoothing must be finite and nonnegative')
    if sigma:
        rho = gaussian_filter(rho, (sigma / params.dy, sigma / params.dx), mode='wrap')
    return rho


def _oxygen_step(oxygen, rho, p, h):
    """One positivity-preserving split step, first order in h.

    Diffusion uses a five-point Laplacian under its explicit stability bound.
    Replenishment is exact. Saturating consumption uses a local backward-Euler
    quadratic solve, so oxygen cannot be overspent. No concentration clipping
    hides an unstable solve. The caller chooses a conservative h.
    """
    lap = ((np.roll(oxygen, 1, 1) + np.roll(oxygen, -1, 1) - 2 * oxygen) / p.dx**2
           + (np.roll(oxygen, 1, 0) + np.roll(oxygen, -1, 0) - 2 * oxygen) / p.dy**2)
    c = oxygen + h * p.oxygen_diffusion * lap
    c = p.ambient + (c - p.ambient) * np.exp(-p.replenishment * h)
    if p.consumption:
        k = p.oxygen_half_saturation
        b = k + h * p.consumption * rho - c
        root = np.sqrt(b*b + 4*k*c)
        # Choose the algebraically equivalent branch that avoids cancellation.
        out = np.empty_like(c)
        positive = b >= 0
        out[positive] = 2*k*c[positive] / (root[positive] + b[positive])
        out[~positive] = (root[~positive] - b[~positive]) / 2
        c = out
    oxygen[:] = c


def _snapshot(array):
    result = array.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class State:
    """Independent read-only snapshot; later steps do not change these arrays."""
    positions: np.ndarray
    headings: np.ndarray
    oxygen: np.ndarray
    time: float
    turn: int


@dataclass(frozen=True)
class Result:
    state: State
    history: dict
    params: Parameters
    scenario: str
    seed: int
    parameter_log: tuple

    @property
    def positions(self):
        return self.state.positions

    @property
    def headings(self):
        return self.state.headings

    @property
    def oxygen(self):
        return self.state.oxygen


class Simulation:
    """Shared engine for interactive and headless experiments.

    An outer turn lasts params.dt; internal substeps enforce diffusion stability,
    limit displacement to 0.3 grid spacings and Dr*h to 0.1, and resolve the
    local consumption time conservatively. These bounds aid accuracy but do
    not replace dt/grid convergence checks. Drawing never consumes RNG state.
    """
    def __init__(self, scenario='feedback', params=None, *, seed=1, record_every=20):
        from .scenarios import get_scenario
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        if isinstance(record_every, bool) or not isinstance(record_every, int) or record_every < 1:
            raise ValueError('record_every must be a positive integer')
        self.scenario = get_scenario(scenario)
        self.params = params or Parameters()
        self.seed = int(seed)
        self.record_every = record_every
        self.reset()

    def reset(self, *, seed=None):
        """Restart with current parameters and the same seed unless supplied."""
        from .scenarios import initial_conditions
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError('seed must be a nonnegative integer')
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._positions, self._headings, self._oxygen = initial_conditions(
            self.scenario, self.params, self.rng)
        self.time, self.turn = 0.0, 0
        self._records = []
        self.parameter_log = [{'time': 0.0, 'turn': 0, 'params': asdict(self.params)}]
        self._record()
        return self.state

    def set_params(self, **changes):
        """Change physical parameters in place; reset for a clean comparison.

        Domain, population and dt changes require a new Simulation. Fixed-field
        scenarios ignore consumption, diffusion, and replenishment by design.
        """
        fixed = {'width', 'height', 'nx', 'ny', 'n_agents', 'dt'}
        if fixed.intersection(changes):
            raise ValueError('Create a new Simulation to change domain, population, grid or dt')
        self.params = replace(self.params, **changes)
        self.parameter_log.append({'time': self.time, 'turn': self.turn, 'params': asdict(self.params)})

    @property
    def state(self):
        return State(_snapshot(self._positions), _snapshot(self._headings),
                     _snapshot(self._oxygen), self.time, self.turn)

    def _record(self):
        from .analysis import diagnostics
        self._records.append(diagnostics(self.state, self.params))

    def step(self, n=1):
        """Advance n outer turns. Chunk size and display rate do not alter dynamics."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError('n must be a nonnegative integer')
        p = self.params
        for _ in range(n):
            remaining = p.dt
            while remaining > p.dt * 1e-12:
                rho = density_grid(self._positions, p) if self.scenario.dynamic else None
                rate = max(p.v_max / (0.3 * min(p.dx, p.dy)),
                           p.rotational_diffusion / 0.1)
                if self.scenario.dynamic:
                    rate = max(rate, 2.5 * p.oxygen_diffusion * (p.dx**-2 + p.dy**-2),
                               p.replenishment / 0.2,
                               p.consumption * rho.max() / (0.2 * p.oxygen_half_saturation))
                h = min(remaining, 1 / rate)
                sensed = sample_field(self._oxygen, self._positions, p)
                self._headings[:] = (self._headings + np.sqrt(2*p.rotational_diffusion*h)
                                     * self.rng.normal(size=p.n_agents)) % (2*np.pi)
                directions = np.column_stack((np.cos(self._headings), np.sin(self._headings)))
                self._positions[:] = (self._positions + h * speed(sensed, p)[:, None] * directions
                                      ) % (p.width, p.height)
                if self.scenario.dynamic:
                    _oxygen_step(self._oxygen, rho, p, h)
                remaining -= h
            self.turn += 1
            self.time += p.dt
            if self.turn % self.record_every == 0:
                self._record()
        return self.state

    def result(self):
        """Snapshot plus regularly sampled diagnostics, always including the endpoint."""
        from .analysis import diagnostics
        records = list(self._records)
        if records[-1]['turn'] != self.turn:
            records.append(diagnostics(self.state, self.params))
        history = {key: np.array([r[key] for r in records]) for key in records[0]}
        import copy
        return Result(self.state, history, self.params, self.scenario.key,
                      self.seed, tuple(copy.deepcopy(self.parameter_log)))


def run(scenario='feedback', params=None, *, seed=1, turns=3000, record_every=20):
    """Finite headless run using exactly the interactive engine."""
    simulation = Simulation(scenario, params, seed=seed, record_every=record_every)
    simulation.step(turns)
    return simulation.result()
