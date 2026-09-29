"""Inverse control: steer an aggregate by moving a low-oxygen spot (milestone M6).

The forward question of Model 1 is "what does this environment do to the
animals". This module asks the inverse: given a path you want an aggregate to
follow, what environment makes it follow? The actuator is deliberately the
weakest one available -- a disc where the gas being exchanged against is poor.
It never writes to the oxygen field, never touches an agent, and carries no
information about where the agents currently are. It is open loop.

The physics it exploits is the same one the rest of the package is about: on the
rising branch of V(O), low oxygen means slow agents, and slow agents accumulate.
Prior art for the idea in synthetic active matter is Arlt et al. (2018) and
Frangipane et al. (2018), who shape bacterial density with patterned light.

A trial has two phases:

  nucleate  the spot is held still at the start of the path, long enough for an
            aggregate to gather in it;
  track     the spot moves along the path and the error is the distance between
            the agents' centre of mass and the spot centre.

Three numbers are searched: the spot radius, its oxygen depth, and how fast it
travels. They trade off against each other -- a deep narrow spot holds an
aggregate tightly but is easy to outrun, and a fast spot simply leaves it
behind -- which is why this is a search and not a calculation.

The optimiser is a compact CMA-ES written here rather than pulled in as a
dependency. Every objective evaluation averages over at least two seeds,
because a single seed's aggregate can happen to nucleate in the right place.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .engine import Simulation
from .environment import moving_spot
from .strains import get_strain

#: (low, high) physical bounds for each searched parameter, in order.
BOUNDS = dict(radius=(6.0, 26.0), level=(0.02, 0.60), speed=(0.01, 0.35))
PARAMETER_ORDER = ('radius', 'level', 'speed')


@dataclass(frozen=True)
class SteeringTask:
    """A path to follow and the domain to follow it in.

    The path is a circle of `path_radius` about the domain centre, travelled at
    the searched speed. A circle keeps the target away from the periodic seam,
    so the tracking error never has to be interpreted through a wrap.
    """
    strain: str = 'NPR1_LIKE'
    side: float = 120.0
    cells: int = 120
    density: float = 0.70
    ambient: float = 1.0
    replenishment: float = 0.06
    dt: float = 0.1
    nucleate_time: float = 400.0
    track_time: float = 900.0
    path_radius: float = 34.0
    sample_every: float = 10.0

    @property
    def n_agents(self):
        return int(round(self.density * self.side * self.side))

    @property
    def center(self):
        return (self.side / 2, self.side / 2)

    def target(self, t, speed):
        """Where the spot should be at time t, given its travel speed.

        Angular speed is `speed / path_radius`, so `speed` is the tangential
        speed in model length units per model time -- directly comparable with
        the agents' own V(O).
        """
        held = max(t - self.nucleate_time, 0.0)
        angle = held * speed / self.path_radius
        cx, cy = self.center
        return (cx + self.path_radius * np.cos(angle),
                cy + self.path_radius * np.sin(angle))


def clamp(values):
    """Map an unbounded CMA-ES vector into the physical parameter box.

    A smooth squash rather than a hard clip, so the optimiser still sees a
    gradient when it pushes a parameter against its bound.
    """
    values = np.atleast_1d(np.asarray(values, dtype=float))
    out = {}
    for value, name in zip(values, PARAMETER_ORDER):
        lo, hi = BOUNDS[name]
        out[name] = float(lo + (hi - lo) / (1 + np.exp(-value)))
    return out


def unclamp(params):
    """The inverse of `clamp`, for seeding the search from a starting guess."""
    values = []
    for name in PARAMETER_ORDER:
        lo, hi = BOUNDS[name]
        u = (float(params[name]) - lo) / (hi - lo)
        u = min(max(u, 1e-6), 1 - 1e-6)
        values.append(float(np.log(u / (1 - u))))
    return np.array(values)


def periodic_center_of_mass(positions, side):
    """Centre of mass on a torus, via the circular mean in each coordinate.

    A plain mean is meaningless here: a cluster straddling the seam averages to
    the middle of the domain, which is exactly where it is not.
    """
    angles = positions / side * 2 * np.pi
    out = []
    for axis in (0, 1):
        mean = np.exp(1j * angles[:, axis]).mean()
        out.append(float(np.angle(mean) % (2 * np.pi) / (2 * np.pi) * side))
    return np.array(out)


def periodic_distance(a, b, side):
    """Shortest distance between two points on a square torus."""
    delta = np.abs(np.asarray(a) - np.asarray(b)) % side
    return float(np.hypot(*np.minimum(delta, side - delta)))


def trial(params, seed, task=None, *, record=False):
    """One steering run. Returns the mean tracking error and, optionally, the track.

    The error is the mean distance between the agents' centre of mass and the
    spot centre over the tracking phase, in model length units. A run whose
    aggregate never formed is not special-cased: a dispersed cloud has its
    centre of mass near the domain centre, which is `path_radius` away from the
    target, so it scores badly on its own.
    """
    task = task or SteeringTask()
    # A dict is already physical; a vector comes from the optimiser's unbounded
    # space and has to be squashed into the parameter box first.
    p = dict(params) if isinstance(params, dict) else clamp(params)
    env = moving_spot(lambda t: task.target(t, p['speed']), p['radius'], p['level'],
                      task.ambient, width=task.side, height=task.side,
                      nx=task.cells, ny=task.cells,
                      replenishment=task.replenishment)
    sim = Simulation(get_strain(task.strain), env,
                     domain=(task.side, task.side, task.cells, task.cells),
                     n_agents=task.n_agents, dt=task.dt, seed=int(seed),
                     record_every=10**9, record_patterns=False)
    sim.run_until(task.nucleate_time)
    errors, track = [], []
    steps = int(round(task.track_time / task.sample_every))
    for i in range(1, steps + 1):
        sim.run_until(task.nucleate_time + i * task.sample_every)
        com = periodic_center_of_mass(np.asarray(sim.state.positions), task.side)
        goal = task.target(sim.time, p['speed'])
        errors.append(periodic_distance(com, goal, task.side))
        if record:
            track.append((sim.time, com[0], com[1], goal[0], goal[1], errors[-1]))
    result = dict(error=float(np.mean(errors)), final_error=float(errors[-1]),
                  worst_error=float(np.max(errors)), params=p, seed=int(seed))
    if record:
        result['track'] = np.array(track)
        result['state'] = sim.state
        result['params_object'] = sim.params
    return result


def _evaluate(payload):
    vector, seeds, task = payload
    p = clamp(vector)
    scores = [trial(p, seed, task)['error'] for seed in seeds]
    return float(np.mean(scores)), float(np.std(scores)), p


def cmaes(objective, x0, sigma0=1.0, *, generations=15, population=None,
          rng=None, callback=None):
    """A compact (mu/mu_w, lambda)-CMA-ES. Minimises `objective(list_of_x)`.

    `objective` takes the whole population at once so the caller can evaluate it
    in parallel. This is the standard algorithm with the usual default weights
    and learning rates; it is written out here to avoid adding a dependency for
    one search, not because anything about it is novel.
    """
    rng = rng or np.random.default_rng(0)
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    lam = population or (4 + int(3 * np.log(n)))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)

    c_sigma = (mueff + 2) / (n + mueff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean, sigma = x0.copy(), float(sigma0)
    p_sigma, p_c = np.zeros(n), np.zeros(n)
    C, eigen_age = np.eye(n), 0
    best = (np.inf, mean.copy())
    history = []

    for generation in range(generations):
        values, B = np.linalg.eigh(C)
        values = np.maximum(values, 1e-20)
        D = np.sqrt(values)
        samples = [mean + sigma * (B @ (D * rng.normal(size=n))) for _ in range(lam)]
        scores = np.asarray(objective(samples), dtype=float)
        order = np.argsort(scores)
        if scores[order[0]] < best[0]:
            best = (float(scores[order[0]]), samples[order[0]].copy())
        selected = np.array([samples[i] for i in order[:mu]])
        old_mean = mean.copy()
        mean = weights @ selected

        C_invsqrt = B @ np.diag(1 / D) @ B.T
        p_sigma = ((1 - c_sigma) * p_sigma
                   + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_invsqrt @ (mean - old_mean)) / sigma)
        norm = np.linalg.norm(p_sigma)
        h_sigma = norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1))) / chi_n < 1.4 + 2 / (n + 1)
        p_c = ((1 - cc) * p_c
               + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma)
        artmp = (selected - old_mean) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(p_c, p_c) + (not h_sigma) * cc * (2 - cc) * C)
             + cmu * (artmp.T @ (weights[:, None] * artmp)))
        C = 0.5 * (C + C.T)
        sigma *= np.exp((c_sigma / d_sigma) * (norm / chi_n - 1))
        history.append(dict(generation=generation, best=best[0],
                            generation_best=float(scores[order[0]]),
                            median=float(np.median(scores)), sigma=sigma))
        if callback:
            callback(history[-1], best[1])
    return dict(x=best[1], value=best[0], params=clamp(best[1]), history=history)


def search(task=None, *, seeds=(1, 2), generations=12, population=8,
           processes=None, seed=0, start=None, progress=True):
    """Optimise (radius, level, speed) for a steering task with CMA-ES.

    Each candidate is scored as the mean tracking error over `seeds`, so a
    parameter set that only works for one initial condition cannot win.
    """
    from concurrent.futures import ProcessPoolExecutor
    task = task or SteeringTask()
    start = start or dict(radius=14.0, level=0.15, speed=0.08)
    x0 = unclamp(start)

    def objective(samples):
        payloads = [(x, tuple(seeds), task) for x in samples]
        with ProcessPoolExecutor(max_workers=processes) as pool:
            results = list(pool.map(_evaluate, payloads))
        if progress:
            best = min(results, key=lambda r: r[0])
            print(f'    best this generation: error {best[0]:6.2f} '
                  f'(+/- {best[1]:.2f}) at ' +
                  ', '.join(f'{k} {v:.3f}' for k, v in best[2].items()))
        return [r[0] for r in results]

    def report(entry, _):
        if progress:
            print(f'  generation {entry["generation"]:2d}: best so far '
                  f'{entry["best"]:6.2f}  sigma {entry["sigma"]:.3f}')

    return cmaes(objective, x0, 1.0, generations=generations, population=population,
                 rng=np.random.default_rng(seed), callback=report)
