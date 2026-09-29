"""Initial conditions and external field protocols, separate from agent rules."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    dynamic: bool
    description: str


SCENARIOS = {
    'uniform': Scenario('uniform', 'Uniform fixed oxygen', False,
                        'Uniform agents; oxygen externally clamped at ambient.'),
    'landscape': Scenario('landscape', 'Prescribed oxygen landscape', False,
                          'Uniform agents; a static periodic oxygen wave along x.'),
    'feedback': Scenario('feedback', 'Oxygen–motility feedback', True,
                         'Uniform agents; oxygen starts at homogeneous reaction equilibrium.'),
    'seeded': Scenario('seeded', 'Relaxation of a density perturbation', True,
                       '20% of agents begin in a central Gaussian patch; others uniform.'),
}


def get_scenario(name):
    try:
        return SCENARIOS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Choose a scenario from {tuple(SCENARIOS)}') from exc


def uniform_oxygen(params):
    """Continuum homogeneous reaction balance (before time discretization).

    k*(ambient-c) = q*rho*c/(K+c). For k=0 and q>0 equilibrium is zero.
    If both vanish, ambient supplies the otherwise arbitrary initial level.
    """
    p = params
    if p.consumption == 0:
        return p.ambient
    if p.replenishment == 0:
        return 0.0
    b = p.oxygen_half_saturation - p.ambient + (
        p.consumption * p.n_agents / (p.width*p.height) / p.replenishment)
    c = p.oxygen_half_saturation * p.ambient
    root = np.sqrt(b*b + 4*c)
    return 2*c/(root+b) if b >= 0 and root+b > 0 else (root-b)/2


def initial_conditions(scenario, p, rng):
    positions = rng.uniform(size=(p.n_agents, 2)) * (p.width, p.height)
    headings = rng.uniform(0, 2*np.pi, p.n_agents)
    if scenario.key == 'seeded':
        m = p.n_agents // 5
        positions[:m] = (rng.normal(size=(m, 2)) * (0.08*p.width, 0.08*p.height)
                         + (p.width/2, p.height/2)) % (p.width, p.height)
    if scenario.key == 'landscape':
        # Smooth periodic profile; its minimum is 0.3 and maximum 0.8.
        x = (np.arange(p.nx) + 0.5) * p.dx
        row = 0.55 + 0.25 * np.cos(2*np.pi*x/p.width)
        oxygen = np.repeat(row[None, :], p.ny, axis=0)
    else:
        value = uniform_oxygen(p) if scenario.dynamic else p.ambient
        oxygen = np.full((p.ny, p.nx), value, dtype=float)
    return positions, headings, oxygen


def info():
    for s in SCENARIOS.values():
        print(f'{s.key:10s}  {s.description}')
