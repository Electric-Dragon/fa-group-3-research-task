"""Model 1: environment-controlled pattern explorer.

Agent parameters are frozen into a named `Strain`. The user controls the
`Environment` -- ambient oxygen as a scalar, a map or a schedule, the
replenishment map, the oxygen diffusivity -- and the population, at runtime.

The package predicts where patterns should appear (`stability`), simulates
(`engine`), classifies what actually formed (`patterns`), and sweeps the two
together into a phase diagram with the predicted boundary drawn on it (`sweep`).

    from celegans import model1 as m1

    env = m1.environment.uniform(m1.environment.pct(21))
    print(m1.predict('NPR1_LIKE', env, density=0.6)['wavelength'])

    sim = m1.Simulation('NPR1_LIKE', env, domain=(128, 128, 128, 128), n_agents=9800)
    sim.step(2000)
    print(sim.result().patterns['label'])
"""
from . import strains, environment, stability, patterns, sweep, control
from .strains import (Strain, NPR1_LIKE, N2_LIKE, CONSTANT_SPEED,
                      MODEL0_DEFAULT, HYPOXIC_BRANCH, PRESETS, get_strain)
from .environment import Environment, pct
from .engine import Simulation, Result, run, density_grid, DIAGNOSTIC_LENGTH
from .stability import (growth_curve, predict, predicted_map, equilibrium_oxygen,
                        critical_density, unstable_density_range, boundary_curve)
from .patterns import classify, structure_factor, coarsening_exponent
from .sweep import sweep_phase_diagram, load_runs
from .control import SteeringTask, trial as steering_trial, search as steering_search

__all__ = [
    'strains', 'environment', 'stability', 'patterns', 'sweep', 'control',
    'Strain', 'NPR1_LIKE', 'N2_LIKE', 'CONSTANT_SPEED', 'MODEL0_DEFAULT',
    'HYPOXIC_BRANCH', 'PRESETS', 'get_strain',
    'Environment', 'pct', 'Simulation', 'Result', 'run', 'density_grid',
    'DIAGNOSTIC_LENGTH', 'growth_curve', 'predict', 'predicted_map',
    'equilibrium_oxygen', 'critical_density', 'unstable_density_range',
    'boundary_curve', 'classify', 'structure_factor', 'coarsening_exponent',
    'Demo', 'plot', 'plot_growth_curve', 'plot_phase_diagram', 'plot_speed_laws',
    'sweep_phase_diagram', 'load_runs',
    'SteeringTask', 'steering_trial', 'steering_search',
]


def Demo(*args, **kwargs):
    """Load widget dependencies only when an interactive demo is requested."""
    from .demo import EnvironmentDemo
    return EnvironmentDemo(*args, **kwargs)


def plot(*args, **kwargs):
    from .visualization import plot_result
    return plot_result(*args, **kwargs)


def plot_growth_curve(*args, **kwargs):
    from .visualization import plot_growth_curve as _f
    return _f(*args, **kwargs)


def plot_phase_diagram(*args, **kwargs):
    from .visualization import plot_phase_diagram as _f
    return _f(*args, **kwargs)


def plot_speed_laws(*args, **kwargs):
    from .visualization import plot_speed_laws as _f
    return _f(*args, **kwargs)
