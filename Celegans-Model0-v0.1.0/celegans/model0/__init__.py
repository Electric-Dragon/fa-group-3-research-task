"""Model 0 public API."""
from .model import Parameters, Simulation, State, Result, run, speed, density_grid
from .scenarios import info, uniform_oxygen
from .analysis import diagnostics, x_profile


def Demo(*args, **kwargs):
    """Load widget dependencies only when an interactive demo is requested."""
    from .demo import OxygenDemo
    return OxygenDemo(*args, **kwargs)


def plot(*args, **kwargs):
    from .visualization import plot_result
    return plot_result(*args, **kwargs)


def compare(*args, **kwargs):
    from .visualization import plot_comparison
    return plot_comparison(*args, **kwargs)
