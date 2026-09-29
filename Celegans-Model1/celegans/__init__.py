"""C. elegans-inspired agent models.

Model 0 (`celegans.model0`) is the released teaching baseline and is re-exported
at the top level unchanged, so its regression tests run untouched.

Model 1 (`celegans.model1`) freezes the agent parameters into named strains and
hands the user the environment instead: ambient oxygen as a scalar, a map or a
schedule, a replenishment map, and runtime changes of population.
"""
from .model0 import (Parameters, Simulation, State, Result, run, speed,
                     density_grid, info, uniform_oxygen, diagnostics,
                     x_profile, Demo, plot, compare)
from . import model1

__version__ = '1.0.0'
