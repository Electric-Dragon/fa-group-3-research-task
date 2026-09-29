"""Diagnostics describe heterogeneity; they do not certify phase coexistence."""
import numpy as np
from .model import density_grid, sample_field, speed

DIAGNOSTIC_LENGTH = 2.0  # fixed model length, independent of consumption footprint


def diagnostics(state, params):
    rho = density_grid(state.positions, params, smoothing=DIAGNOSTIC_LENGTH)
    relative = rho / rho.mean()
    polar = np.abs(np.mean(np.exp(1j * state.headings)))
    return dict(turn=state.turn, time=state.time,
                heterogeneity=float(np.mean((relative - 1)**2)),
                mean_oxygen=float(state.oxygen.mean()),
                min_oxygen=float(state.oxygen.min()),
                mean_speed=float(speed(sample_field(state.oxygen, state.positions, params), params).mean()),
                polar_order=float(polar))


def x_profile(result, bins=24):
    """Relative particle density along x; mean equals one for equal-width bins."""
    counts, edges = np.histogram(result.positions[:, 0], bins=bins,
                                range=(0, result.params.width))
    return 0.5*(edges[1:] + edges[:-1]), counts / counts.mean()
