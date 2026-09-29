"""What the user controls in Model 1: the oxygen environment, nothing else.

The environment enters the oxygen equation only through the replenishment term

    dO/dt |_env = f(x, y) * (O_am(x, y, t) - O)

so the agents' own consumption always competes with it. Nothing here ever
writes to the oxygen field directly: a "low-oxygen spot" is a region where the
gas being exchanged against is poor, not a region where oxygen is deleted. That
distinction matters, because the feedback that makes patterns lives in O.

Oxygen is normalised so ambient air (about 21% O2) is 1.0; `pct` converts.
Replenishment f is a rate: a glass cover is a patch of small f, not of small
O_am, and the two look different because f also sets how fast the field relaxes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable
import numpy as np

AIR_PERCENT = 21.0


def pct(percent):
    """Percent oxygen -> normalised units. pct(21) == 1.0, pct(7) == 1/3."""
    return np.asarray(percent, dtype=float) / AIR_PERCENT


def cell_centers(width, height, nx, ny):
    """Meshgrid of cell-centre coordinates, indexed [y, x] like every field."""
    x = (np.arange(nx) + 0.5) * (width / nx)
    y = (np.arange(ny) + 0.5) * (height / ny)
    return np.meshgrid(x, y)


@dataclass(frozen=True)
class Environment:
    """Ambient oxygen, replenishment rate, and oxygen diffusivity.

    ambient : float, [ny, nx] array, or callable (x_grid, y_grid, t) -> array.
        The callable is evaluated once per **outer turn** (see `Simulation`),
        so a schedule resolves changes no finer than dt.
    replenishment : float or [ny, nx] array. Static in time by design: a cover
        glass does not move. Use `ambient` for anything time-dependent.
    oxygen_diffusion : float.
    """
    ambient: float | np.ndarray | Callable = 1.0
    replenishment: float | np.ndarray = 0.06
    oxygen_diffusion: float = 1.0
    label: str = 'uniform'

    def __post_init__(self):
        if not callable(self.ambient):
            a = np.asarray(self.ambient, dtype=float)
            if not np.isfinite(a).all():
                raise ValueError('ambient must be finite')
            if (a < 0).any() or (a > 1).any():
                raise ValueError('ambient must lie in [0, 1] (1 = air)')
        f = np.asarray(self.replenishment, dtype=float)
        if not np.isfinite(f).all() or (f < 0).any():
            raise ValueError('replenishment must be finite and nonnegative')
        if not np.isfinite(self.oxygen_diffusion) or self.oxygen_diffusion < 0:
            raise ValueError('oxygen_diffusion must be finite and nonnegative')

    # -- resolution onto a grid ------------------------------------------
    def ambient_field(self, x_grid, y_grid, t):
        """O_am on the grid at time t. Scalars stay scalars (bit-for-bit)."""
        if callable(self.ambient):
            value = np.asarray(self.ambient(x_grid, y_grid, float(t)), dtype=float)
            if value.ndim == 0:
                return float(value)
            if value.shape != x_grid.shape:
                value = np.broadcast_to(value, x_grid.shape)
            if (value < 0).any() or (value > 1).any() or not np.isfinite(value).all():
                raise ValueError('ambient callable returned values outside [0, 1]')
            return value
        a = np.asarray(self.ambient, dtype=float)
        if a.ndim == 0:
            return float(a)
        if a.shape != x_grid.shape:
            raise ValueError(f'ambient array has shape {a.shape}, expected {x_grid.shape}')
        return a

    def replenishment_field(self, shape):
        """f on the grid. Scalars stay scalars, so the scalar path is exact."""
        f = np.asarray(self.replenishment, dtype=float)
        if f.ndim == 0:
            return float(f)
        if f.shape != shape:
            raise ValueError(f'replenishment array has shape {f.shape}, expected {shape}')
        return f

    @property
    def time_varying(self):
        return callable(self.ambient)

    def representative(self, grid=None, t=0.0):
        """(ambient, replenishment) scalars for linear stability.

        Linear stability is a statement about a *homogeneous* reference state,
        so a field has to be collapsed to one number before it can be used, and
        the collapse is only meaningful when the variation is small.

        Arrays are reduced to their mean. A callable ambient is evaluated at
        time `t` and reduced to its mean over `grid` -- the (x, y) meshgrid the
        engine already holds. Without a grid it is evaluated at the origin
        instead, which is exact for a schedule or any spatially uniform
        callable and only indicative for a spot or a gradient; pass the grid
        when the answer has to be right for those. `stability.predict` reports
        `homogeneous_reference=False` whenever this collapse happened.
        """
        f = float(np.mean(np.asarray(self.replenishment, dtype=float)))
        if callable(self.ambient):
            x, y = grid if grid is not None else (np.zeros((1, 1)), np.zeros((1, 1)))
            a = float(np.mean(np.asarray(self.ambient(x, y, float(t)), dtype=float)))
        else:
            a = float(np.mean(np.asarray(self.ambient, dtype=float)))
        return a, f

    def variant(self, **changes):
        return replace(self, **changes)


# -- builders ------------------------------------------------------------

def uniform(level=1.0, replenishment=0.06, oxygen_diffusion=1.0):
    """Spatially and temporally constant ambient oxygen."""
    return Environment(float(level), replenishment, oxygen_diffusion,
                       label=f'uniform {level:.3g}')


def linear_gradient(lo, hi, axis='x', *, width, height, nx, ny,
                    replenishment=0.06, oxygen_diffusion=1.0):
    """A smooth **cosine** ramp between lo and hi across the domain.

    The domain is periodic, so a genuinely linear ramp would leave a jump at
    the seam and the discontinuity, not the gradient, would dominate the
    dynamics. The cosine profile runs lo -> hi -> lo with no seam; the name is
    kept because it is the periodic stand-in for the usual gradient chamber.
    Half the domain therefore carries the increasing branch.
    """
    x_grid, y_grid = cell_centers(width, height, nx, ny)
    coord, span = (x_grid, width) if axis == 'x' else (y_grid, height)
    if axis not in ('x', 'y'):
        raise ValueError("axis must be 'x' or 'y'")
    mid, amp = 0.5 * (hi + lo), 0.5 * (hi - lo)
    field = mid - amp * np.cos(2 * np.pi * coord / span)
    return Environment(field, replenishment, oxygen_diffusion,
                       label=f'cosine {lo:.3g}-{hi:.3g} along {axis}')


def spot(center, radius, level, background=1.0, *, width, height, nx, ny,
         softness=0.15, replenishment=0.06, oxygen_diffusion=1.0):
    """A static disc of ambient `level` on a `background` field.

    The edge is a tanh of width `softness * radius` so the discretised disc has
    no stair-step, which otherwise seeds patterns at the grid scale.
    """
    x_grid, y_grid = cell_centers(width, height, nx, ny)
    field = background + (level - background) * _disc(
        x_grid, y_grid, center, radius, softness, width, height)
    return Environment(field, replenishment, oxygen_diffusion,
                       label=f'spot r={radius:.3g} at {tuple(center)}')


def moving_spot(path, radius, level, background=1.0, *, width, height, nx, ny,
                softness=0.15, replenishment=0.06, oxygen_diffusion=1.0):
    """A disc whose centre follows `path(t) -> (x, y)`.

    This is the actuator for the inverse-control experiment: the only thing
    that moves is where the gas exchange is poor.
    """
    dom = (width, height)

    def ambient(x_grid, y_grid, t):
        center = np.asarray(path(t), dtype=float)
        return background + (level - background) * _disc(
            x_grid, y_grid, center, radius, softness, *dom)

    return Environment(ambient, replenishment, oxygen_diffusion,
                       label=f'moving spot r={radius:.3g}')


def schedule(steps, replenishment=0.06, oxygen_diffusion=1.0):
    """Piecewise-constant ambient level in time: [(t0, level0), (t1, level1), ...].

    The level in force is the last entry whose time is <= t. Reproduces the
    21% -> 7% -> 21% protocol of Gray et al. (2004) Fig. 4d as
    `schedule([(0, pct(21)), (300, pct(7)), (600, pct(21))])`.
    """
    steps = sorted(((float(t), float(v)) for t, v in steps), key=lambda s: s[0])
    if not steps:
        raise ValueError('schedule needs at least one (time, level) pair')
    if steps[0][0] > 0:
        steps.insert(0, (0.0, steps[0][1]))
    times = np.array([s[0] for s in steps])
    levels = np.array([s[1] for s in steps])
    if ((levels < 0) | (levels > 1)).any():
        raise ValueError('schedule levels must lie in [0, 1]')

    def ambient(x_grid, y_grid, t):
        return float(levels[np.searchsorted(times, t, side='right') - 1])

    return Environment(ambient, replenishment, oxygen_diffusion,
                       label='schedule ' + ', '.join(f't={t:g}:{v:.3g}' for t, v in steps))


def ramp(start_level, end_level, duration, replenishment=0.06, oxygen_diffusion=1.0):
    """Ambient sweeping linearly from start_level to end_level over `duration`,
    then held. Used for the hysteresis experiment."""
    start_level, end_level, duration = float(start_level), float(end_level), float(duration)
    if duration <= 0:
        raise ValueError('duration must be positive')

    def ambient(x_grid, y_grid, t):
        u = min(max(t / duration, 0.0), 1.0)
        return start_level + (end_level - start_level) * u

    return Environment(ambient, replenishment, oxygen_diffusion,
                       label=f'ramp {start_level:.3g}->{end_level:.3g} over {duration:g}')


def cover_glass(center, radius, covered_rate, open_rate=0.06, *, width, height,
                nx, ny, softness=0.15, ambient=1.0, oxygen_diffusion=1.0):
    """A patch of reduced gas exchange: f is small inside the disc.

    The ambient level is unchanged; what changes is how fast the field is
    pulled towards it, so the agents' own consumption wins inside the patch.
    """
    x_grid, y_grid = cell_centers(width, height, nx, ny)
    disc = _disc(x_grid, y_grid, center, radius, softness, width, height)
    f = open_rate + (covered_rate - open_rate) * disc
    return Environment(ambient, f, oxygen_diffusion,
                       label=f'cover glass r={radius:.3g} f={covered_rate:.3g}')


def _disc(x_grid, y_grid, center, radius, softness, width, height):
    """Smooth periodic indicator of a disc: 1 inside, 0 outside."""
    dx = (x_grid - center[0] + 0.5 * width) % width - 0.5 * width
    dy = (y_grid - center[1] + 0.5 * height) % height - 0.5 * height
    r = np.hypot(dx, dy)
    edge = max(softness * radius, 1e-9)
    return 0.5 * (1 - np.tanh((r - radius) / edge))
