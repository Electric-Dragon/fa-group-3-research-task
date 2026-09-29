"""Frozen agent parameters. In Model 1 the user never edits these.

A `Strain` bundles everything about the agents: how fast they move at a given
local oxygen level, how quickly they forget their heading, how much oxygen they
consume and over what footprint. Model 1's controls are all environmental, so
the strain is the only place agent behaviour is set, and it is set once.

The presets are **teaching presets, not fitted parameters**. They are named
after the two strains in Demir et al. (2020) and Gray et al. (2004) because
they reproduce the qualitative contrast reported there (a steep, low-oxygen
preference that aggregates readily versus a shallow response that needs high
density), not because any number here was measured in a worm. Units are the
arbitrary model units of Model 0, with oxygen normalised so ambient air = 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.special import expit

from ..model0.model import Parameters

#: Speed laws a Strain may use. 'sigmoid' is Model 0's monotone response.
#: 'hypoxic' adds the extra high-speed branch at very low oxygen reported in
#: Demir et al. (2020) Fig. 2b. It is off by default: the linear-stability
#: analysis in `stability.py` is valid for either, but the non-monotone law
#: makes V'(O) change sign, so "pattern" and "no pattern" regions interleave.
SPEED_LAWS = ('sigmoid', 'hypoxic')


@dataclass(frozen=True)
class Strain:
    """Immutable agent parameters.

    Attributes
    ----------
    name : label used in plots and output files.
    v_min, v_max : speed floor and ceiling of the monotone sigmoid branch.
    oxygen_midpoint : normalised oxygen at which the sigmoid is half way up.
    response_width : oxygen width of the sigmoid; small = steep = strong beta.
    rotational_diffusion : the paper's "tau", which is a **rate** (1/time),
        not a time. In two dimensions D_W = V^2 / (2 * rotational_diffusion).
    consumption : q, oxygen * area / (agent * time).
    oxygen_half_saturation : K in the Michaelis-Menten sink q*rho*O/(K+O).
    kernel_width : sigma of the Gaussian consumption footprint (a footprint,
        not a body radius).
    speed_law : 'sigmoid' or 'hypoxic'.
    hypoxic_amplitude, hypoxic_midpoint, hypoxic_width : the optional low-oxygen
        branch, used only when speed_law == 'hypoxic'.
    """
    name: str
    v_min: float
    v_max: float
    oxygen_midpoint: float
    response_width: float
    rotational_diffusion: float
    consumption: float
    oxygen_half_saturation: float
    kernel_width: float
    speed_law: str = 'sigmoid'
    hypoxic_amplitude: float = 0.0
    hypoxic_midpoint: float = 0.08
    hypoxic_width: float = 0.03

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('name must be a non-empty string')
        if self.speed_law not in SPEED_LAWS:
            raise ValueError(f'speed_law must be one of {SPEED_LAWS}')
        positive = ('v_max', 'response_width', 'oxygen_half_saturation',
                    'kernel_width', 'hypoxic_width')
        nonnegative = ('v_min', 'rotational_diffusion', 'consumption',
                       'hypoxic_amplitude')
        for field in positive + nonnegative + ('oxygen_midpoint', 'hypoxic_midpoint'):
            value = getattr(self, field)
            if not np.isfinite(value):
                raise ValueError(f'{field} must be finite')
            if field in positive and value <= 0:
                raise ValueError(f'{field} must be positive')
            if field in nonnegative and value < 0:
                raise ValueError(f'{field} must be nonnegative')
        if self.v_min > self.v_max:
            raise ValueError('v_min must not exceed v_max')
        if not 0 <= self.oxygen_midpoint <= 1:
            raise ValueError('oxygen_midpoint must lie in [0, 1]')

    # -- speed law -------------------------------------------------------
    def speed(self, oxygen):
        """V(O): strictly positive, and monotone unless speed_law='hypoxic'."""
        oxygen = np.asarray(oxygen, dtype=float)
        span = self.v_max - self.v_min
        v = self.v_min + span * expit(
            (oxygen - self.oxygen_midpoint) / self.response_width)
        if self.speed_law == 'hypoxic' and self.hypoxic_amplitude:
            v = v + self.hypoxic_amplitude * expit(
                (self.hypoxic_midpoint - oxygen) / self.hypoxic_width)
        return v

    def speed_derivative(self, oxygen):
        """V'(O), analytic. Drives beta = V V' / (2 D_r), hence the instability."""
        oxygen = np.asarray(oxygen, dtype=float)
        span = self.v_max - self.v_min
        s = expit((oxygen - self.oxygen_midpoint) / self.response_width)
        dv = span * s * (1 - s) / self.response_width
        if self.speed_law == 'hypoxic' and self.hypoxic_amplitude:
            t = expit((self.hypoxic_midpoint - oxygen) / self.hypoxic_width)
            dv = dv - self.hypoxic_amplitude * t * (1 - t) / self.hypoxic_width
        return dv

    @property
    def max_speed(self):
        """Upper bound on V(O) over all O; the substep displacement limit uses it."""
        return self.v_max + (self.hypoxic_amplitude
                             if self.speed_law == 'hypoxic' else 0.0)

    def persistence_length(self, oxygen):
        """V / D_r: the length over which a heading is remembered.

        The drift-diffusion description behind `stability.py` assumes the
        pattern wavelength is much larger than this.
        """
        if self.rotational_diffusion == 0:
            return np.inf
        return self.speed(oxygen) / self.rotational_diffusion

    # -- interoperability with Model 0 -----------------------------------
    def to_parameters(self, *, width, height, nx, ny, n_agents, dt,
                      oxygen_diffusion=1.0, replenishment=0.06, ambient=1.0):
        """A Model 0 `Parameters` carrying this strain plus a domain.

        Model 1 uses it only for the geometry helpers (`dx`, `dy`, the
        cloud-in-cell weights) and for the Model 0 speed law when
        speed_law == 'sigmoid'. Environment fields that Model 1 allows to be
        arrays are passed here as representative scalars; the engine's own
        oxygen step uses the arrays, never these.
        """
        return Parameters(
            width=width, height=height, nx=nx, ny=ny, n_agents=n_agents, dt=dt,
            v_min=self.v_min, v_max=self.v_max,
            oxygen_midpoint=self.oxygen_midpoint,
            response_width=self.response_width,
            rotational_diffusion=self.rotational_diffusion,
            oxygen_diffusion=oxygen_diffusion, replenishment=replenishment,
            ambient=ambient, consumption=self.consumption,
            oxygen_half_saturation=self.oxygen_half_saturation,
            kernel_width=self.kernel_width, speed_response=True)

    def variant(self, **changes):
        """A copy with fields replaced; the original stays frozen."""
        return replace(self, **changes)


# -- presets -------------------------------------------------------------
# Tuned with stability.py; the reasoning and the numbers are in VALIDATION.md
# ("Preset tuning"). Over ambient 0.3-1.0 and density 0.3-1.3 agents/area,
# NPR1_LIKE is unstable over a broad band that starts at low density, while
# N2_LIKE only becomes unstable well above it. These are teaching presets.

MODEL0_DEFAULT = Strain(
    name='MODEL0_DEFAULT',
    v_min=0.06, v_max=2.0,
    oxygen_midpoint=0.55, response_width=0.045,
    rotational_diffusion=0.25,
    consumption=0.06, oxygen_half_saturation=0.10,
    kernel_width=1.2)
"""Model 0's released defaults, verbatim. Kept as a regression anchor: Model 1's
engine run with this strain and a uniform environment must reproduce Model 0's
trajectories, and it is the strain the documented stability oracle refers to
(O_eq 0.510, wavelength 21.5, lambda_max 0.057 at density 0.586, ambient 1.0)."""

NPR1_LIKE = Strain(
    name='NPR1_LIKE',
    v_min=0.06, v_max=2.0,
    oxygen_midpoint=0.55, response_width=0.09,
    rotational_diffusion=0.25,
    consumption=0.06, oxygen_half_saturation=0.10,
    kernel_width=1.2)
"""Steep, high-oxygen-avoiding response; stands in for the aggregating npr-1
strain. Model 0's defaults with `response_width` doubled from 0.045 to 0.09.
The widening is deliberate: at 0.045 the response is so steep that V'(O) is
appreciable only in a narrow oxygen window, so the unstable region is a thin
diagonal sliver in the (ambient, density) plane and no single ambient level
holds more than about three unstable densities. At 0.09 the response is still
steep (it rises over roughly 0.36 in normalised oxygen, i.e. about 7% O2) but
the unstable band is wide enough to sweep density across it, which is what the
dots/stripes/holes ordering needs."""

N2_LIKE = Strain(
    name='N2_LIKE',
    v_min=0.15, v_max=1.60,
    oxygen_midpoint=0.35, response_width=0.10,
    rotational_diffusion=0.25,
    consumption=0.06, oxygen_half_saturation=0.10,
    kernel_width=1.2)
"""Weaker aerotaxis; stands in for solitary N2. Two changes from NPR1_LIKE: the
speed span v_max - v_min is smaller (1.45 against 1.94), which lowers beta
directly, and the midpoint sits at 0.35 rather than 0.55, so the agents only
reach the responsive part of their speed law once consumption has pulled the
local oxygen much further down. Both push the instability to higher density: at
ambient air the predicted threshold is near 0.73 agents/area against 0.44 for
NPR1_LIKE.

The midpoint shift does most of the work, and that is a modelling choice worth
being explicit about. Weakening beta alone (a much wider `response_width` at the
same midpoint) also pushes the threshold up, but it pushes the whole strain so
close to marginal that the predicted wavelength exceeds 45 model units
everywhere it patterns, which would need a domain of roughly 280 units a side
and about 80,000 agents to classify honestly. The present N2_LIKE keeps the
predicted wavelength near 31-36 where it is solidly unstable, so the same
phase-diagram box serves both strains."""

CONSTANT_SPEED = Strain(
    name='CONSTANT_SPEED',
    v_min=0.63, v_max=0.63,
    oxygen_midpoint=0.55, response_width=0.045,
    rotational_diffusion=0.25,
    consumption=0.06, oxygen_half_saturation=0.10,
    kernel_width=1.2)
"""Control: V'(O) = 0 everywhere, so beta = 0 and no oxygen-driven instability
exists at any density or ambient level. The speed is fixed near the NPR1_LIKE
homogeneous equilibrium speed at Model 0's default density and ambient air, so
the persistence length is comparable and the comparison isolates beta. Agents
still consume oxygen, so the oxygen field is not trivially uniform -- only the
feedback onto motility is cut."""

HYPOXIC_BRANCH = Strain(
    name='HYPOXIC_BRANCH',
    v_min=0.06, v_max=2.0,
    oxygen_midpoint=0.55, response_width=0.09,
    rotational_diffusion=0.25,
    consumption=0.06, oxygen_half_saturation=0.10,
    kernel_width=1.2,
    speed_law='hypoxic', hypoxic_amplitude=1.2,
    hypoxic_midpoint=0.08, hypoxic_width=0.03)
"""Optional, off the main path: NPR1_LIKE plus the extra high-speed branch at
very low oxygen (Demir et al. 2020 Fig. 2b). Included so the non-monotone case
can be explored; none of the acceptance criteria use it, and `critical_density`
is only indicative for it because instability stops being monotone in density."""

PRESETS = {s.name: s for s in (NPR1_LIKE, N2_LIKE, CONSTANT_SPEED,
                               MODEL0_DEFAULT, HYPOXIC_BRANCH)}


def get_strain(name):
    """Look a preset up by name; a Strain instance passes straight through."""
    if isinstance(name, Strain):
        return name
    try:
        return PRESETS[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f'Choose a strain from {tuple(PRESETS)}') from exc


def info():
    """Print a one-line summary of every preset."""
    for strain in PRESETS.values():
        print(f'{strain.name:16s} v {strain.v_min:.2f}-{strain.v_max:.2f}  '
              f'midpoint {strain.oxygen_midpoint:.2f}  '
              f'width {strain.response_width:.3f}  law {strain.speed_law}')
