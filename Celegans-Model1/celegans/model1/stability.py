"""Linear stability of the homogeneous state: where patterns should appear.

This is the prediction half of Model 1. It never runs a simulation; it
linearises the mean-field limit of the agent rule and asks whether a small
density ripple grows.

The coarse-grained equations (Demir et al. 2020, Methods) are

    dW/dt = div[ D_W grad W ] + div[ beta W grad O ]
    dO/dt = D_O lap O + f (O_am - O) - q W_s O / (K + O)

with D_W = V(O)^2 / (2 D_r) and beta = V(O) V'(O) / (2 D_r), where D_r is the
rotational diffusion rate (the paper's "tau", which is a rate, not a time).
W_s is the density seen by the oxygen field, i.e. W convolved with the Gaussian
consumption footprint, which contributes the factor g(k) below.

Linearising W = W0 + w e^{ikx + lt}, O = O_eq + o e^{ikx + lt}:

    M(k) = [[ -D_W k^2,   -beta W k^2        ],
            [ -a g(k),    -(D_O k^2 + f + b) ]]

    a = q O_eq / (K + O_eq)        d(sink)/dW
    b = q W K / (K + O_eq)^2       d(sink)/dO
    g(k) = exp(-sigma^2 k^2 / 2)   Fourier transform of the footprint

lambda(k) is the larger eigenvalue; the state is unstable when max_k Re lambda > 0,
equivalently when for some k

    D_W (D_O k^2 + f + b) < beta W a g(k).

Note the structure: the destabilising term needs beta > 0, i.e. V'(O_eq) > 0.
Agents must sit on the **rising** branch of the speed law. On the flat plateau
at high ambient, or for a constant-speed strain, the right-hand side vanishes
and no wavenumber is unstable at any density.

Limits of this prediction, all of which matter near the boundary:

* It is a drift-diffusion limit. It needs the pattern wavelength to be much
  larger than the persistence length V/D_r; `predict` reports the ratio and
  flags it below 5.
* It is linear. It says where a ripple starts to grow, not what it grows into,
  and nothing here predicts dots versus stripes versus holes -- that is set by
  density and is measured, not predicted (see `patterns.py`).
* It is deterministic and infinite-N. A finite number of agents adds noise that
  both seeds growth early and destroys marginal patterns; agreement should be
  expected away from the boundary and fuzz on it.
"""
from __future__ import annotations

import numpy as np

from .strains import get_strain
from .environment import Environment


def equilibrium_oxygen(strain, ambient, replenishment, density):
    """O_eq from f (A - O) = q W O / (K + O), the homogeneous reaction balance.

    The same algebra as Model 0's `uniform_oxygen`, written against a strain and
    a density rather than a full `Parameters`, and vectorised so a map of
    ambient levels or replenishment rates gives the matching map of equilibria.
    The branch choice avoids catastrophic cancellation for small K.
    """
    strain = get_strain(strain)
    q, K = strain.consumption, strain.oxygen_half_saturation
    A = np.asarray(ambient, dtype=float)
    f = np.asarray(replenishment, dtype=float)
    W = np.asarray(density, dtype=float)
    scalar = A.ndim == 0 and f.ndim == 0 and W.ndim == 0
    A, f, W = np.broadcast_arrays(A, f, W)
    out = np.array(A, dtype=float)                     # q == 0 or W == 0 -> A
    active = (q != 0) & (W != 0)
    starved = active & (f == 0)                        # no supply -> fully consumed
    solve = active & (f != 0)
    if solve.any():
        b = K - A[solve] + q * W[solve] / f[solve]
        c = K * A[solve]
        root = np.sqrt(b * b + 4 * c)
        safe = (b >= 0) & (root + b > 0)
        value = np.where(safe, 2 * c / np.where(safe, root + b, 1.0), (root - b) / 2)
        out[solve] = value
    out[starved] = 0.0
    return float(out) if scalar else out
def coefficients(strain, ambient, replenishment, density, oxygen_diffusion=1.0):
    """Everything the dispersion relation needs, at the homogeneous state."""
    strain = get_strain(strain)
    O_eq = equilibrium_oxygen(strain, ambient, replenishment, density)
    V = float(strain.speed(O_eq))
    dV = float(strain.speed_derivative(O_eq))
    D_r = strain.rotational_diffusion
    if D_r <= 0:
        raise ValueError('rotational_diffusion must be positive for the diffusive limit')
    K, q = strain.oxygen_half_saturation, strain.consumption
    return dict(
        O_eq=O_eq, V=V, dV=dV,
        D_W=V * V / (2 * D_r),
        beta=V * dV / (2 * D_r),
        a=q * O_eq / (K + O_eq),
        b=q * density * K / (K + O_eq) ** 2,
        f=float(replenishment), D_O=float(oxygen_diffusion),
        sigma=strain.kernel_width, W=float(density))


def growth_curve(strain, env, density, k):
    """lambda(k): the largest real part of the eigenvalues of M(k).

    `k` may be a scalar or an array of wavenumbers (radians per model length).
    """
    if isinstance(env, Environment):
        ambient, f = env.representative()
        D_O = env.oxygen_diffusion
    else:                                    # (ambient, replenishment) pair
        ambient, f = env
        D_O = 1.0
    c = coefficients(strain, ambient, f, density, D_O)
    k = np.asarray(k, dtype=float)
    k2 = k * k
    g = np.exp(-c['sigma'] ** 2 * k2 / 2)
    m11 = -c['D_W'] * k2
    m22 = -(c['D_O'] * k2 + c['f'] + c['b'])
    m12 = -c['beta'] * c['W'] * k2
    m21 = -c['a'] * g
    trace = m11 + m22
    det = m11 * m22 - m12 * m21
    disc = trace * trace - 4 * det
    # Complex pair -> both roots share Re = trace/2, which is negative here.
    root = np.sqrt(np.maximum(disc, 0.0))
    return 0.5 * (trace + root)


def predict(strain, env, density, *, k_max=None, n_k=2001):
    """Scan lambda(k) and report the fastest-growing mode.

    Returns a dict with `unstable`, `k_star`, `wavelength`, `growth_rate`, the
    homogeneous state (`O_eq`, `V`, `dV`), the transport coefficients (`D_W`,
    `beta`) and the validity flags described in the module docstring.
    """
    strain = get_strain(strain)
    if isinstance(env, Environment):
        ambient, f = env.representative()
        D_O = env.oxygen_diffusion
        inhomogeneous = (not np.isscalar(env.ambient)) or env.time_varying \
            or np.asarray(env.replenishment).ndim > 0
    else:
        ambient, f = env
        D_O, inhomogeneous = 1.0, False
    c = coefficients(strain, ambient, f, density, D_O)
    if k_max is None:
        # Beyond a few footprint widths g(k) has killed the coupling; the scan
        # only has to cover where instability can live.
        k_max = max(10.0 / max(c['sigma'], 1e-6), 5.0)
    k = np.linspace(1e-6, k_max, int(n_k))
    lam = growth_curve(strain, (ambient, f) if not isinstance(env, Environment) else env,
                       density, k)
    i = int(np.argmax(lam))
    k_star, growth = float(k[i]), float(lam[i])
    unstable = growth > 0
    wavelength = 2 * np.pi / k_star if k_star > 0 else np.inf
    persistence = c['V'] / strain.rotational_diffusion
    ratio = wavelength / persistence if persistence > 0 else np.inf
    return dict(
        unstable=bool(unstable),
        k_star=k_star if unstable else float('nan'),
        wavelength=float(wavelength) if unstable else float('nan'),
        growth_rate=growth,
        O_eq=c['O_eq'], V=c['V'], dV=c['dV'], D_W=c['D_W'], beta=c['beta'],
        a=c['a'], b=c['b'], density=float(density), ambient=float(ambient),
        replenishment=float(f), oxygen_diffusion=float(D_O),
        persistence_length=float(persistence),
        wavelength_over_persistence=float(ratio) if unstable else float('nan'),
        diffusive_limit_ok=bool(unstable and ratio >= 5.0),
        homogeneous_reference=not inhomogeneous,
        strain=strain.name)


def predicted_map(strain, ambients, densities, *, replenishment=0.06,
                  oxygen_diffusion=1.0):
    """Instability over an (ambient, density) grid.

    Returns dicts of 2D arrays indexed [i_density, i_ambient] -- the same
    orientation the phase diagram is drawn in -- holding `unstable` (bool),
    `growth_rate`, `wavelength` and `O_eq`.
    """
    ambients = np.atleast_1d(np.asarray(ambients, dtype=float))
    densities = np.atleast_1d(np.asarray(densities, dtype=float))
    shape = (densities.size, ambients.size)
    out = dict(unstable=np.zeros(shape, dtype=bool),
               growth_rate=np.zeros(shape), wavelength=np.full(shape, np.nan),
               O_eq=np.zeros(shape))
    for j, A in enumerate(ambients):
        for i, W in enumerate(densities):
            p = predict(strain, (A, replenishment), W)
            for key in out:
                out[key][i, j] = p[key]
    out['ambients'], out['densities'] = ambients, densities
    return out


def critical_density(strain, ambient, *, replenishment=0.06, oxygen_diffusion=1.0,
                     lo=1e-3, hi=5.0, samples=200, tol=1e-4):
    """Lowest density at which the homogeneous state is unstable, or NaN.

    Instability is **not** monotone in density: raising the density lowers O_eq,
    which first brings the agents onto the rising branch of V(O) and then takes
    them off it again onto the lower plateau, where V' -- and with it beta --
    dies. So this coarse-scans [lo, hi] for the first unstable sample and then
    bisects the bracket around it, rather than bisecting the whole interval.
    `unstable_density_range` returns the whole window.
    """
    edges = unstable_density_range(strain, ambient, replenishment=replenishment,
                                   oxygen_diffusion=oxygen_diffusion,
                                   lo=lo, hi=hi, samples=samples, tol=tol)
    return edges[0]


def unstable_density_range(strain, ambient, *, replenishment=0.06,
                           oxygen_diffusion=1.0, lo=1e-3, hi=5.0,
                           samples=200, tol=1e-4):
    """(lowest, highest) unstable density at fixed ambient, or (NaN, NaN).

    The scan is on a log grid because the lower edge is usually much sharper
    than the upper one. A window narrower than the grid spacing can be missed;
    increase `samples` if a boundary looks ragged.
    """
    def unstable(W):
        return predict(strain, (ambient, replenishment), W,
                       )['unstable'] if W > 0 else False

    grid = np.geomspace(lo, hi, int(samples))
    flags = np.array([unstable(W) for W in grid])
    if not flags.any():
        return float('nan'), float('nan')
    first, last = int(np.argmax(flags)), int(len(flags) - 1 - np.argmax(flags[::-1]))

    def refine(a, b, target_at_b):
        """Bisect the sign change between a (not target) and b (target)."""
        while b - a > tol:
            mid = 0.5 * (a + b)
            if unstable(mid) == target_at_b:
                b = mid
            else:
                a = mid
        return b

    low = grid[first] if first == 0 else refine(grid[first - 1], grid[first], True)
    high = grid[last] if last == len(grid) - 1 else refine(grid[last + 1], grid[last], True)
    return float(low), float(high)


def boundary_curve(strain, ambients, *, replenishment=0.06, oxygen_diffusion=1.0,
                   lo=1e-3, hi=5.0, samples=200):
    """Lower and upper instability boundaries in density for each ambient level.

    Two arrays suitable for plotting straight onto a phase diagram. NaN marks
    ambient levels where the strain never patterns at any density.
    """
    ambients = np.atleast_1d(np.asarray(ambients, dtype=float))
    low = np.empty(ambients.size)
    high = np.empty(ambients.size)
    for i, A in enumerate(ambients):
        low[i], high[i] = unstable_density_range(
            strain, A, replenishment=replenishment,
            oxygen_diffusion=oxygen_diffusion, lo=lo, hi=hi, samples=samples)
    return low, high
