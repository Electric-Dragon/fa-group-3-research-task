"""Automatic pattern classification from the coarse-grained density field.

The classifier answers "what does this look like", which is a different
question from "is the homogeneous state unstable" (`stability.py`). Linear
theory predicts *whether* a ripple grows and at *what* wavelength; it says
nothing about whether the grown state is dots, stripes or holes. That is set by
how much of the area the dense phase occupies, which grows with density, and it
is measured here rather than predicted.

Four numbers do the work:

* **Heterogeneity H** = variance of (rho / mean rho), Model 0's measure, on the
  field smoothed at the fixed diagnostic length. High H alone does **not**
  establish phase coexistence -- a noisy field has high H too -- which is why
  H is compared against the shot-noise level a Poisson point cloud of the same
  density would produce, not against a bare constant.
* **Structure factor** S(k) of the density fluctuation, radially averaged. Its
  peak gives a characteristic length L = 2 pi / k_peak to compare against the
  predicted wavelength.
* **Dense-phase area fraction** phi, the fraction of cells above a threshold.
  The threshold is **Otsu's** (the mean is available as `threshold='mean'`).
  The mean is the tempting choice because it assumes nothing, but it is a
  systematically biased separator: in a phase-separated system the two
  coexisting densities sit either side of the mean at a spacing set by the
  lever rule, so thresholding at the mean forces phi towards 0.5 whenever the
  density contrast is symmetric, and misplaces the boundary whenever it is not.
  In this model the contrast is strongly asymmetric at high density -- voids
  are deeply depleted while the dense phase is only modestly above average --
  so the mean falls *inside* the dense phase and phi is under-reported by
  0.1-0.2. Otsu picks the threshold that best separates the two populations and
  does not have that bias. See VALIDATION.md for the measured difference.
* **Topology on the torus**: the number of connected components of the dense
  phase (n_dense) and of the dilute phase (n_dilute), with periodic wrap
  handled, and chi = n_dense - n_dilute. A stripe wrapping the torus is one
  component, not two, which is the whole reason the wrap has to be handled.

The label rule and its thresholds are heuristics. They are returned in the
output dict alongside every raw number, so a label can be recomputed later with
different thresholds without rerunning anything.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

#: Dense-phase area fraction below which a non-percolating dense phase is 'dots'.
DOTS_AREA_FRACTION = 0.35
#: Dense-phase area fraction above which a non-percolating dilute phase is 'holes'.
HOLES_AREA_FRACTION = 0.65
#: H must exceed this multiple of the Poisson shot-noise level to be a pattern.
SHOT_NOISE_FACTOR = 3.0

LABELS = ('uniform', 'dots', 'stripes', 'holes')


def shot_noise_heterogeneity(density, smoothing):
    """H that an uncorrelated Poisson cloud of this density would show.

    For density W smoothed by a normalised Gaussian of width sigma, the variance
    of the smoothed field is W * integral(g^2) = W / (4 pi sigma^2), so
    H = variance / W^2 = 1 / (4 pi sigma^2 W). At Model 0's defaults
    (W = 0.586, sigma = 2.0) this gives 0.034, matching the 0.026-0.036 that
    Model 0 reports for its initial uniform conditions.
    """
    density, smoothing = float(density), float(smoothing)
    if density <= 0 or smoothing <= 0:
        return np.inf
    return 1.0 / (4 * np.pi * smoothing ** 2 * density)


def structure_factor(rho, dx, dy, *, n_bins=None):
    """Radially averaged S(k) of the density fluctuation.

    Returns (k, S) with k in radians per model length, excluding k = 0. Bins are
    one fundamental wavenumber wide, which is the finest the box can resolve,
    and each is reported at the mean |k| of the modes it holds.
    """
    ny, nx = rho.shape
    delta = rho - rho.mean()
    power = np.abs(np.fft.fft2(delta)) ** 2 / (nx * ny)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kmag = np.hypot(*np.meshgrid(kx, ky))
    fundamental = 2 * np.pi / max(nx * dx, ny * dy)
    if n_bins is None:
        n_bins = int(np.ceil(kmag.max() / fundamental))
    edges = np.arange(n_bins + 1) * fundamental
    which = np.digitize(kmag.ravel(), edges) - 1
    valid = (which >= 0) & (which < n_bins)
    counts = np.bincount(which[valid], minlength=n_bins)
    totals = np.bincount(which[valid], weights=power.ravel()[valid], minlength=n_bins)
    # Represent each bin by the mean |k| of the modes in it rather than by the
    # bin centre: the modes of a 2D grid are not spread evenly across a radial
    # bin, and the centre biases the recovered wavelength by several percent.
    radii = np.bincount(which[valid], weights=kmag.ravel()[valid], minlength=n_bins)
    keep = counts > 0
    centres = np.where(keep, radii / np.maximum(counts, 1), 0.0)
    keep &= centres > 0
    return centres[keep], totals[keep] / counts[keep]


def peak_wavenumber(k, s):
    """(k_peak, length_scale) from the strongest radial mode, or (nan, nan).

    k_peak can only take the discrete radial-bin values, so a length-scale
    history steps between them rather than varying smoothly, and a single
    measurement carries the bin width as its resolution (about 10% -- see
    VALIDATION.md). Sub-bin interpolation of the peak would sharpen it, and was
    deliberately left out: every phase-diagram sweep in `results/` was measured
    with this estimator, and one consistent estimator across all of them is
    worth more than a few percent on any one of them.
    """
    if k.size == 0 or not np.any(s > 0):
        return float('nan'), float('nan')
    i = int(np.argmax(s))
    k_peak = float(k[i])
    return k_peak, (2 * np.pi / k_peak if k_peak > 0 else float('nan'))


def periodic_label(mask, connectivity=2):
    """Connected components of a boolean mask on a torus.

    `scipy.ndimage.label` has no periodic mode, so components are found on the
    open grid and then merged across opposite edges with a union-find. Without
    this a single stripe wrapping the domain is counted as two components and
    every classification downstream is wrong.

    connectivity 1 is the four-cell cross, 2 the eight-cell square. The dense
    phase is labelled with 2 and the dilute phase with 1: using the same
    connectivity for both would let a diagonal chain be simultaneously connected
    in the foreground and in the background.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return labels, 0
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    offsets = (0,) if connectivity == 1 else (-1, 0, 1)
    for shift in offsets:                       # top row against bottom row
        a, b = labels[0], np.roll(labels[-1], shift)
        for i, j in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            union(int(i), int(j))
    for shift in offsets:                       # left column against right column
        a, b = labels[:, 0], np.roll(labels[:, -1], shift)
        for i, j in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            union(int(i), int(j))
    roots = np.array([find(i) for i in range(n + 1)])
    roots[0] = 0
    unique = np.unique(roots[1:])
    remap = np.zeros(n + 1, dtype=np.int64)
    remap[unique] = np.arange(1, unique.size + 1)
    return remap[roots][labels], int(unique.size)


def percolates(labels, count):
    """True if any component spans the full domain in x or in y.

    A wrapping stripe touches every row (or every column), and this is the test
    used for it. It is a heuristic: a very wide blob that happens to reach every
    column also passes, so it is only trusted together with the area fraction.
    """
    ny, nx = labels.shape
    for i in range(1, count + 1):
        ys, xs = np.nonzero(labels == i)
        if np.unique(xs).size == nx or np.unique(ys).size == ny:
            return True
    return False


def _otsu(values):
    """Otsu's threshold on a 1D array of densities."""
    counts, edges = np.histogram(values, bins=256)
    centres = 0.5 * (edges[1:] + edges[:-1])
    weight1 = np.cumsum(counts)
    weight2 = counts.sum() - weight1
    ok = (weight1 > 0) & (weight2 > 0)
    if not ok.any():
        return float(values.mean())
    mean1 = np.cumsum(counts * centres) / np.maximum(weight1, 1)
    total = (counts * centres).sum()
    mean2 = (total - np.cumsum(counts * centres)) / np.maximum(weight2, 1)
    variance = weight1 * weight2 * (mean1 - mean2) ** 2
    variance[~ok] = -np.inf
    return float(centres[int(np.argmax(variance))])


def classify(rho, dx, dy, *, smoothing=2.0, threshold='otsu',
             dots_area_fraction=DOTS_AREA_FRACTION,
             holes_area_fraction=HOLES_AREA_FRACTION,
             shot_noise_factor=SHOT_NOISE_FACTOR):
    """Label a coarse-grained density field and return every number behind it.

    `rho` must already be smoothed at `smoothing` (the engine uses the fixed
    diagnostic length of 2.0, the same one Model 0 reports H at), because the
    shot-noise reference and the component counts both depend on it.

    The rule, in order:

    1. H at or below `shot_noise_factor` times the Poisson level -> 'uniform'.
       Nothing beyond the fluctuation a random cloud would show.
    2. Dense phase not percolating and phi <= `dots_area_fraction` -> 'dots'.
    3. Dilute phase not percolating and phi >= `holes_area_fraction` -> 'holes'.
    4. Otherwise -> 'stripes' (which covers labyrinths; the classifier does not
       distinguish straight stripes from a disordered bicontinuous state).
    """
    rho = np.asarray(rho, dtype=float)
    mean = float(rho.mean())
    heterogeneity = float(np.mean((rho / mean - 1) ** 2)) if mean > 0 else 0.0
    baseline = shot_noise_heterogeneity(mean, smoothing)
    k, s = structure_factor(rho, dx, dy)
    k_peak, length_scale = peak_wavenumber(k, s)

    if threshold == 'otsu':
        cut = _otsu(rho.ravel())
    elif threshold == 'mean':
        cut = mean
    else:
        raise ValueError("threshold must be 'otsu' or 'mean'")
    dense = rho > cut
    area_fraction = float(dense.mean())
    dense_labels, n_dense = periodic_label(dense, connectivity=2)
    dilute_labels, n_dilute = periodic_label(~dense, connectivity=1)
    dense_perc = percolates(dense_labels, n_dense)
    dilute_perc = percolates(dilute_labels, n_dilute)

    if heterogeneity <= shot_noise_factor * baseline:
        label = 'uniform'
    elif not dense_perc and area_fraction <= dots_area_fraction:
        label = 'dots'
    elif not dilute_perc and area_fraction >= holes_area_fraction:
        label = 'holes'
    else:
        label = 'stripes'

    return dict(
        label=label,
        heterogeneity=heterogeneity,
        shot_noise_heterogeneity=float(baseline),
        heterogeneity_ratio=float(heterogeneity / baseline) if np.isfinite(baseline) else np.inf,
        contrast=float(np.sqrt(heterogeneity)),
        k_peak=k_peak, length_scale=length_scale,
        area_fraction=area_fraction, threshold=float(cut),
        n_dense=int(n_dense), n_dilute=int(n_dilute),
        euler=int(n_dense - n_dilute),
        dense_percolates=bool(dense_perc), dilute_percolates=bool(dilute_perc),
        settings=dict(smoothing=float(smoothing), threshold=threshold,
                      dots_area_fraction=float(dots_area_fraction),
                      holes_area_fraction=float(holes_area_fraction),
                      shot_noise_factor=float(shot_noise_factor)))


def coarsening_exponent(times, lengths, *, late_fraction=0.5, min_points=4):
    """Fit L(t) ~ t^n on a log-log plot over the late part of a history.

    Returns (exponent, intercept, n_points) or (nan, nan, n) when there are too
    few usable points. Only the last `late_fraction` of the record is used, so
    the early transient while the pattern is still forming does not enter, and
    a single fitted exponent from one run is an estimate, not a measurement of
    a growth law.
    """
    times = np.asarray(times, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    ok = np.isfinite(times) & np.isfinite(lengths) & (times > 0) & (lengths > 0)
    times, lengths = times[ok], lengths[ok]
    if times.size < min_points:
        return float('nan'), float('nan'), int(times.size)
    start = int(times.size * (1 - late_fraction))
    times, lengths = times[start:], lengths[start:]
    if times.size < min_points or np.ptp(times) <= 0:
        return float('nan'), float('nan'), int(times.size)
    slope, intercept = np.polyfit(np.log(times), np.log(lengths), 1)
    return float(slope), float(intercept), int(times.size)
