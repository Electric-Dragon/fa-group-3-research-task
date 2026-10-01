"""Figures for Model 1. Rendering never alters simulation state or RNG state.

Four families: the prediction (speed laws, dispersion relations, predicted
maps), the run (density, oxygen, structure factor, histories), the sweep
(phase diagrams with the predicted boundary drawn over the simulated labels),
and the time-lapse (how each morphology forms, from precomputed frames).
"""
import json
from io import BytesIO
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .engine import density_grid, DIAGNOSTIC_LENGTH
from .strains import get_strain
from .stability import growth_curve, predict, boundary_curve
from .patterns import structure_factor, LABELS

NAVY = '#18334a'
TEAL = '#087e8b'
ORANGE = '#d66b32'
PLUM = '#7d4a8f'

#: One colour per label, used identically in every phase diagram in the package.
LABEL_COLORS = {'uniform': '#dfe6ec', 'dots': '#f2b134',
                'stripes': '#087e8b', 'holes': '#7d2a4f'}


def _figure(figsize):
    fig = Figure(figsize=figsize, constrained_layout=True, facecolor='white')
    FigureCanvasAgg(fig)
    return fig


# -- prediction ----------------------------------------------------------

def _structure_panel(ax, rho, p, prediction=None, *, annotate=True):
    """S(k) on a log scale, cropped to the decades and wavenumbers that matter.

    The raw spectrum spans twenty decades because the smoothed field has almost
    no power at the grid scale; showing all of it hides the peak, so the view is
    clipped to four decades below the maximum and to a few times the peak k.
    """
    k, s = structure_factor(rho, p.dx, p.dy)
    ax.semilogy(k, s, color=TEAL, lw=1.8)
    peak = float(np.max(s)) if s.size else 1.0
    k_at_peak = float(k[int(np.argmax(s))]) if s.size else 1.0
    k_ref = prediction['k_star'] if (prediction and prediction['unstable']) else k_at_peak
    ax.set_xlim(0, min(float(k.max()), max(6 * k_ref, 4 * k_at_peak, 0.6)))
    ax.set_ylim(peak * 1e-4, peak * 4)
    if prediction and prediction['unstable']:
        ax.axvline(prediction['k_star'], color=ORANGE, ls='--', lw=1.6)
        if annotate:
            ax.text(prediction['k_star'], peak * 2,
                    f' predicted k* = {prediction["k_star"]:.3f}',
                    color=ORANGE, fontsize=8, va='top')
    ax.set(xlabel='k [rad / model length]', ylabel='S(k)')
    return k, s


def plot_speed_laws(strains, *, oxygen=None, mark=None):
    """V(O) and V'(O) for a list of strains, with optional equilibrium markers.

    `mark` is a mapping strain name -> oxygen level (typically O_eq), drawn as a
    dot on each curve: where the agents actually sit is what decides whether
    beta is large, and the curve alone does not show that.
    """
    strains = [get_strain(s) for s in strains]
    oxygen = np.linspace(0, 1, 400) if oxygen is None else np.asarray(oxygen)
    fig = _figure((10, 4.0))
    axes = fig.subplots(1, 2)
    for strain, color in zip(strains, (TEAL, ORANGE, PLUM, NAVY, '#9aa5b1')):
        axes[0].plot(oxygen, strain.speed(oxygen), color=color, lw=2, label=strain.name)
        axes[1].plot(oxygen, strain.speed_derivative(oxygen), color=color, lw=2)
        if mark and strain.name in mark:
            level = mark[strain.name]
            axes[0].plot([level], [strain.speed(level)], 'o', color=color, ms=7)
            axes[1].plot([level], [strain.speed_derivative(level)], 'o', color=color, ms=7)
    axes[0].set(xlabel='Oxygen / reference (1 = air)', ylabel='Speed V(O) [length/time]',
                title='Speed response')
    axes[1].set(xlabel='Oxygen / reference (1 = air)', ylabel="dV/dO",
                title="Slope: beta = V V' / (2 D_r) needs this positive")
    axes[1].axhline(0, color='0.6', lw=0.8)
    axes[0].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen agent parameters: what each strain does with local oxygen',
                 fontsize=13, color=NAVY, weight='bold')
    return fig


def plot_growth_curve(strain, env, densities, *, k_max=1.2, labels=None):
    """lambda(k) at several densities, with the marginal line at zero."""
    strain = get_strain(strain)
    k = np.linspace(1e-3, k_max, 800)
    fig = _figure((7.2, 4.4))
    ax = fig.subplots()
    densities = np.atleast_1d(densities)
    colors = [TEAL, ORANGE, PLUM, NAVY, '#9aa5b1', '#4a7f3f']
    for i, density in enumerate(densities):
        lam = growth_curve(strain, env, density, k)
        p = predict(strain, env, density)
        name = labels[i] if labels else f'W = {density:.2f}'
        ax.plot(k, lam, lw=2, color=colors[i % len(colors)], label=name)
        if p['unstable']:
            ax.plot([p['k_star']], [p['growth_rate']], 'o',
                    color=colors[i % len(colors)], ms=6)
    ax.axhline(0, color='0.35', lw=1)
    ax.set(xlabel='Wavenumber k [rad / model length]',
           ylabel='Growth rate lambda(k) [1 / model time]',
           title=f'{strain.name}: linear growth of a density ripple')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def plot_predicted_map(strain, predicted, *, title=None):
    """Predicted growth rate over (ambient, density) with the boundary drawn."""
    strain = get_strain(strain)
    ambients, densities = predicted['ambients'], predicted['densities']
    fig = _figure((10, 4.2))
    axes = fig.subplots(1, 2)
    extent = _extent(ambients, densities)
    growth = np.where(predicted['growth_rate'] > 0, predicted['growth_rate'], np.nan)
    im = axes[0].imshow(growth, origin='lower', extent=extent, aspect='auto', cmap='magma')
    axes[0].figure.colorbar(im, ax=axes[0], label='Max growth rate [1/time]')
    axes[0].set_title('Predicted growth rate (blank = stable)', loc='left',
                      color=NAVY, weight='bold')
    im = axes[1].imshow(predicted['wavelength'], origin='lower', extent=extent,
                        aspect='auto', cmap='viridis')
    axes[1].figure.colorbar(im, ax=axes[1], label='Wavelength [model length]')
    axes[1].set_title('Predicted wavelength', loc='left', color=NAVY, weight='bold')
    for ax in axes:
        ax.contour(ambients, densities, predicted['unstable'].astype(float),
                   levels=[0.5], colors='white', linewidths=1.6)
        ax.set(xlabel='Ambient oxygen / air', ylabel='Density [agents / area]')
    fig.suptitle(title or f'{strain.name}: linear stability prediction',
                 fontsize=13, color=NAVY, weight='bold')
    return fig


# -- one run -------------------------------------------------------------

def plot_result(result, *, title=None, prediction=None):
    """Six panels: oxygen, density, S(k), heterogeneity, environment, labels."""
    p, state = result.params, result.state
    rho = density_grid(state.positions, p, smoothing=DIAGNOSTIC_LENGTH)
    fig = _figure((12, 7.6))
    axes = fig.subplots(2, 3, gridspec_kw={'height_ratios': [1.3, 0.8]})
    _oxygen_panel(axes[0, 0], state, p)
    _density_panel(axes[0, 1], rho, p)

    prediction = prediction if prediction is not None else _auto_prediction(result)
    _structure_panel(axes[0, 2], rho, p, prediction)
    axes[0, 2].set_title('Structure factor', loc='left', color=NAVY, weight='bold')

    h = result.history
    axes[1, 0].plot(h['time'], h['heterogeneity'], color=TEAL, lw=2, label='H')
    if 'shot_noise_heterogeneity' in result.patterns:
        axes[1, 0].axhline(result.patterns.get('shot_noise_heterogeneity', np.nan),
                           color='0.6', ls=':', lw=1.2)
    axes[1, 0].set(title='Density heterogeneity', xlabel='Time [model time]',
                   ylabel='H = variance(rho / mean)', ylim=(0, None))
    axes[1, 1].plot(h['time'], h['mean_oxygen'], color=TEAL, lw=2, label='Mean oxygen')
    if 'mean_ambient' in h:
        axes[1, 1].plot(h['time'], h['mean_ambient'], color='0.5', lw=1.4, ls='--',
                        label='Ambient O_am')
    axes[1, 1].plot(h['time'], h['mean_speed'] / max(result.strain.max_speed, 1e-9),
                    color=ORANGE, lw=1.8, label='Mean speed / v_max')
    axes[1, 1].set(title='Environment and motility', xlabel='Time [model time]',
                   ylabel='Normalized value', ylim=(0, 1.05))
    axes[1, 1].legend(frameon=False, fontsize=8)
    if 'length_scale' in h:
        axes[1, 2].plot(h['time'], h['length_scale'], color=PLUM, lw=2,
                        label='Measured L = 2pi/k_peak')
        if prediction and prediction['unstable']:
            axes[1, 2].axhline(prediction['wavelength'], color=ORANGE, ls='--', lw=1.4,
                               label='Predicted wavelength')
        axes[1, 2].legend(frameon=False, fontsize=8)
    else:
        axes[1, 2].plot(h['time'], h['n_agents'], color=PLUM, lw=2)
    axes[1, 2].set(title='Length scale', xlabel='Time [model time]',
                   ylabel='Length [model length]')
    for ax in axes[1]:
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    label = result.patterns.get('label', 'n/a')
    fig.suptitle(title or (f'Model 1 · {result.strain.name} · {result.environment.label} · '
                           f'W = {result.density:.2f} · t = {state.time:.0f} · '
                           f'label: {label}'),
                 fontsize=13, color=NAVY, weight='bold')
    return fig


def _auto_prediction(result):
    try:
        return predict(result.strain, result.environment, result.density)
    except Exception:
        return None


def _oxygen_panel(ax, state, p, sample=4000):
    im = ax.imshow(state.oxygen, origin='lower', extent=(0, p.width, 0, p.height),
                   vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    # Deterministic thinning; plotting must never draw from the engine RNG.
    stride = max(1, p.n_agents // sample)
    ax.scatter(*state.positions[::stride].T, s=1.2, color='white', alpha=0.4, linewidths=0)
    ax.set(xlabel='x [model length]', ylabel='y [model length]')
    ax.set_title('Oxygen + agents', loc='left', color=NAVY, weight='bold')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='Oxygen / reference')


def _density_panel(ax, rho, p):
    im = ax.imshow(rho / rho.mean(), origin='lower', extent=(0, p.width, 0, p.height),
                   vmin=0, vmax=4, cmap='magma', interpolation='nearest')
    ax.set(xlabel='x [model length]', ylabel='y [model length]')
    ax.set_title('Coarse-grained density', loc='left', color=NAVY, weight='bold')
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                       label='Density / mean (capped at 4)')


# -- sweeps --------------------------------------------------------------

def plot_phase_diagram(summary, strain=None, *, title=None, replenishment=0.06,
                       show_boundary=True, ax=None):
    """Simulated labels over (ambient, density) with the predicted boundary.

    Marker colour is the majority label across seeds; opacity is the share of
    seeds that agreed, so a washed-out marker is a cell the seeds disputed.
    The solid line is `stability.boundary_curve`: the prediction, computed
    independently of anything that was simulated.
    """
    own = ax is None
    if own:
        fig = _figure((7.6, 5.6))
        ax = fig.subplots()
    else:
        fig = ax.figure
    labels, agreement = summary['label'], summary['agreement']
    ambient, density = summary['ambient'], summary['density']
    for lab in LABELS:
        mask = labels == lab
        if not mask.any():
            continue
        # Per-point RGBA rather than one global alpha, so seed disagreement
        # shows cell by cell.
        ax.scatter(ambient[mask], density[mask], s=210, marker='s',
                   c=[_rgba(LABEL_COLORS[lab], a) for a in agreement[mask]],
                   linewidths=0.6, edgecolors='white')
    if show_boundary and strain is not None:
        grid = np.linspace(np.nanmin(ambient), np.nanmax(ambient), 60)
        low, high = boundary_curve(strain, grid, replenishment=replenishment)
        # Shade the band rather than only drawing its two edges. The unstable
        # region is bounded both below and above, and at most ambient levels the
        # lower edge falls outside the swept density range -- so a reader shown
        # only the lines sees one curve through the middle of the data and reads
        # it as "the" boundary, which is the opposite of what it is.
        lo = np.where(np.isfinite(low), low, np.nan)
        hi = np.where(np.isfinite(high), high, np.nan)
        ax.fill_between(grid, lo, hi, color='#b9dbe0', alpha=0.45, zorder=0,
                        label='predicted unstable')
        ax.plot(grid, lo, color=NAVY, lw=2.0)
        ax.plot(grid, hi, color=NAVY, lw=2.0, ls='--')
        ax.set_ylim(np.nanmin(density) - 0.06, np.nanmax(density) + 0.06)
    handles = [Line2D([], [], marker='s', ls='', ms=10, color=LABEL_COLORS[l], label=l)
               for l in LABELS if (labels == l).any()]
    if show_boundary and strain is not None:
        handles += [Patch(facecolor='#b9dbe0', alpha=0.45, edgecolor=NAVY,
                          label='predicted unstable'),
                    Line2D([], [], color=NAVY, lw=2.0, label='predicted onset'),
                    Line2D([], [], color=NAVY, lw=2.0, ls='--', label='predicted upper edge')]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='upper left',
              bbox_to_anchor=(1.01, 1.0))
    ax.set(xlabel='Ambient oxygen / air (1.0 = 21%)',
           ylabel='Density [agents / model area]')
    ax.set_title(title or 'Simulated pattern vs predicted instability',
                 loc='left', color=NAVY, weight='bold')
    ax.grid(alpha=0.18)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def _rgba(hex_color, alpha):
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(hex_color)
    return (r, g, b, 0.35 + 0.65 * float(np.clip(alpha, 0, 1)))


def plot_phase_comparison(summaries, strains=None, *, replenishment=0.06, title=None):
    """Phase diagrams for several strains side by side on shared axes."""
    n = len(summaries)
    fig = _figure((6.6 * n, 5.4))
    axes = np.atleast_1d(fig.subplots(1, n))
    for ax, (name, summary) in zip(axes, summaries.items()):
        strain = (strains or {}).get(name, name)
        plot_phase_diagram(summary, strain, title=name, ax=ax,
                           replenishment=replenishment)
    fig.suptitle(title or 'Phase diagrams: simulation against linear prediction',
                 fontsize=13, color=NAVY, weight='bold')
    return fig


def plot_metric_map(summary, key, *, title=None, cmap='viridis', label=None):
    """Any numeric column of a sweep summary as an (ambient, density) image."""
    fig = _figure((6.6, 4.8))
    ax = fig.subplots()
    extent = _extent(summary['ambient'][0], summary['density'][:, 0])
    im = ax.imshow(summary[key], origin='lower', extent=extent, aspect='auto', cmap=cmap)
    fig.colorbar(im, ax=ax, label=label or key)
    ax.set(xlabel='Ambient oxygen / air', ylabel='Density [agents / area]')
    ax.set_title(title or key, loc='left', color=NAVY, weight='bold')
    return fig


def _extent(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    hx = 0.5 * (x[1] - x[0]) if x.size > 1 else 0.5
    hy = 0.5 * (y[1] - y[0]) if y.size > 1 else 0.5
    return (x[0] - hx, x[-1] + hx, y[0] - hy, y[-1] + hy)


def png_frame(result, **kwargs):
    """`plot_result` rendered to PNG bytes, for the widget image panel."""
    fig = plot_result(result, **kwargs)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=86)
    fig.clear()
    return buffer.getvalue()


def demo_frame(sim, predicted=None, *, sample=3500):
    """The four live panels of the interactive demo, as a PNG byte string.

    Density, oxygen with the ambient map behind it, the structure factor against
    the predicted k*, and a small (ambient, density) map with the current
    operating point on it. `predicted` is a cached `predicted_map` so the demo
    does not recompute a 2D stability scan every frame.
    """
    p = sim.params
    rho = density_grid(sim.state.positions, p, smoothing=DIAGNOSTIC_LENGTH)
    metrics = _demo_metrics(rho, p)
    prediction = predict(sim.strain, sim.environment, sim.density)
    fig = _figure((12.2, 6.4))
    axes = fig.subplots(2, 3, gridspec_kw={'height_ratios': [1.0, 1.0]})
    gs = axes[0, 0].get_gridspec()
    for ax in axes[:, 2]:
        ax.remove()
    tall = fig.add_subplot(gs[:, 2])

    _density_panel(axes[0, 0], rho, p)
    _oxygen_panel(axes[0, 1], sim.state, p, sample=sample)

    _structure_panel(axes[1, 0], rho, p, prediction)
    axes[1, 0].set_title(f'Structure factor · measured L = {metrics["length_scale"]:.1f}',
                         loc='left', color=NAVY, weight='bold', fontsize=10)

    ambient_map = sim.ambient_field
    if np.isscalar(ambient_map):
        axes[1, 1].axhline(float(ambient_map), color=TEAL, lw=2)
        axes[1, 1].set_ylim(0, 1.05)
        axes[1, 1].set(xlabel='(ambient is uniform)', ylabel='O_am / air')
    else:
        im = axes[1, 1].imshow(ambient_map, origin='lower', vmin=0, vmax=1,
                               extent=(0, p.width, 0, p.height), cmap='cividis')
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.03, label='O_am / air')
        axes[1, 1].set(xlabel='x [model length]', ylabel='y [model length]')
    axes[1, 1].set_title('Ambient oxygen you are imposing', loc='left',
                         color=NAVY, weight='bold', fontsize=10)

    if predicted is not None:
        extent = _extent(predicted['ambients'], predicted['densities'])
        tall.imshow(predicted['unstable'].astype(float), origin='lower', extent=extent,
                    aspect='auto', cmap=ListedColormap(['#eef2f5', '#b9dbe0']),
                    norm=BoundaryNorm([0, 0.5, 1], 2))
        tall.contour(predicted['ambients'], predicted['densities'],
                     predicted['unstable'].astype(float), levels=[0.5],
                     colors=[NAVY], linewidths=1.8)
    mean_ambient = float(np.mean(ambient_map))
    tall.plot([mean_ambient], [sim.density], marker='*', ms=20, color=ORANGE,
              markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    tall.set(xlabel='Ambient oxygen / air', ylabel='Density [agents / area]')
    tall.set_title('Where you are\n(shaded = predicted unstable)', loc='left',
                   color=NAVY, weight='bold', fontsize=10)
    for ax in (axes[1, 0], tall):
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(
        f'{sim.strain.name} · {sim.environment.label} · N = {p.n_agents} '
        f'(W = {sim.density:.2f}) · t = {sim.time:.0f} · label: {metrics["label"]}'
        + ('  · predicted: unstable' if prediction['unstable'] else '  · predicted: stable'),
        fontsize=12.5, color=NAVY, weight='bold')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=80)
    fig.clear()
    return buffer.getvalue(), metrics, prediction


def _demo_metrics(rho, p):
    from .patterns import classify
    return classify(rho, p.dx, p.dy, smoothing=DIAGNOSTIC_LENGTH)


# -- time-lapse ----------------------------------------------------------
#
# The frames are precomputed by scripts/timelapse_figure.py and stored in
# assets/timelapse.json, because watching four conditions form takes about
# 100,000 turns in total -- far too long for a notebook cell. The stored
# frames are relative density (rho / mean rho) clipped to [0, 3] and
# quantised to a byte, which is all the colour ramp resolves anyway.

TIMELAPSE_VMAX = 3.0


def load_timelapse(path=None):
    """Read the precomputed time-lapse payload and decode its frames.

    Returns the payload with each case's `frames` replaced by a stacked float
    array of relative density, shaped (n_times, grid, grid).
    """
    if path is None:
        path = Path(__file__).resolve().parents[2] / 'assets' / 'timelapse.json'
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found - run: python scripts/timelapse_figure.py')
    payload = json.loads(path.read_text())
    grid = int(payload['grid'])
    for case in payload['cases']:
        frames = np.array(case['frames'], dtype=float) / 255.0 * TIMELAPSE_VMAX
        case['frames'] = frames.reshape(-1, grid, grid)
    return payload


def _timelapse_panel(ax, field, label, title=None, grid=None):
    """One density snapshot with the classifier's verdict banded underneath."""
    grid = field.shape[0] if grid is None else grid
    ax.imshow(field, origin='lower', vmin=0, vmax=TIMELAPSE_VMAX, cmap='magma',
              interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8.5, color=NAVY, pad=3)
    ax.add_patch(Patch(color=LABEL_COLORS[label]))          # placeholder, replaced below
    ax.patches[-1].remove()
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), grid, grid * 0.085,
                           color=LABEL_COLORS[label], zorder=5))
    ax.text(grid / 2, grid * 0.042, label, ha='center', va='center',
            fontsize=7.5, zorder=6,
            color='white' if label in ('stripes', 'holes') else '#222')


def plot_timelapse(payload=None, *, shown=(0, 2, 4, 6, 8, 11)):
    """The static grid: one row per condition, one column per sampled time.

    This is the version that survives with no kernel running, so it is what the
    notebook shows first and what `assets/timelapse.png` holds.
    """
    payload = load_timelapse() if payload is None else payload
    cases, times = payload['cases'], payload['times']
    shown = [i for i in shown if i < len(times)]
    rows, cols = len(cases), len(shown)
    fig = _figure((1.75 * cols + 2.1, 1.75 * rows + 1.0))
    axes = fig.subplots(rows, cols, squeeze=False)
    for r, case in enumerate(cases):
        for c, i in enumerate(shown):
            _timelapse_panel(axes[r][c], case['frames'][i], case['labels'][i],
                             title=f't = {times[i]}')
        axes[r][0].set_ylabel(f"{case['name']}\nW = {case['density']:.2f}\n"
                              f"N = {case['n_agents']:,}",
                              fontsize=9, color=NAVY, rotation=0,
                              ha='right', va='center', labelpad=50)
    fig.suptitle('Pattern formation over time  ·  same strain, same air, different '
                 "density  ·  band under each frame = the classifier's label",
                 fontsize=12.5, color=NAVY, weight='bold')
    return fig


def plot_timelapse_at(payload, index):
    """All conditions at one instant, side by side, with the H trace beneath.

    The panel the interactive scrubber redraws as the time slider moves.
    """
    cases, times = payload['cases'], payload['times']
    fig = _figure((3.1 * len(cases), 4.5))
    grid = fig.add_gridspec(2, len(cases), height_ratios=(1.0, 0.45))
    for c, case in enumerate(cases):
        ax = fig.add_subplot(grid[0, c])
        _timelapse_panel(ax, case['frames'][index], case['labels'][index])
        ax.set_title(f"{case['name']}  ·  W = {case['density']:.2f}",
                     fontsize=10, color=NAVY, weight='bold', loc='left')
    trace = fig.add_subplot(grid[1, :])
    colors = (ORANGE, TEAL, PLUM, '#9aa5b1')
    for case, color in zip(cases, colors):
        trace.plot(times, case['heterogeneity'], color=color, lw=1.8,
                   label=case['name'])
        trace.plot([times[index]], [case['heterogeneity'][index]], 'o',
                   color=color, ms=7)
    trace.axvline(times[index], color=NAVY, lw=1.2, ls='--')
    trace.set(xlabel='Time [model time]', ylabel='Heterogeneity H', ylim=(0, None))
    trace.legend(frameon=False, fontsize=8, ncol=len(cases))
    trace.grid(alpha=0.18)
    trace.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Density at t = {times[index]}', fontsize=12.5, color=NAVY,
                 weight='bold')
    return fig


def timelapse_frame(payload, index):
    """`plot_timelapse_at` as PNG bytes, for the scrubber's image widget."""
    fig = plot_timelapse_at(payload, index)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=84)
    fig.clear()
    return buffer.getvalue()
