"""Matplotlib figures and PNG frames; rendering never alters simulation state."""
from io import BytesIO
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from .model import density_grid
from .analysis import DIAGNOSTIC_LENGTH

NAVY = '#18334a'
TEAL = '#087e8b'
ORANGE = '#d66b32'


def _figure(figsize):
    fig = Figure(figsize=figsize, constrained_layout=True, facecolor='white')
    FigureCanvasAgg(fig)
    return fig


def _map(ax, state, p, kind='oxygen', headings=True):
    if kind == 'oxygen':
        field, label, cmap, vmax = state.oxygen, 'Oxygen / reference', 'viridis', 1
    else:
        rho = density_grid(state.positions, p, smoothing=DIAGNOSTIC_LENGTH)
        field, label, cmap, vmax = rho/rho.mean(), 'Density / mean (scale capped at 4)', 'magma', 4
    im = ax.imshow(field, origin='lower', extent=(0, p.width, 0, p.height),
                   vmin=0, vmax=vmax, cmap=cmap, interpolation='nearest')
    if kind == 'oxygen':
        ax.scatter(*state.positions.T, s=2, color='white', alpha=0.48, linewidths=0)
        if headings:
            # Deterministic subset for legibility; these dashes are headings, not bodies.
            stride = max(1, p.n_agents // 180)
            pos, theta = state.positions[::stride], state.headings[::stride]
            ends = pos + 0.7*np.column_stack((np.cos(theta), np.sin(theta)))
            ax.add_collection(LineCollection(np.stack((pos, ends), axis=1),
                                             colors='white', linewidths=0.5, alpha=0.8))
    ax.set(xlabel='x [model length]', ylabel='y [model length]',
           xlim=(0, p.width), ylim=(0, p.height))
    ax.set_title('Oxygen + agents' if kind == 'oxygen' else 'Coarse-grained density',
                 loc='left', color=NAVY, weight='bold')
    ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label=label)


def plot_result(result, *, title=None):
    """Four-panel scientific view; accepts a Result from run() or sim.result()."""
    fig = _figure((11, 7.2))
    axes = fig.subplots(2, 2, gridspec_kw={'height_ratios': [1.25, 0.75]})
    _map(axes[0, 0], result.state, result.params)
    _map(axes[0, 1], result.state, result.params, 'density')
    h = result.history
    axes[1, 0].plot(h['time'], h['heterogeneity'], color=TEAL, lw=2)
    axes[1, 0].set(title='Density heterogeneity', xlabel='Time [model time]',
                   ylabel='H = variance(density / mean)', ylim=(0, None))
    axes[1, 1].plot(h['time'], h['mean_oxygen'], label='Mean oxygen', color=TEAL, lw=2)
    # Normalize against the fixed speed ceiling for this result's parameter set.
    axes[1, 1].plot(h['time'], h['mean_speed']/result.params.v_max,
                   label='Mean speed / current v_max', color=ORANGE, lw=1.8)
    axes[1, 1].set(title='Environment and motility', xlabel='Time [model time]',
                   ylabel='Normalized value', ylim=(0, 1.05))
    axes[1, 1].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.grid(alpha=0.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title or f'Model 0 · {result.scenario} · t = {result.state.time:.1f} · seed {result.seed}',
                 fontsize=14, color=NAVY, weight='bold')
    return fig


def plot_comparison(results):
    """Mapping label -> Result; final relative-density maps and common history."""
    n = len(results)
    if not 1 <= n <= 6:
        raise ValueError('Compare between one and six results')
    fig = _figure((4*n, 6.1))
    grid = fig.add_gridspec(2, n, height_ratios=(1.15, 0.85))
    history_ax = fig.add_subplot(grid[1, :])
    for i, (label, result) in enumerate(results.items()):
        ax = fig.add_subplot(grid[0, i])
        _map(ax, result.state, result.params, 'density')
        ax.set_title(label, fontsize=11, loc='left', weight='bold')
        history_ax.plot(result.history['time'], result.history['heterogeneity'], label=label, lw=2)
    history_ax.set(xlabel='Time [model time]', ylabel='Density heterogeneity H', ylim=(0, None))
    history_ax.legend(frameon=False, ncol=min(n, 3))
    history_ax.grid(alpha=0.18)
    history_ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Controlled comparisons · same model, selected coupling removed',
                 fontsize=14, weight='bold', color=NAVY)
    return fig


def png_frame(result):
    fig = plot_result(result)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=90)
    fig.clear()
    return buffer.getvalue()
