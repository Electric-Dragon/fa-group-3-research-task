"""Interactive notebook demo: the environment and the population are the controls.

Everything about the agents is fixed by the strain dropdown, and changing it
needs a Reset -- that is the point of Model 1. What can be changed mid-run is
the environment (ambient level, its spatial shape, the replenishment rate) and
the number of agents.

Same conventions as Model 0's demo: a cooperative asyncio loop so widgets stay
responsive, only one active demo at a time, the run starts paused, and drawing
never consumes RNG state.
"""
import asyncio
import html
import weakref
import numpy as np

from .engine import Simulation
from .environment import (Environment, uniform, linear_gradient, spot, schedule,
                          cover_glass, pct)
from .strains import PRESETS
from .stability import predicted_map
from .visualization import demo_frame

_active_demo = None

ENVIRONMENT_PRESETS = ('uniform', 'cosine gradient', 'low-oxygen spot',
                       'cover glass', 'O2 step schedule')


class EnvironmentDemo:
    """Widget wrapper around `Simulation` with environment-only controls.

    Parameters
    ----------
    strain : preset name shown first in the dropdown.
    domain : (width, height, nx, ny). Changing it needs a new demo.
    n_agents : starting population.
    seed, turns : as in Model 0's demo; seed applies on Reset.
    """

    def __init__(self, strain='NPR1_LIKE', *, domain=(96.0, 96.0, 96, 96),
                 n_agents=5400, seed=1, turns=20000, dt=0.1):
        try:
            import ipywidgets as w
        except ImportError as exc:
            raise ImportError('Install requirements.txt in the notebook kernel environment.') from exc
        global _active_demo
        previous = _active_demo() if _active_demo else None
        if previous is not None:
            previous.pause()
        self._task = None
        self._shown = False
        self._closed = False
        self.domain = tuple(domain)
        self.sim = Simulation(strain, uniform(1.0), domain=self.domain,
                              n_agents=n_agents, dt=dt, seed=seed,
                              record_every=50, record_patterns=False)
        # One stability scan, reused every frame; the strain dropdown refreshes it.
        self._predicted = None
        self._refresh_predicted()

        self.play = w.Button(description='Start', icon='play', button_style='success')
        self.reset_button = w.Button(description='Reset', icon='refresh')
        self.step_button = w.Button(description='Advance frame', icon='step-forward')
        self.strain_widget = w.Dropdown(options=list(PRESETS), value=self.sim.strain.name,
                                        description='Strain',
                                        style={'description_width': '110px'})
        self.seed_widget = w.BoundedIntText(value=seed, min=0, max=2**31 - 1, description='Seed')
        self.turns_widget = w.BoundedIntText(value=turns, min=1, max=10_000_000,
                                             description='Stop at turn')
        self.draw_widget = w.IntSlider(value=50, min=5, max=400, step=5,
                                       description='Steps / draw', continuous_update=False)

        self.env_widget = w.Dropdown(options=ENVIRONMENT_PRESETS, value='uniform',
                                     description='Environment',
                                     style={'description_width': '110px'})
        self.ambient_widget = w.FloatSlider(value=21.0, min=1.0, max=21.0, step=0.5,
                                            description='Ambient O2 [%]', readout_format='.1f',
                                            continuous_update=False,
                                            style={'description_width': '145px'},
                                            layout=w.Layout(width='360px'))
        self.contrast_widget = w.FloatSlider(value=7.0, min=0.5, max=21.0, step=0.5,
                                             description='Spot / low end [%]',
                                             readout_format='.1f', continuous_update=False,
                                             style={'description_width': '145px'},
                                             layout=w.Layout(width='360px'))
        self.replenishment_widget = w.FloatSlider(
            value=0.06, min=0.0, max=0.30, step=0.005, description='Replenishment f',
            readout_format='.3f', continuous_update=False,
            style={'description_width': '145px'}, layout=w.Layout(width='360px'))
        self.diffusion_widget = w.FloatSlider(
            value=1.0, min=0.0, max=8.0, step=0.1, description='Oxygen diffusion',
            readout_format='.1f', continuous_update=False,
            style={'description_width': '145px'}, layout=w.Layout(width='360px'))
        self.radius_widget = w.FloatSlider(
            value=0.22, min=0.05, max=0.45, step=0.01, description='Spot radius / side',
            readout_format='.2f', continuous_update=False,
            style={'description_width': '145px'}, layout=w.Layout(width='360px'))

        self.count_widget = w.BoundedIntText(value=max(200, n_agents // 10), min=1,
                                             max=500_000, description='N')
        self.mode_widget = w.Dropdown(options=['uniform', 'point'], value='uniform',
                                      description='Where',
                                      style={'description_width': '60px'},
                                      layout=w.Layout(width='190px'))
        self.add_button = w.Button(description='Add N', icon='plus')
        self.remove_button = w.Button(description='Remove N', icon='minus')

        self.status = w.HTML()
        self.image = w.Image(format='png', layout=w.Layout(width='100%', max_width='1200px'))

        for widget in (self.env_widget, self.ambient_widget, self.contrast_widget,
                       self.replenishment_widget, self.diffusion_widget, self.radius_widget):
            widget.observe(self._environment_changed, names='value')
        self.strain_widget.observe(self._strain_changed, names='value')
        self.play.on_click(lambda _: self.pause() if self.running else self.start())
        self.reset_button.on_click(lambda _: self.reset())
        self.step_button.on_click(lambda _: self.advance())
        self.add_button.on_click(lambda _: self._change_population(+1))
        self.remove_button.on_click(lambda _: self._change_population(-1))

        environment_box = w.HBox(
            [w.VBox([self.env_widget, self.ambient_widget, self.contrast_widget]),
             w.VBox([self.replenishment_widget, self.diffusion_widget, self.radius_widget])],
            layout=w.Layout(flex_flow='row wrap'))
        population_box = w.HBox([self.count_widget, self.mode_widget,
                                 self.add_button, self.remove_button],
                                layout=w.Layout(flex_flow='row wrap'))
        self.widget = w.VBox([
            w.HTML('<h3>Model 1 · environment-controlled patterns</h3>'
                   '<p>Agent parameters are frozen by the strain. You control the '
                   '<b>environment</b> and the <b>population</b>, both live. '
                   'Changing the strain resets the run, because a strain is not '
                   'something an experiment changes mid-plate.</p>'),
            w.HBox([self.strain_widget]),
            environment_box, population_box,
            w.HBox([self.seed_widget, self.turns_widget, self.draw_widget],
                   layout=w.Layout(flex_flow='row wrap')),
            w.HBox([self.play, self.reset_button, self.step_button]),
            self.status, self.image])
        self._render()

    # -- environment wiring ----------------------------------------------
    def _refresh_predicted(self):
        self._predicted = predicted_map(
            self.sim.strain, np.linspace(0.05, 1.0, 30), np.linspace(0.05, 1.6, 30),
            replenishment=float(np.mean(np.asarray(self.sim.environment.replenishment))))

    def build_environment(self):
        """The Environment the current widget settings describe."""
        width, height, nx, ny = self.domain
        level = pct(self.ambient_widget.value)
        low = pct(self.contrast_widget.value)
        f = self.replenishment_widget.value
        D = self.diffusion_widget.value
        radius = self.radius_widget.value * min(width, height)
        kind = self.env_widget.value
        if kind == 'cosine gradient':
            return linear_gradient(low, level, 'x', width=width, height=height,
                                   nx=nx, ny=ny, replenishment=f, oxygen_diffusion=D)
        if kind == 'low-oxygen spot':
            return spot((width / 2, height / 2), radius, low, level, width=width,
                        height=height, nx=nx, ny=ny, replenishment=f, oxygen_diffusion=D)
        if kind == 'cover glass':
            return cover_glass((width / 2, height / 2), radius, 0.1 * f, f,
                               width=width, height=height, nx=nx, ny=ny,
                               ambient=level, oxygen_diffusion=D)
        if kind == 'O2 step schedule':
            # Gray et al. 2004 Fig. 4d: air, then hypoxia, then air again.
            now = self.sim.time
            return schedule([(now, level), (now + 300, low), (now + 600, level)],
                            replenishment=f, oxygen_diffusion=D)
        return uniform(level, f, D)

    def _environment_changed(self, _):
        self.sim.set_environment(self.build_environment())
        self._refresh_predicted()
        if not self.running:
            self._render()

    def _strain_changed(self, _):
        self.pause()
        self.sim = Simulation(self.strain_widget.value, self.build_environment(),
                              domain=self.domain, n_agents=self.sim.params.n_agents,
                              dt=self.sim.params.dt, seed=self.seed_widget.value,
                              record_every=50, record_patterns=False)
        self._refresh_predicted()
        self._render()

    def _change_population(self, sign):
        n = int(self.count_widget.value)
        if sign > 0:
            self.sim.add_agents(n, mode=self.mode_widget.value,
                                center=(self.domain[0] / 2, self.domain[1] / 2),
                                spread=0.05 * min(self.domain[0], self.domain[1]))
        else:
            self.sim.remove_agents(n)
        if not self.running:
            self._render()

    # -- loop -------------------------------------------------------------
    @property
    def running(self):
        return self._task is not None and not self._task.done()

    def _render(self):
        png, metrics, prediction = demo_frame(self.sim, self._predicted)
        self.image.value = png
        self.status.value = (
            f'<b>Turn {self.sim.turn}</b> · time {self.sim.time:.1f} · '
            f'N = {self.sim.params.n_agents} · W = {self.sim.density:.3f} · '
            f'H = {metrics["heterogeneity"]:.3f} '
            f'({metrics["heterogeneity_ratio"]:.1f}x shot noise) · '
            f'label <b>{metrics["label"]}</b> · '
            f'mean O = {self.sim.state.oxygen.mean():.3f} · '
            f'predicted {"unstable" if prediction["unstable"] else "stable"}'
            + (f', wavelength {prediction["wavelength"]:.0f}' if prediction['unstable'] else ''))

    def show(self):
        from IPython.display import display
        if not self._shown:
            display(self.widget)
            self._shown = True
        return None

    def start(self):
        global _active_demo
        if self._closed or self.running:
            return
        if self.sim.turn >= self.turns_widget.value:
            self.status.value = 'Run complete. Increase Stop at turn, or Reset.'
            return
        previous = _active_demo() if _active_demo else None
        if previous is not None and previous is not self:
            previous.pause()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.status.value = ('Use a running Jupyter kernel for Start, '
                                 'or Advance frame for a finite step.')
            return
        _active_demo = weakref.ref(self)
        self._task = loop.create_task(self._loop())
        self.play.description, self.play.icon = 'Pause', 'pause'
        self.step_button.disabled = True

    async def _loop(self):
        try:
            while self.sim.turn < self.turns_widget.value:
                end = min(self.sim.turn + self.draw_widget.value, self.turns_widget.value)
                while self.sim.turn < end:
                    self.sim.step(min(5, end - self.sim.turn))
                    await asyncio.sleep(0)   # keep widgets and interrupts alive
                self._render()
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.status.value = f'<b>Stopped:</b> {html.escape(str(exc))}'
        finally:
            if self._task is asyncio.current_task():
                self._task = None
                self.play.description, self.play.icon = 'Start', 'play'
                self.step_button.disabled = False

    def pause(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self.play.description, self.play.icon = 'Start', 'play'
        self.step_button.disabled = False

    def reset(self):
        self.pause()
        self.sim.set_environment(self.build_environment())
        self.sim.reset(seed=self.seed_widget.value)
        self._render()

    def advance(self):
        self.pause()
        n = min(self.draw_widget.value, max(0, self.turns_widget.value - self.sim.turn))
        self.sim.step(n)
        self._render()

    def close(self):
        self.pause()
        self._closed = True
        self.widget.close()
