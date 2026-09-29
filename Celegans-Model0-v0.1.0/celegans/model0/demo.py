"""Notebook widgets with a cooperative asyncio loop and explicit run controls."""
import asyncio
import html
import weakref
from .model import Simulation
from .visualization import png_frame

_active_demo = None


class OxygenDemo:
    """Interactive wrapper around Simulation; starts paused.

    Physical sliders take effect immediately. Seed takes effect on Reset.
    Steps/draw changes rendering cadence only. Only one demo runs at a time.
    """
    def __init__(self, scenario='feedback', params=None, *, seed=1, turns=3000):
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
        self.sim = Simulation(scenario, params, seed=seed)
        p = self.sim.params
        self.play = w.Button(description='Start', icon='play', button_style='success')
        self.reset_button = w.Button(description='Reset', icon='refresh')
        self.step_button = w.Button(description='Advance frame', icon='step-forward')
        self.seed_widget = w.BoundedIntText(value=seed, min=0, max=2**31-1, description='Seed')
        self.turns_widget = w.BoundedIntText(value=turns, min=1, max=1_000_000, description='Stop at turn')
        self.draw_widget = w.IntSlider(value=40, min=5, max=200, step=5,
                                        description='Steps / draw', continuous_update=False)
        self.status = w.HTML()
        self.image = w.Image(format='png', layout=w.Layout(width='100%', max_width='1100px'))
        self.sliders = {}
        specs = [
            ('consumption', 'Consumption q', 0, 0.12, 0.002, '.3f'),
            ('replenishment', 'Replenishment k', 0, 0.20, 0.005, '.3f'),
            ('oxygen_diffusion', 'Oxygen diffusion', 0, 8, 0.1, '.1f'),
            ('ambient', 'Ambient oxygen', 0, 1, 0.01, '.2f'),
            ('rotational_diffusion', 'Angular diffusion', 0, 1, 0.01, '.2f'),
            ('oxygen_midpoint', 'Speed midpoint', 0.1, 0.9, 0.01, '.2f'),
            ('response_width', 'Response width', 0.01, 0.20, 0.005, '.3f'),
        ]
        for name, label, lo, hi, step, fmt in specs:
            slider = w.FloatSlider(value=getattr(p, name), min=min(lo, getattr(p, name)),
                                   max=max(hi, getattr(p, name)), step=step,
                                   readout_format=fmt, description=label,
                                   continuous_update=False,
                                   style={'description_width': '145px'},
                                   layout=w.Layout(width='360px'))
            if not self.sim.scenario.dynamic and name in {
                'consumption', 'replenishment', 'oxygen_diffusion', 'ambient'}:
                slider.disabled = True
            slider.observe(self._change, names='value')
            self.sliders[name] = slider
        self.response = w.Checkbox(value=p.speed_response, description='Speed depends on oxygen')
        self.response.observe(self._change, names='value')
        controls = list(self.sliders.values())
        physical = w.HBox([w.VBox(controls[:4]), w.VBox(controls[4:] + [self.response])],
                          layout=w.Layout(flex_flow='row wrap'))
        self.widget = w.VBox([
            w.HTML(f'<h3>Model 0 · {html.escape(self.sim.scenario.title)}</h3>'
                   '<p>Physical controls apply immediately. Reset for a clean comparison. '
                   'Disabled field controls do not apply to a prescribed landscape.</p>'),
            physical,
            w.HBox([self.seed_widget, self.turns_widget, self.draw_widget],
                   layout=w.Layout(flex_flow='row wrap')),
            w.HBox([self.play, self.reset_button, self.step_button]), self.status, self.image])
        self.play.on_click(lambda _: self.pause() if self.running else self.start())
        self.reset_button.on_click(lambda _: self.reset())
        self.step_button.on_click(lambda _: self.advance())
        self._render()

    @property
    def running(self):
        return self._task is not None and not self._task.done()

    def _change(self, _):
        changes = {name: float(widget.value) for name, widget in self.sliders.items()}
        changes['speed_response'] = self.response.value
        self.sim.set_params(**changes)
        if not self.running:
            self._render()

    def _render(self):
        result = self.sim.result()
        self.image.value = png_frame(result)
        self.status.value = (f'<b>Turn {self.sim.turn}</b> · time {self.sim.time:.1f} · '
                             f'H = {result.history["heterogeneity"][-1]:.3f} · '
                             f'mean oxygen = {self.sim.state.oxygen.mean():.3f}')

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
            self.status.value = 'Use a running Jupyter kernel for Start, or Advance frame for a finite step.'
            return
        _active_demo = weakref.ref(self)
        self._task = loop.create_task(self._loop())
        self.play.description, self.play.icon = 'Pause', 'pause'
        self.step_button.disabled = True

    async def _loop(self):
        try:
            while self.sim.turn < self.turns_widget.value:
                end = min(self.sim.turn+self.draw_widget.value, self.turns_widget.value)
                while self.sim.turn < end:
                    self.sim.step(min(5, end-self.sim.turn))
                    await asyncio.sleep(0)  # allow widgets and interrupt handling
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
        self.sim.reset(seed=self.seed_widget.value)
        self._render()

    def advance(self):
        self.pause()
        n = min(self.draw_widget.value, max(0, self.turns_widget.value-self.sim.turn))
        self.sim.step(n)
        self._render()

    def close(self):
        self.pause()
        self._closed = True
        self.widget.close()
