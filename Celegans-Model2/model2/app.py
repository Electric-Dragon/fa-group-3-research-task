"""Interactive window: worms (left), oxygen (right), controls.

    python -m model2                       # open the window
    python -m model2 --screenshot out.png --minutes 10 --density 40
                                           # run headless, save one frame

Space toggles run/pause.
"""
import argparse
import time

import numpy as np
import taichi as ti

from . import params as P, sim

RES = (1520, 700)
PANEL = 580                      # px, side of each square panel
WORMS_X0, FIELD_X0, PANEL_Y0 = 320, 920, 50
BAR_Y0, BAR_H = 22, 14           # colour bar under the field panel
WORM_LENGTH = 0.5                # mm, drawn segment length (display only)

STEPS_PER_DRAW = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
SCALES = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


@ti.data_oriented
class View:
    def __init__(self, s):
        self.s = s
        self.img = ti.Vector.field(3, ti.f32, RES)
        self.verts = ti.Vector.field(2, ti.f32, 2 * s.n_max)

    @ti.func
    def _colormap(self, t):
        # Five-stop viridis approximation
        t = ti.min(ti.max(t, 0.0), 1.0) * 4.0
        k = ti.min(int(t), 3)
        fr = t - k
        c0 = ti.Vector([0.0, 0.0, 0.0])
        c1 = ti.Vector([0.0, 0.0, 0.0])
        if k == 0:
            c0, c1 = ti.Vector([0.267, 0.005, 0.329]), ti.Vector([0.229, 0.322, 0.546])
        elif k == 1:
            c0, c1 = ti.Vector([0.229, 0.322, 0.546]), ti.Vector([0.128, 0.567, 0.551])
        elif k == 2:
            c0, c1 = ti.Vector([0.128, 0.567, 0.551]), ti.Vector([0.369, 0.789, 0.383])
        else:
            c0, c1 = ti.Vector([0.369, 0.789, 0.383]), ti.Vector([0.993, 0.906, 0.144])
        return c0 * (1 - fr) + c1 * fr

    @ti.kernel
    def paint(self, field: ti.template(), vmax: ti.f32):
        G = P.GRID
        for x, y in self.img:
            c = ti.Vector([0.11, 0.11, 0.12])
            if WORMS_X0 <= x < WORMS_X0 + PANEL and PANEL_Y0 <= y < PANEL_Y0 + PANEL:
                c = ti.Vector([0.97, 0.97, 0.95])
            elif FIELD_X0 <= x < FIELD_X0 + PANEL and PANEL_Y0 <= y < PANEL_Y0 + PANEL:
                i = (x - FIELD_X0) * G // PANEL
                j = (y - PANEL_Y0) * G // PANEL
                c = self._colormap(field[i, j] / vmax)
            elif FIELD_X0 <= x < FIELD_X0 + PANEL and BAR_Y0 <= y < BAR_Y0 + BAR_H:
                c = self._colormap((x - FIELD_X0) / PANEL)
            self.img[x, y] = c

    @ti.kernel
    def worms(self, n: ti.i32, length: ti.f32):
        sx = PANEL / P.BOX / RES[0]
        sy = PANEL / P.BOX / RES[1]
        ox = WORMS_X0 / RES[0]
        oy = PANEL_Y0 / RES[1]
        for k in range(self.s.n_max):
            if k < n:
                head = self.s.pos[k]
                th = self.s.angle[k]
                tail = head - length * ti.Vector([ti.cos(th), ti.sin(th)])
                tail = ti.min(ti.max(tail, 0.0), P.BOX)      # don't draw across the wrap
                self.verts[2 * k] = ti.Vector([ox + head[0] * sx, oy + head[1] * sy])
                self.verts[2 * k + 1] = ti.Vector([ox + tail[0] * sx, oy + tail[1] * sy])
            else:
                self.verts[2 * k] = ti.Vector([-1.0, -1.0])
                self.verts[2 * k + 1] = ti.Vector([-1.0, -1.0])


class App:
    def __init__(self, density=40, show=True):
        self.s = sim.Simulation(density=density)
        self.view = View(self.s)
        self.window = ti.ui.Window('C. elegans - Model 2', RES, vsync=True, show_window=show)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.running = False
        self.density = density
        self.ambient_i = len(P.AMBIENT_STEPS) - 1
        self.f_i = self.d_i = self.k_i = SCALES.index(1.0)
        self.speed_i = STEPS_PER_DRAW.index(50)
        self.rate = 0.0

    # ---- controls -------------------------------------------------------

    def controls(self):
        s = self.s
        fx = 1 / RES[0]
        with self.gui.sub_window('Controls', 0, 0, (WORMS_X0 - 10) * fx, 1.0) as g:
            if g.button('Pause' if self.running else 'Run'):
                self.running = not self.running
            if g.button('Reset'):
                s.density = self.density
                s.reset(seed=int(time.time()))
            g.text('')

            g.text(f'Strain: {s.strain.name}  ({s.strain.source})')
            other = P.N2 if s.strain is P.NPR1 else P.NPR1
            if g.button(f'Switch to {other.name}'):
                s.strain = other
            g.text('')

            self.density = g.slider_int('Density', self.density, 1, P.DENSITY_MAX)
            note = '' if self.density == round(s.density) else '  (press Reset)'
            g.text(f'  {self.density} worms/mm^2{note}')

            self.ambient_i = g.slider_int('Ambient O2', self.ambient_i, 0, len(P.AMBIENT_STEPS) - 1)
            s.ambient = P.AMBIENT_STEPS[self.ambient_i]
            g.text(f'  {s.ambient * 100:g} %')
            g.text('')

            g.text('Oxygen field (x paper value)')
            self.f_i = g.slider_int('Regeneration', self.f_i, 0, len(SCALES) - 1)
            s.f_scale = SCALES[self.f_i]
            g.text(f'  f = {s.f:.3g} /s  (x{s.f_scale:g})')
            self.d_i = g.slider_int('Diffusion', self.d_i, 0, len(SCALES) - 1)
            s.d_scale = SCALES[self.d_i]
            g.text(f'  D_O = {s.d_o:.2g} mm^2/s  (x{s.d_scale:g})')
            self.k_i = g.slider_int('Absorption', self.k_i, 0, len(SCALES) - 1)
            s.k_scale = SCALES[self.k_i]
            g.text(f'  k_c = {s.k_c:.2g} /s per worm/mm^2  (x{s.k_scale:g})')
            g.text('')


            self.speed_i = g.slider_int('Steps / draw', self.speed_i, 0, len(STEPS_PER_DRAW) - 1)
            g.text(f'  {STEPS_PER_DRAW[self.speed_i]} steps of {P.DT:g} s per frame')
            g.text('')

            st = s.stats()
            t = int(st['time'])
            g.text(f'Worms      {st["n"]:,}  ({st["density"]:.0f}/mm^2)')
            g.text(f'Time       {t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}')
            g.text(f'O2 mean    {st["o_mean"] * 100:.1f} %   min {st["o_min"] * 100:.1f} %')
            g.text(f'Speed      {self.rate:,.0f} steps/s')

        y = 1 - (PANEL_Y0 + PANEL + 52) / RES[1]
        with self.gui.sub_window('Worms', WORMS_X0 * fx, y, PANEL * fx, 0.068) as g:
            g.text(f'every worm drawn ({s.n:,}), {P.BOX:g} x {P.BOX:g} mm')
        with self.gui.sub_window('Oxygen', FIELD_X0 * fx, y, PANEL * fx, 0.068) as g:
            g.text('O2: 0 % (dark) to 21 % (yellow); colour bar below')

    # ---- frame ----------------------------------------------------------

    def draw(self):
        s = self.s
        self.view.paint(s.O, 0.21)
        self.view.worms(s.n, WORM_LENGTH)
        self.canvas.set_image(self.view.img)
        self.canvas.lines(self.view.verts, width=0.0018, color=(0.08, 0.08, 0.12))

    def run(self):
        while self.window.running:
            if self.window.get_event(ti.ui.PRESS) and self.window.event.key == ti.ui.SPACE:
                self.running = not self.running
            self.controls()
            if self.running:
                steps = STEPS_PER_DRAW[self.speed_i]
                t0 = time.perf_counter()
                self.s.step(steps)
                ti.sync()
                self.rate = steps / max(time.perf_counter() - t0, 1e-9)
            self.draw()
            self.window.show()

    def screenshot(self, path, minutes):
        self.s.step(int(minutes * 60 / P.DT))
        self.controls()
        self.draw()
        self.window.save_image(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--arch', default='gpu', help='gpu (Metal) or cpu')
    ap.add_argument('--density', type=int, default=40)
    ap.add_argument('--screenshot', help='run headless and save one frame to this path')
    ap.add_argument('--minutes', type=float, default=10.0, help='simulated time before --screenshot')
    args = ap.parse_args()

    sim.init(args.arch)
    if args.screenshot:
        App(args.density, show=False).screenshot(args.screenshot, args.minutes)
        print('saved', args.screenshot)
    else:
        App(args.density).run()


if __name__ == '__main__':
    main()
