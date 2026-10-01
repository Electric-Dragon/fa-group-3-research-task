"""Worm agents + oxygen field, on Taichi (Demir et al. 2020, Eqs. 45-47).

Per step: worms -> density W -> oxygen -> worms move at V(O).
Call `init()` once before creating a Simulation.
"""
import math

import numpy as np
import taichi as ti

from . import params as P


def init(arch='gpu', seed=0):
    ti.init(arch=getattr(ti, arch), random_seed=seed, default_fp=ti.f32,
            offline_cache=True, log_level=ti.WARN)


@ti.data_oriented
class Simulation:
    def __init__(self, density=40.0, strain=P.NPR1, ambient=0.21, seed=0):
        G, GW = P.GRID, P.DENSITY_GRID
        self.dx = P.BOX / G
        self.dw = P.BOX / GW
        self.n_max = int(P.DENSITY_MAX * P.BOX ** 2)

        self.pos = ti.Vector.field(2, ti.f32, self.n_max)
        self.angle = ti.field(ti.f32, self.n_max)
        self.speed = ti.field(ti.f32, self.n_max)
        self.raw = ti.field(ti.f32, (GW, GW))
        self.tmp = ti.field(ti.f32, (GW, GW))
        self.W = ti.field(ti.f32, (GW, GW))   # worms/mm^2, on the coarse density grid
        self.O = ti.field(ti.f32, (G, G))     # oxygen fraction
        self.new = ti.field(ti.f32, (G, G))

        self.radius = int(math.ceil(3 * P.SMOOTHING / self.dw))
        k = np.arange(-self.radius, self.radius + 1) * self.dw
        w = np.exp(-0.5 * (k / P.SMOOTHING) ** 2)
        self.weights = ti.field(ti.f32, len(k))
        self.weights.from_numpy((w / w.sum()).astype(np.float32))

        # Live controls
        self.strain = strain
        self.ambient = ambient
        self.f_scale = self.d_scale = self.k_scale = 1.0
        self.density = density
        self.reset(seed)

    # ---- controls -------------------------------------------------------

    def reset(self, seed=0):
        """Uniform random worms at self.density, oxygen at ambient."""
        rng = np.random.default_rng(seed)
        self.n = int(round(self.density * P.BOX ** 2))
        pos = np.zeros((self.n_max, 2), np.float32)
        pos[:self.n] = rng.uniform(0, P.BOX, (self.n, 2))
        self.pos.from_numpy(pos)
        self.angle.from_numpy(rng.uniform(0, 2 * np.pi, self.n_max).astype(np.float32))
        self.O.fill(self.ambient)
        self.steps = 0

    @property
    def f(self):
        return P.F * self.f_scale

    @property
    def d_o(self):
        return P.D_O * self.d_scale

    @property
    def k_c(self):
        return P.K_C * self.k_scale

    # ---- stepping -------------------------------------------------------

    def step(self, steps=1):
        s = self.strain
        dx2 = self.dx * self.dx
        n_o = max(1, math.ceil(self.d_o * P.DT / dx2 / 0.2), math.ceil(self.f * P.DT / 0.5))
        a, b, c = s.a * 1e3, s.b * 1e3, s.c * 1e3
        for _ in range(steps):
            self._deposit(self.n)
            self._blur()
            for _ in range(n_o):
                self._oxygen(P.DT / n_o, self.d_o, self.f, self.k_c, self.ambient)
            self._move(self.n, P.DT, P.TAU, a, b, c)
            self.steps += 1

    @ti.func
    def _sample(self, field: ti.template(), p, h, G):
        """Bilinear, periodic sample of a cell-centred field with cell size h."""
        g = p / h - 0.5
        base = ti.floor(g)
        fr = g - base
        i, j = int(base[0]), int(base[1])
        v00 = field[i % G, j % G]
        v10 = field[(i + 1) % G, j % G]
        v01 = field[i % G, (j + 1) % G]
        v11 = field[(i + 1) % G, (j + 1) % G]
        return (v00 * (1 - fr[0]) * (1 - fr[1]) + v10 * fr[0] * (1 - fr[1])
                + v01 * (1 - fr[0]) * fr[1] + v11 * fr[0] * fr[1])

    @ti.kernel
    def _deposit(self, n: ti.i32):
        G = P.DENSITY_GRID
        inv_area = 1.0 / (self.dw * self.dw)
        for I in ti.grouped(self.raw):
            self.raw[I] = 0.0
        for k in range(n):
            g = self.pos[k] / self.dw - 0.5
            base = ti.floor(g)
            fr = g - base
            i, j = int(base[0]), int(base[1])
            for di, dj in ti.static(ti.ndrange(2, 2)):
                wx = fr[0] if di else 1.0 - fr[0]
                wy = fr[1] if dj else 1.0 - fr[1]
                ti.atomic_add(self.raw[(i + di) % G, (j + dj) % G], wx * wy * inv_area)

    @ti.kernel
    def _blur(self):
        G = P.DENSITY_GRID
        r = self.radius
        for i, j in self.tmp:
            acc = 0.0
            for k in range(2 * r + 1):
                acc += self.weights[k] * self.raw[(i + k - r) % G, j]
            self.tmp[i, j] = acc
        for i, j in self.W:
            acc = 0.0
            for k in range(2 * r + 1):
                acc += self.weights[k] * self.tmp[i, (j + k - r) % G]
            self.W[i, j] = acc

    @ti.func
    def _laplacian(self, field: ti.template(), i, j):
        G = P.GRID
        return (field[(i + 1) % G, j] + field[(i - 1) % G, j] + field[i, (j + 1) % G]
                + field[i, (j - 1) % G] - 4.0 * field[i, j]) / (self.dx * self.dx)

    @ti.kernel
    def _oxygen(self, dt: ti.f32, d: ti.f32, f: ti.f32, k_c: ti.f32, ambient: ti.f32):
        # Paper Eq. 46: dO/dt = D_O lap O + f (O_am - O) - k_c W   (O >= 0, A5)
        for i, j in self.O:
            o = self.O[i, j]
            w = self._sample(self.W, (ti.Vector([i, j]) + 0.5) * self.dx, self.dw, P.DENSITY_GRID)
            rate = d * self._laplacian(self.O, i, j) + f * (ambient - o) - k_c * w
            self.new[i, j] = ti.max(o + dt * rate, 0.0)
        for I in ti.grouped(self.O):
            self.O[I] = self.new[I]

    @ti.kernel
    def _move(self, n: ti.i32, dt: ti.f32, tau: ti.f32, a: ti.f32, b: ti.f32, c: ti.f32):
        # Run at V(O), tumble to a random heading at rate tau (paper Eqs. 3-4, 47).
        for k in range(n):
            p = self.pos[k]
            o = self._sample(self.O, p, self.dx, P.GRID)
            v = ti.max(a * o * o + b * o + c, 0.0)
            if ti.random() < tau * dt:
                self.angle[k] = ti.random() * 2.0 * math.pi
            th = self.angle[k]
            p += v * dt * ti.Vector([ti.cos(th), ti.sin(th)])
            self.pos[k] = p - P.BOX * ti.floor(p / P.BOX)
            self.speed[k] = v

    # ---- readouts -------------------------------------------------------

    @property
    def time(self):
        return self.steps * P.DT

    def stats(self):
        o = self.O.to_numpy()
        out = {'n': self.n, 'density': self.n / P.BOX ** 2, 'time': self.time,
               'o_mean': float(o.mean()), 'o_min': float(o.min())}
        return out

    def contrast(self):
        """std/mean of the smoothed worm density: ~0.1 uniform, >~0.5 patterned."""
        w = self.W.to_numpy()
        return float(w.std() / w.mean()) if w.mean() > 0 else 0.0
