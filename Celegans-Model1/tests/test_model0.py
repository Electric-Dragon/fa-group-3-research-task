"""Run from the project root: python -m unittest discover -s tests -v."""
import unittest
from dataclasses import replace
import numpy as np
import celegans as ce
from celegans.model0.model import sample_field, _oxygen_step


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.p = ce.Parameters(width=16, height=12, nx=16, ny=12, n_agents=120)

    def test_density_mass_and_periodic_shift(self):
        pos = ce.Simulation(params=self.p).state.positions
        rho = ce.density_grid(pos, self.p)
        self.assertAlmostEqual(rho.sum()*self.p.dx*self.p.dy, self.p.n_agents, places=10)
        np.testing.assert_allclose(ce.density_grid(pos+(16, 12), self.p), rho, atol=1e-14)
        field = np.arange(192).reshape(12, 16)
        np.testing.assert_allclose(sample_field(field, pos, self.p),
                                   sample_field(field, pos+(16, 12), self.p), atol=1e-12)

    def test_diffusion_preserves_mass_and_damps_fourier_mode(self):
        p = replace(self.p, consumption=0, replenishment=0)
        x = (np.arange(p.nx)+0.5)*p.dx
        wave = np.repeat(np.cos(2*np.pi*x/p.width)[None, :], p.ny, axis=0)
        oxygen = 0.5+0.1*wave
        before = oxygen.copy()
        h = 0.01
        _oxygen_step(oxygen, np.zeros_like(oxygen), p, h)
        eigenvalue = -4*np.sin(np.pi/p.nx)**2/p.dx**2
        np.testing.assert_allclose(oxygen, 0.5+0.1*(1+h*p.oxygen_diffusion*eigenvalue)*wave)
        self.assertAlmostEqual(oxygen.sum(), before.sum(), places=10)

    def test_replenishment_has_analytic_solution(self):
        p = replace(self.p, oxygen_diffusion=0, consumption=0)
        oxygen = np.full((p.ny, p.nx), 0.2)
        _oxygen_step(oxygen, np.zeros_like(oxygen), p, 0.4)
        np.testing.assert_allclose(oxygen, p.ambient+(0.2-p.ambient)*np.exp(-p.replenishment*0.4))

    def test_consumption_positive_and_decreasing_even_for_large_sink(self):
        p = replace(self.p, replenishment=0, oxygen_diffusion=0, consumption=100)
        oxygen = np.linspace(0, 1, p.nx*p.ny).reshape(p.ny, p.nx)
        before = oxygen.copy()
        _oxygen_step(oxygen, np.full_like(oxygen, 100), p, 10)
        self.assertTrue(np.isfinite(oxygen).all())
        self.assertTrue((oxygen >= 0).all())
        self.assertTrue((oxygen <= before).all())

    def test_chunking_rendering_snapshots_and_reset(self):
        a, b = ce.Simulation(params=self.p), ce.Simulation(params=self.p)
        initial = a.state
        a.step(35)
        for _ in range(7):
            b.step(5)
            b.result()  # diagnostics must not consume RNG state
        np.testing.assert_array_equal(a.state.positions, b.state.positions)
        np.testing.assert_array_equal(a.state.oxygen, b.state.oxygen)
        self.assertFalse(initial.positions.flags.writeable)
        saved = a.result()
        a.reset()
        np.testing.assert_array_equal(a.state.positions, initial.positions)
        a.step(35)
        np.testing.assert_array_equal(a.state.positions, saved.positions)

    def test_fixed_landscape_stays_fixed(self):
        sim = ce.Simulation('landscape', self.p)
        oxygen = sim.state.oxygen
        sim.step(100)
        np.testing.assert_array_equal(oxygen, sim.state.oxygen)

    def test_uniform_no_sink_stays_uniform(self):
        p = replace(self.p, consumption=0)
        result = ce.run(params=p, turns=100)
        np.testing.assert_allclose(result.oxygen, p.ambient, atol=1e-14)

    def test_periodic_motion_and_bounded_oxygen(self):
        p = replace(self.p, oxygen_diffusion=12, dt=0.4, consumption=0.3)
        state = ce.run(params=p, turns=60).state
        self.assertTrue(np.isfinite(state.oxygen).all())
        self.assertTrue((state.oxygen >= 0).all() and (state.oxygen <= 1).all())
        self.assertTrue((state.positions >= 0).all())
        self.assertTrue((state.positions < (p.width, p.height)).all())

    def test_validation(self):
        for change in [{'dt': 0}, {'response_width': 0}, {'n_agents': 1.5},
                       {'consumption': -1}, {'v_min': 3}, {'ambient': 2}]:
            with self.assertRaises(ValueError):
                replace(self.p, **change)
        with self.assertRaises(ValueError):
            ce.Simulation(params=self.p).step(-1)


if __name__ == '__main__':
    unittest.main()
