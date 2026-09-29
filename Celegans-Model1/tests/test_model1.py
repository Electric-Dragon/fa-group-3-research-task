"""Run from the project root: python -m unittest discover -s tests -v.

These checks cover the parts of Model 1 that are new relative to Model 0: the
stability prediction, array-valued environments, runtime population changes,
periodic topology and the pattern classifier. Model 0's own invariants stay in
`test_model0.py` and are run unchanged against the copied package.
"""
import unittest
import numpy as np

import celegans as ce
from celegans import model1 as m1
from celegans.model1 import environment as envmod
from celegans.model1.engine import Simulation, density_grid, oxygen_step
from celegans.model1.patterns import (classify, periodic_label, structure_factor,
                                      shot_noise_heterogeneity, coarsening_exponent)
from celegans.model1.strains import (NPR1_LIKE, N2_LIKE, CONSTANT_SPEED,
                                     MODEL0_DEFAULT, Strain)
from celegans.model1 import stability as st


def synthetic(kind, n=96, period=24):
    """Idealised density fields for classifier tests, on a periodic grid."""
    y, x = np.mgrid[0:n, 0:n]
    if kind == 'stripes':
        return 1.0 + 0.6 * np.cos(2 * np.pi * x / period)
    inside = ((x % period - period / 2) ** 2 + (y % period - period / 2) ** 2) < (period / 6) ** 2
    if kind == 'dots':
        return 1.0 + np.where(inside, 3.0, -0.2)
    if kind == 'holes':
        return 1.0 + np.where(inside, -0.9, 0.3)
    raise ValueError(kind)


class StabilityChecks(unittest.TestCase):
    """The dispersion relation against the numbers quoted in the handoff."""

    ORACLE = dict(O_eq=0.510, V=0.626, dV=8.91, D_W=0.784, beta=11.16,
                  growth_rate=0.057, wavelength=21.6)

    def setUp(self):
        self.density = 1800 / (64 * 48)

    def test_reproduces_documented_oracle(self):
        p = st.predict(MODEL0_DEFAULT, (1.0, 0.06), self.density)
        self.assertTrue(p['unstable'])
        for key, expected in self.ORACLE.items():
            self.assertAlmostEqual(p[key] / expected, 1.0, delta=0.02,
                                   msg=f'{key}: got {p[key]}, expected ~{expected}')

    def test_equilibrium_matches_model0(self):
        for n_agents in (400, 1800, 5000):
            p = ce.Parameters(n_agents=n_agents)
            self.assertAlmostEqual(
                st.equilibrium_oxygen(MODEL0_DEFAULT, p.ambient, p.replenishment,
                                      n_agents / (p.width * p.height)),
                ce.uniform_oxygen(p), places=14)

    def test_equilibrium_vectorises(self):
        ambients = np.array([1.0, 0.5, 0.2])
        values = st.equilibrium_oxygen(MODEL0_DEFAULT, ambients, 0.06, 0.6)
        for ambient, value in zip(ambients, values):
            self.assertAlmostEqual(
                value, st.equilibrium_oxygen(MODEL0_DEFAULT, float(ambient), 0.06, 0.6))
        # No replenishment and a live sink consumes everything; no sink holds ambient.
        self.assertEqual(st.equilibrium_oxygen(MODEL0_DEFAULT, 1.0, 0.0, 0.6), 0.0)
        self.assertEqual(st.equilibrium_oxygen(MODEL0_DEFAULT, 1.0, 0.06, 0.0), 1.0)

    def test_constant_speed_is_never_unstable(self):
        for ambient in np.linspace(0.02, 1.0, 12):
            for density in np.geomspace(0.01, 20, 12):
                p = st.predict(CONSTANT_SPEED, (float(ambient), 0.06), float(density))
                self.assertFalse(p['unstable'])
                self.assertEqual(p['beta'], 0.0)
                self.assertLessEqual(p['growth_rate'], 0.0)
        self.assertTrue(np.isnan(st.critical_density(CONSTANT_SPEED, 1.0)))

    def test_flat_plateau_of_the_speed_law_is_stable(self):
        """On the upper plateau V' ~ 0, so beta ~ 0 and nothing can grow.

        Reached here with a strain whose midpoint is far below the equilibrium
        oxygen, which is the 'ambient too high to matter' case.
        """
        plateau = MODEL0_DEFAULT.variant(name='plateau', oxygen_midpoint=0.05)
        p = st.predict(plateau, (1.0, 0.06), 0.3)
        self.assertGreater(p['O_eq'], 0.5)
        self.assertLess(p['beta'], 1e-3)
        self.assertFalse(p['unstable'])

    def test_growth_curve_shape(self):
        """Zero at k = 0, positive in a band, negative once the footprint cuts in.

        lambda(0) = 0 because agent number is conserved: a uniform shift of the
        density is not a growing mode. The unstable band here reaches down to
        k = 0, so k* is an interior maximum rather than an edge.
        """
        lam = st.growth_curve(MODEL0_DEFAULT, (1.0, 0.06), self.density,
                              np.array([1e-6, 1e-3, 0.29, 5.0, 20.0]))
        self.assertLess(abs(lam[0]), 1e-11)
        self.assertGreater(lam[2], 0)
        self.assertGreater(lam[2], lam[1])
        self.assertLess(lam[3], 0)
        self.assertLess(lam[4], 0)

    def test_preset_separation_is_the_documented_one(self):
        """NPR1_LIKE patterns at lower density than N2_LIKE at every ambient level."""
        for ambient in (0.5, 0.7, 0.9, 1.0):
            npr = st.unstable_density_range(NPR1_LIKE, ambient)[0]
            n2 = st.unstable_density_range(N2_LIKE, ambient)[0]
            self.assertTrue(np.isfinite(npr) and np.isfinite(n2))
            self.assertLess(npr, n2, msg=f'at ambient {ambient}')

    def test_predicted_map_orientation(self):
        ambients, densities = np.linspace(0.4, 1.0, 5), np.linspace(0.3, 1.1, 4)
        grid = st.predicted_map(NPR1_LIKE, ambients, densities)
        self.assertEqual(grid['unstable'].shape, (densities.size, ambients.size))
        for i, density in enumerate(densities):
            for j, ambient in enumerate(ambients):
                self.assertEqual(grid['unstable'][i, j],
                                 st.predict(NPR1_LIKE, (ambient, 0.06), density)['unstable'])


class EnvironmentChecks(unittest.TestCase):
    def setUp(self):
        self.domain = (32.0, 24.0, 32, 24)
        self.grid = envmod.cell_centers(*self.domain)

    def test_percent_helper(self):
        self.assertAlmostEqual(float(envmod.pct(21)), 1.0)
        self.assertAlmostEqual(float(envmod.pct(7)), 1 / 3)

    def test_uniform_arrays_reduce_to_the_scalar_case_bit_for_bit(self):
        """A constant map must not change a single bit of the oxygen update.

        This is the guarantee that makes array environments safe: they are the
        same code path, not a parallel one.
        """
        p = ce.Parameters(width=32, height=24, nx=32, ny=24, n_agents=300)
        rng = np.random.default_rng(0)
        rho = rng.uniform(0, 2, (p.ny, p.nx))
        base = rng.uniform(0.1, 0.9, (p.ny, p.nx))
        scalar, array = base.copy(), base.copy()
        oxygen_step(scalar, rho, p, 0.1, 0.7, 0.06)
        oxygen_step(array, rho, p, 0.1,
                    np.full((p.ny, p.nx), 0.7), np.full((p.ny, p.nx), 0.06))
        np.testing.assert_array_equal(scalar, array)

    def test_uniform_environment_run_matches_array_environment_run(self):
        shape = (24, 32)
        scalar_env = envmod.uniform(0.7, 0.06, 1.0)
        array_env = envmod.Environment(np.full(shape, 0.7), np.full(shape, 0.06), 1.0)
        runs = []
        for env in (scalar_env, array_env):
            sim = Simulation(NPR1_LIKE, env, domain=self.domain, n_agents=300,
                             seed=5, record_patterns=False)
            sim.step(40)
            runs.append(sim.state)
        np.testing.assert_array_equal(runs[0].positions, runs[1].positions)
        np.testing.assert_array_equal(runs[0].oxygen, runs[1].oxygen)

    def test_schedule_switches_on_the_right_turn(self):
        env = envmod.schedule([(0, 1.0), (2.0, 1 / 3), (4.0, 1.0)])
        sim = Simulation(NPR1_LIKE, env, domain=self.domain, n_agents=100,
                         dt=0.1, seed=1, record_patterns=False)
        # Ambient is resolved at the start of each outer turn. Turn index i
        # covers [i*dt, (i+1)*dt) and is evaluated at i*dt, so turn 20 is the
        # first that begins at t = 2.0 and therefore the first low one.
        used = []
        for _ in range(60):
            sim.step(1)
            used.append(float(np.mean(sim.ambient_field)))
        self.assertAlmostEqual(used[19], 1.0)
        self.assertAlmostEqual(used[20], 1 / 3)
        self.assertAlmostEqual(used[39], 1 / 3)
        self.assertAlmostEqual(used[40], 1.0)

    def test_cosine_gradient_is_seamless(self):
        env = envmod.linear_gradient(0.2, 1.0, 'x', width=32, height=24, nx=32, ny=24)
        field = env.ambient_field(*self.grid, 0.0)
        row = field[0]
        # No jump across the periodic seam, and the stated extremes are reached.
        self.assertLess(abs(row[0] - row[-1]), abs(row.max() - row.min()) * 0.25)
        self.assertAlmostEqual(row.min(), 0.2, delta=0.02)
        self.assertAlmostEqual(row.max(), 1.0, delta=0.02)

    def test_spot_and_cover_glass_shapes(self):
        low = envmod.spot((16, 12), 5.0, 0.2, 1.0, width=32, height=24, nx=32, ny=24)
        field = low.ambient_field(*self.grid, 0.0)
        self.assertAlmostEqual(field[12, 16], 0.2, delta=0.02)
        self.assertAlmostEqual(field[0, 0], 1.0, delta=0.02)
        cover = envmod.cover_glass((16, 12), 5.0, 0.006, 0.06,
                                   width=32, height=24, nx=32, ny=24)
        f = cover.replenishment_field((24, 32))
        self.assertAlmostEqual(f[12, 16], 0.006, delta=0.002)
        self.assertAlmostEqual(f[0, 0], 0.06, delta=0.002)
        self.assertTrue(np.isscalar(cover.ambient) or np.ndim(cover.ambient) == 0)

    def test_environment_never_writes_the_oxygen_field_directly(self):
        """A low-oxygen spot must not clamp O: consumption still has to compete.

        Inside the spot the field relaxes towards O_am but is pulled below it by
        the agents, so it sits strictly under the imposed level.
        """
        env = envmod.spot((16, 12), 6.0, 0.5, 1.0, width=32, height=24, nx=32, ny=24)
        sim = Simulation(NPR1_LIKE, env, domain=self.domain, n_agents=900,
                         seed=2, record_patterns=False)
        sim.step(200)
        self.assertLess(sim.state.oxygen[12, 16], 0.5)
        self.assertGreater(sim.state.oxygen[12, 16], 0.0)

    def test_validation_rejects_bad_environments(self):
        for kwargs in ({'ambient': 1.5}, {'ambient': -0.1}, {'replenishment': -1},
                       {'oxygen_diffusion': -1}, {'ambient': np.nan}):
            with self.assertRaises(ValueError):
                envmod.Environment(**kwargs)
        with self.assertRaises(ValueError):
            envmod.schedule([])
        with self.assertRaises(ValueError):
            envmod.linear_gradient(0.2, 1.0, 'z', width=8, height=8, nx=8, ny=8)


class PopulationChecks(unittest.TestCase):
    def setUp(self):
        self.domain = (32.0, 24.0, 32, 24)
        self.sim = Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain,
                              n_agents=400, seed=4, record_patterns=False)

    def test_density_integral_tracks_the_agent_count(self):
        for change in (lambda: self.sim.add_agents(250),
                       lambda: self.sim.add_agents(120, 'point', (10, 10), 2.0),
                       lambda: self.sim.remove_agents(300)):
            change()
            p = self.sim.params
            rho = density_grid(self.sim.state.positions, p)
            self.assertAlmostEqual(rho.sum() * p.dx * p.dy, p.n_agents, places=9)
            self.assertEqual(len(self.sim.state.positions), p.n_agents)
            self.assertEqual(len(self.sim.state.headings), p.n_agents)

    def test_population_changes_are_seed_reproducible(self):
        def scripted():
            sim = Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain,
                             n_agents=400, seed=9, record_patterns=False)
            sim.step(10)
            sim.add_agents(150, 'point', (8, 8), 1.5)
            sim.step(10)
            sim.remove_agents(200)
            sim.step(10)
            return sim.state
        a, b = scripted(), scripted()
        np.testing.assert_array_equal(a.positions, b.positions)
        np.testing.assert_array_equal(a.oxygen, b.oxygen)

    def test_changes_are_logged_with_time_and_turn(self):
        self.sim.step(7)
        self.sim.add_agents(50)
        self.sim.step(3)
        self.sim.set_environment(envmod.uniform(0.5))
        self.sim.remove_agents(20)
        events = [(e['event'], e['turn'], e['n_agents']) for e in self.sim.parameter_log]
        self.assertIn(('add_agents', 7, 450), events)
        self.assertIn(('set_environment', 10, 450), events)
        self.assertIn(('remove_agents', 10, 430), events)

    def test_population_floor_and_set_population(self):
        self.sim.remove_agents(10_000)
        self.assertEqual(self.sim.params.n_agents, 1)
        self.sim.set_population(500)
        self.assertEqual(self.sim.params.n_agents, 500)
        self.sim.set_population(120)
        self.assertEqual(self.sim.params.n_agents, 120)
        with self.assertRaises(ValueError):
            self.sim.add_agents(-5)


class EngineChecks(unittest.TestCase):
    def setUp(self):
        self.domain = (32.0, 24.0, 32, 24)

    def test_reproduces_model0_trajectory(self):
        """Model 1 with the Model 0 strain and a uniform environment is Model 0.

        Agreement is to rounding, not to the last bit: deposition sums with
        `np.bincount` instead of `np.add.at`, which changes the summation order.
        """
        p = ce.Parameters(width=32, height=24, nx=32, ny=24, n_agents=400)
        reference = ce.run(params=p, turns=200, seed=3)
        sim = Simulation(MODEL0_DEFAULT, envmod.uniform(1.0, 0.06, 1.0),
                         domain=self.domain, n_agents=400, dt=0.1, seed=3,
                         record_patterns=False)
        sim.step(200)
        np.testing.assert_allclose(sim.state.positions, reference.positions, atol=1e-10)
        np.testing.assert_allclose(sim.state.oxygen, reference.oxygen, atol=1e-12)

    def test_density_grid_matches_model0(self):
        p = ce.Parameters(width=32, height=24, nx=32, ny=24, n_agents=500)
        positions = np.random.default_rng(1).uniform(size=(500, 2)) * (32, 24)
        np.testing.assert_allclose(density_grid(positions, p),
                                   ce.density_grid(positions, p), atol=1e-12)

    def test_chunking_and_drawing_do_not_change_the_trajectory(self):
        def build():
            return Simulation(NPR1_LIKE, envmod.uniform(0.8), domain=self.domain,
                              n_agents=300, seed=6, record_patterns=False)
        a, b = build(), build()
        a.step(35)
        for _ in range(7):
            b.step(5)
            b.result()               # diagnostics must not consume RNG state
        np.testing.assert_array_equal(a.state.positions, b.state.positions)
        np.testing.assert_array_equal(a.state.oxygen, b.state.oxygen)

    def test_reset_restores_initial_conditions(self):
        sim = Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain,
                         n_agents=300, seed=8, record_patterns=False)
        initial = sim.state
        sim.step(30)
        after = sim.state
        sim.reset()
        np.testing.assert_array_equal(sim.state.positions, initial.positions)
        sim.step(30)
        np.testing.assert_array_equal(sim.state.positions, after.positions)
        self.assertFalse(initial.positions.flags.writeable)

    def test_oxygen_stays_positive_and_bounded_with_a_moving_environment(self):
        env = envmod.moving_spot(lambda t: (16 + 8 * np.cos(t / 20), 12), 5.0, 0.1, 1.0,
                                 width=32, height=24, nx=32, ny=24)
        sim = Simulation(NPR1_LIKE, env, domain=self.domain, n_agents=600,
                         dt=0.4, seed=1, record_patterns=False)
        sim.step(150)
        oxygen = sim.state.oxygen
        self.assertTrue(np.isfinite(oxygen).all())
        self.assertTrue((oxygen >= 0).all() and (oxygen <= 1).all())
        self.assertTrue((sim.state.positions >= 0).all())
        self.assertTrue((sim.state.positions < (32, 24)).all())

    def test_initial_oxygen_is_the_local_equilibrium(self):
        sim = Simulation(NPR1_LIKE, envmod.uniform(0.6), domain=self.domain,
                         n_agents=500, seed=1, record_patterns=False)
        expected = st.equilibrium_oxygen(NPR1_LIKE, 0.6, 0.06, sim.density)
        np.testing.assert_allclose(sim.state.oxygen, expected, atol=1e-12)

    def test_constant_speed_strain_stays_uniform(self):
        sim = Simulation(CONSTANT_SPEED, envmod.uniform(1.0), domain=(64.0, 64.0, 64, 64),
                         n_agents=2400, seed=2, record_every=10**9, record_patterns=False)
        sim.step(3000)
        metrics = classify(density_grid(sim.state.positions, sim.params, smoothing=2.0),
                           sim.params.dx, sim.params.dy)
        self.assertEqual(metrics['label'], 'uniform')

    def test_validation(self):
        with self.assertRaises(ValueError):
            Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain, seed=-1)
        with self.assertRaises(ValueError):
            Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain, init='blob')
        with self.assertRaises(ValueError):
            Simulation(NPR1_LIKE, 'not an environment', domain=self.domain)
        with self.assertRaises(ValueError):
            Simulation(NPR1_LIKE, envmod.uniform(1.0), domain=self.domain).step(-1)
        with self.assertRaises(ValueError):
            Strain(name='bad', v_min=2.0, v_max=1.0, oxygen_midpoint=0.5,
                   response_width=0.1, rotational_diffusion=0.2, consumption=0.05,
                   oxygen_half_saturation=0.1, kernel_width=1.0)


class PatternChecks(unittest.TestCase):
    def test_wrapping_stripe_is_one_component(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[:, 5:8] = True                      # a stripe spanning y through the seam
        self.assertEqual(periodic_label(mask, 2)[1], 1)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:8, :] = True                      # and the other orientation
        self.assertEqual(periodic_label(mask, 2)[1], 1)

    def test_blob_split_by_the_seam_is_one_component(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, :3] = True
        mask[8:12, 18:] = True
        self.assertEqual(periodic_label(mask, 2)[1], 1)
        corners = np.zeros((20, 20), dtype=bool)
        corners[:2, :2] = corners[:2, -2:] = corners[-2:, :2] = corners[-2:, -2:] = True
        self.assertEqual(periodic_label(corners, 2)[1], 1)

    def test_separate_blobs_stay_separate(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:9, 5:9] = True
        mask[25:29, 25:29] = True
        self.assertEqual(periodic_label(mask, 2)[1], 2)

    def test_labels_synthetic_images(self):
        self.assertEqual(classify(synthetic('dots'), 1.0, 1.0)['label'], 'dots')
        self.assertEqual(classify(synthetic('stripes'), 1.0, 1.0)['label'], 'stripes')
        self.assertEqual(classify(synthetic('holes'), 1.0, 1.0)['label'], 'holes')

    def test_area_fraction_ordering_of_the_synthetic_images(self):
        phi = {k: classify(synthetic(k), 1.0, 1.0)['area_fraction']
               for k in ('dots', 'stripes', 'holes')}
        self.assertLess(phi['dots'], phi['stripes'])
        self.assertLess(phi['stripes'], phi['holes'])

    def test_euler_descriptor_sign(self):
        dots = classify(synthetic('dots'), 1.0, 1.0)
        holes = classify(synthetic('holes'), 1.0, 1.0)
        self.assertGreater(dots['euler'], 0)
        self.assertLess(holes['euler'], 0)

    def test_noise_alone_is_uniform(self):
        """Shot noise at the real agent density must not read as a pattern.

        The field is an actual Poisson cloud put through the real deposition, so
        this is the same object the engine hands the classifier.
        """
        p = ce.Parameters(width=96, height=96, nx=96, ny=96, n_agents=5400)
        rng = np.random.default_rng(3)
        rho = density_grid(rng.uniform(size=(5400, 2)) * (96, 96), p, smoothing=2.0)
        metrics = classify(rho, p.dx, p.dy, smoothing=2.0)
        self.assertEqual(metrics['label'], 'uniform')
        # And the shot-noise formula predicts the measured level within 25%.
        self.assertAlmostEqual(metrics['heterogeneity'] / metrics['shot_noise_heterogeneity'],
                               1.0, delta=0.25)

    def test_structure_factor_recovers_an_imposed_wavelength(self):
        for period in (16, 24, 32):
            rho = synthetic('stripes', n=128, period=period)
            k, s = structure_factor(rho, 1.0, 1.0)
            measured = 2 * np.pi / k[int(np.argmax(s))]
            # Radial bins are one fundamental wavenumber wide, so a few percent.
            self.assertAlmostEqual(measured / period, 1.0, delta=0.10)

    def test_shot_noise_formula_matches_model0_initial_heterogeneity(self):
        # Model 0 reports H = 0.026-0.036 initially at W = 1800/(64*48), sigma = 2.
        self.assertAlmostEqual(shot_noise_heterogeneity(1800 / (64 * 48), 2.0),
                               0.034, delta=0.004)

    def test_thresholds_are_reported_with_the_label(self):
        metrics = classify(synthetic('dots'), 1.0, 1.0)
        self.assertEqual(metrics['settings']['threshold'], 'otsu')
        self.assertEqual(metrics['settings']['dots_area_fraction'], 0.35)
        for key in ('heterogeneity', 'k_peak', 'area_fraction', 'n_dense', 'n_dilute'):
            self.assertIn(key, metrics)

    def test_both_thresholds_work_and_agree_on_the_synthetic_images(self):
        """Otsu is the default, but the idealised images do not depend on it.

        The two separators only part company on fields whose density contrast is
        asymmetric, which is the case VALIDATION.md documents with numbers.
        """
        for kind in ('dots', 'stripes', 'holes'):
            field = synthetic(kind)
            for threshold in ('otsu', 'mean'):
                metrics = classify(field, 1.0, 1.0, threshold=threshold)
                self.assertEqual(metrics['label'], kind)
                self.assertEqual(metrics['settings']['threshold'], threshold)
        with self.assertRaises(ValueError):
            classify(synthetic('dots'), 1.0, 1.0, threshold='median')

    def test_coarsening_fit(self):
        times = np.geomspace(10, 1000, 30)
        lengths = 3.0 * times ** 0.33
        exponent, _, n = coarsening_exponent(times, lengths)
        self.assertAlmostEqual(exponent, 0.33, places=6)
        self.assertGreater(n, 5)
        self.assertTrue(np.isnan(coarsening_exponent([1, 2], [1, 2])[0]))


class SweepChecks(unittest.TestCase):
    def test_seeds_are_stable_and_distinct(self):
        config = m1.sweep.SweepConfig(strain='NPR1_LIKE')
        seeds = [t['seed'] for t in config.tasks()]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertEqual(seeds, [t['seed'] for t in m1.sweep.SweepConfig(strain='NPR1_LIKE').tasks()])
        self.assertNotEqual(seeds[0], m1.sweep.SweepConfig(strain='N2_LIKE').seed_for(0, 0, 0))

    def test_plan_warns_when_the_box_is_too_small(self):
        tight = m1.sweep.SweepConfig(strain='NPR1_LIKE', width=40.0, height=40.0,
                                     nx=40, ny=40, ambients=(1.0,), densities=(0.6,))
        _, warnings = m1.sweep.plan(tight)
        self.assertTrue(any('per box side' in w for w in warnings))
        roomy = m1.sweep.SweepConfig(strain='NPR1_LIKE', width=400.0, height=400.0,
                                     nx=400, ny=400, ambients=(1.0,), densities=(0.6,))
        self.assertEqual([w for w in m1.sweep.plan(roomy)[1] if 'per box side' in w], [])

    def test_majority_and_agreement(self):
        rows = []
        for rep, label in enumerate(('dots', 'dots', 'uniform')):
            rows.append(dict(i_ambient=0, i_density=0, rep=rep, label=label,
                             ambient=1.0, density=0.6, predicted_unstable=1))
        summary = m1.sweep.majority_labels(rows)
        self.assertEqual(summary['label'][0, 0], 'dots')
        self.assertAlmostEqual(summary['agreement'][0, 0], 2 / 3)


if __name__ == '__main__':
    unittest.main()


class ControlChecks(unittest.TestCase):
    """Inverse control: the geometry helpers and the optimiser, not the physics.

    A full steering trial is a minute of simulation, so it is exercised by
    `scripts/steer.py` rather than by the test suite; what is checked here is
    that the pieces it relies on are correct.
    """

    def test_periodic_center_of_mass_handles_the_seam(self):
        from celegans.model1.control import periodic_center_of_mass, periodic_distance
        side = 72.0
        straddling = np.array([[1.0, 1.0], [71.0, 71.0]])
        com = periodic_center_of_mass(straddling, side)
        # The true midpoint is the origin, not the middle of the domain.
        self.assertLess(periodic_distance(com, (0.0, 0.0), side), 1e-9)
        blob = np.array([[30.0, 40.0], [32.0, 42.0], [31.0, 41.0]])
        np.testing.assert_allclose(periodic_center_of_mass(blob, side), [31, 41], atol=1e-6)

    def test_periodic_distance_takes_the_short_way(self):
        from celegans.model1.control import periodic_distance
        self.assertAlmostEqual(periodic_distance((1, 0), (71, 0), 72.0), 2.0)
        self.assertAlmostEqual(periodic_distance((0, 0), (36, 0), 72.0), 36.0)

    def test_parameter_squash_round_trips_and_respects_bounds(self):
        from celegans.model1.control import clamp, unclamp, BOUNDS, PARAMETER_ORDER
        for vector in ([0, 0, 0], [3, -3, 1.5], [-8, 8, 0]):
            params = clamp(vector)
            for name in PARAMETER_ORDER:
                lo, hi = BOUNDS[name]
                self.assertGreater(params[name], lo)
                self.assertLess(params[name], hi)
            np.testing.assert_allclose(unclamp(params), vector, atol=1e-6)

    def test_path_is_held_during_nucleation_then_travels(self):
        from celegans.model1.control import SteeringTask
        task = SteeringTask(side=100.0, cells=100, nucleate_time=50.0, path_radius=30.0)
        start = task.target(0.0, 0.1)
        self.assertEqual(task.target(25.0, 0.1), start)   # held while nucleating
        self.assertEqual(task.target(50.0, 0.1), start)
        moved = task.target(150.0, 0.1)
        self.assertGreater(np.hypot(moved[0] - start[0], moved[1] - start[1]), 1.0)
        # It stays on the circle it was given.
        self.assertAlmostEqual(np.hypot(moved[0] - 50.0, moved[1] - 50.0), 30.0, places=9)

    def test_cmaes_minimises_a_quadratic(self):
        from celegans.model1.control import cmaes
        target = np.array([1.5, -0.5, 0.25])

        def objective(samples):
            return [float(np.sum((np.asarray(x) - target) ** 2)) for x in samples]

        outcome = cmaes(objective, np.zeros(3), 1.0, generations=60, population=10,
                        rng=np.random.default_rng(1))
        self.assertLess(outcome['value'], 1e-4)
        np.testing.assert_allclose(outcome['x'], target, atol=2e-2)
        self.assertEqual(len(outcome['history']), 60)
