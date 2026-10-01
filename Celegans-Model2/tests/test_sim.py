"""Numeric checks for Model 2. Run: python -m pytest tests -q"""
import numpy as np
import pytest
import taichi as ti

from model2 import params as P, sim


@pytest.fixture(scope='module')
def s():
    sim.init('gpu')
    return sim.Simulation(density=40)


def test_paper_parabola():
    # Paper Eq. 47 (A1): ~225 um/s at 0 %, ~16 um/s minimum at 10.5 %, ~222 um/s at 21 %
    assert P.NPR1.speed(0.0) * 1e3 == pytest.approx(225, abs=1)
    assert P.NPR1.speed(0.105) * 1e3 == pytest.approx(15.5, abs=1)
    assert P.NPR1.speed(0.21) * 1e3 == pytest.approx(222, abs=2)


def test_density_conserves_worms(s):
    s.density = 40
    s.reset(seed=1)
    s.step(1)
    area = (P.BOX / P.DENSITY_GRID) ** 2
    assert s.W.to_numpy().sum() * area == pytest.approx(s.n, rel=1e-3)


def test_worms_stay_in_box(s):
    s.density = 40
    s.reset(seed=2)
    s.step(200)
    pos = s.pos.to_numpy()[:s.n]
    assert pos.min() >= 0 and pos.max() < P.BOX


def test_no_worms_relaxes_to_ambient(s):
    s.density = 0
    s.ambient = 0.14
    s.reset()
    s.O.fill(0.05)
    s.step(400)                     # 20 s >> 1/f
    assert s.O.to_numpy().mean() == pytest.approx(0.14, abs=1e-4)


def test_uniform_oxygen_matches_paper(s):
    # Uniform state of Eq. 46: O_eq = O_am - k_c W / f
    s.density = 20
    s.ambient = 0.21
    s.reset(seed=3)
    s.step(200)
    expected = 0.21 - P.K_C * 20 / P.F
    assert s.O.to_numpy().mean() == pytest.approx(expected, rel=0.01)

