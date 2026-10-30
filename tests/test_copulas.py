import numpy as np
import pytest
from scipy import integrate

from pf.models.copulas import PairCopula, kendall_tau, param_from_tau, select

CASES = [("gauss", 0), ("t", 0), ("clayton", 0), ("clayton", 90), ("clayton", 180), ("clayton", 270),
         ("gumbel", 0), ("gumbel", 90), ("gumbel", 180), ("gumbel", 270)]


def make(family, rotation, tau=0.5):
    signed = -tau if rotation in (90, 270) else tau
    p = param_from_tau(family, signed if family in ("gauss", "t") else tau)
    return PairCopula(family, rotation, (p[0], 5.0) if family == "t" else p)


@pytest.mark.parametrize("family,rotation", CASES)
def test_h_is_integral_of_density(family, rotation):
    c = make(family, rotation)
    for u, v in [(0.3, 0.7), (0.05, 0.9), (0.8, 0.2)]:
        hq = integrate.quad(lambda s: np.exp(c.logpdf(s, v)), 0, u, limit=200)[0]
        h1q = integrate.quad(lambda s: np.exp(c.logpdf(u, s)), 0, v, limit=200)[0]
        assert float(c.h(u, v)) == pytest.approx(hq, abs=1e-6)
        assert float(c.h1(u, v)) == pytest.approx(h1q, abs=1e-6)


@pytest.mark.parametrize("family,rotation", CASES)
def test_density_integrates_to_one(family, rotation):
    c = make(family, rotation, tau=0.3)
    total = integrate.dblquad(lambda v, u: np.exp(c.logpdf(u, v)), 0, 1, 0, 1, epsabs=1e-5)[0]
    assert total == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("family,rotation", CASES)
def test_hinv_round_trip_and_sample_tau(family, rotation):
    c = make(family, rotation)
    rng = np.random.default_rng(0)
    w, v = rng.random(20_000), rng.random(20_000)
    u = c.hinv(w, v)
    np.testing.assert_allclose(c.h(u, v), w, atol=1e-9)
    np.testing.assert_allclose(c.h1(v, c.hinv1(w, v)), w, atol=1e-9)
    assert kendall_tau(u, v)[0] == pytest.approx(c.tau(), abs=0.02)


@pytest.mark.parametrize("tau", [0.02, 0.3, 0.9, 0.95])
def test_strong_and_weak_dependence_is_finite(tau):
    rng = np.random.default_rng(1)
    u, v = rng.random(1000), rng.random(1000)
    for fam in ("gauss", "clayton", "gumbel"):
        for rot in ((0,) if fam == "gauss" else (0, 90, 180, 270)):
            c = PairCopula(fam, rot, param_from_tau(fam, tau))
            assert np.isfinite(c.logpdf(u, v)).all() and np.isfinite(c.hinv(u, v)).all()


def test_tail_dependence_closed_forms():
    th = 2.0
    assert PairCopula("clayton", 0, (th,)).tail_dependence() == pytest.approx((2 ** (-1 / th), 0.0))
    assert PairCopula("gumbel", 180, (th,)).tail_dependence() == pytest.approx((2 - 2 ** (1 / th), 0.0))
    lam_l, lam_u = PairCopula("t", 0, (0.5, 4.0)).tail_dependence()
    assert lam_l == lam_u and 0 < lam_l < 1


@pytest.mark.parametrize("family,rotation", [("clayton", 0), ("gumbel", 0), ("clayton", 270), ("t", 0)])
def test_select_recovers_family(family, rotation):
    true = make(family, rotation, tau=0.5) if family != "t" else PairCopula("t", 0, (0.7, 3.0))
    rng = np.random.default_rng(2)
    v = rng.random(4000)
    u = true.hinv(rng.random(4000), v)
    cop, info = select(u, v)
    assert (cop.family, cop.rotation) == (true.family, true.rotation)
    assert info["tau"] == pytest.approx(true.tau(), abs=0.03)
    if family == "t":
        assert cop.params[1] == pytest.approx(3.0, abs=1.0)


def test_select_independence():
    rng = np.random.default_rng(3)
    cop, info = select(rng.random(2000), rng.random(2000))
    assert cop.family == "indep" and info["pvalue"] > 0.05


def test_serialization_and_swap():
    c = PairCopula("gumbel", 90, (2.5,))
    assert PairCopula.from_dict(c.to_dict()) == c
    assert c.swap().rotation == 270 and c.swap().swap() == c
    u, v = np.array([0.2, 0.6]), np.array([0.7, 0.1])
    np.testing.assert_allclose(c.logpdf(u, v), c.swap().logpdf(v, u))
