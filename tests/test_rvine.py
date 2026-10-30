import numpy as np
import pytest

from pf.models.copulas import PairCopula, kendall_tau
from pf.models.rvine import Edge, RVine, pseudo_observations


def factor_data(n=3000, d=7, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_t(4, size=(n, 1))
    X = 0.8 * f + rng.standard_normal((n, d)) * np.linspace(0.4, 1.3, d)
    X[:, 2] *= -1
    return pseudo_observations(X)


@pytest.fixture(scope="module")
def vine():
    return RVine.fit(factor_data(), truncation=3)


def test_structure_is_valid_rvine(vine):
    d = vine.d
    assert [len(t) for t in vine.trees] == list(range(d - 1, 0, -1))
    for k, level in enumerate(vine.trees, start=1):
        for e in level:
            assert len(e.cond) == k - 1 and e.a != e.b and e.a not in e.cond and e.b not in e.cond
    # warunek bliskości: krawędź drzewa k+1 łączy krawędzie drzewa k o wspólnym węźle
    for k in range(1, d - 1):
        fulls = {e.full for e in vine.trees[k - 1]}
        for e in vine.trees[k]:
            assert e.cond | {e.a} in fulls and e.cond | {e.b} in fulls


def test_truncation(vine):
    assert all(e.copula.family == "indep" for level in vine.trees[3:] for e in level)
    assert any(e.copula.family != "indep" for e in vine.trees[2])
    assert len(vine.summary()) == sum(len(t) for t in vine.trees[:3])


def test_first_tree_is_maximum_spanning_tree(vine):
    U = factor_data()
    d = U.shape[1]
    tau = np.array([[abs(kendall_tau(U[:, i], U[:, j])[0]) for j in range(d)] for i in range(d)])
    chosen = sum(tau[e.a, e.b] for e in vine.trees[0])
    # porównanie z drzewem-gwiazdą wokół każdego węzła (dowolne drzewo rozpinające ≤ maksymalne)
    for root in range(d):
        assert chosen >= sum(tau[root, j] for j in range(d) if j != root) - 1e-12


def test_sampling_plan_covers_all_variables(vine):
    plan = vine.sampling_plan()
    assert sorted(a for a, _ in plan) == list(range(vine.d))
    simulated = set()
    for a, chain in reversed(plan):
        for e in chain:
            assert e.other(a) in simulated and e.cond <= simulated
        simulated.add(a)


def test_simulation_reproduces_dependence(vine):
    U = factor_data()
    S = vine.simulate(6000, np.random.default_rng(1))
    assert S.shape == (6000, vine.d)
    np.testing.assert_allclose(S.mean(axis=0), 0.5, atol=0.02)
    for e in vine.trees[0]:
        assert kendall_tau(S[:, e.a], S[:, e.b])[0] == pytest.approx(kendall_tau(U[:, e.a], U[:, e.b])[0],
                                                                     abs=0.04)


def test_known_dvine_round_trip():
    """D-vine 0–1–2–3 z zadanymi kopułami: estymacja odtwarza rodziny i τ."""
    trees = [[Edge(1, 0, 1, frozenset(), PairCopula("clayton", 0, (2 * 0.7 / 0.3,))),
              Edge(1, 1, 2, frozenset(), PairCopula("gumbel", 0, (1 / 0.3,))),
              Edge(1, 2, 3, frozenset(), PairCopula("gumbel", 90, (1 / 0.3,)))],
             [Edge(2, 0, 2, frozenset({1}), PairCopula("gauss", 0, (0.2,))),
              Edge(2, 1, 3, frozenset({2}), PairCopula("t", 0, (0.2, 4.0)))],
             [Edge(3, 0, 3, frozenset({1, 2}), PairCopula("indep"))]]
    true = RVine([str(i) for i in range(4)], trees, truncation=3)
    U = true.simulate(8000, np.random.default_rng(5))
    for e in trees[0]:
        assert kendall_tau(U[:, e.a], U[:, e.b])[0] == pytest.approx(e.copula.tau(), abs=0.03)
    fit = RVine.fit(U, truncation=3)
    got = {frozenset((e.a, e.b)): e for e in fit.trees[0]}
    for e in trees[0]:
        f = got[frozenset((e.a, e.b))]
        assert f.copula.family == e.copula.family
        assert f.copula.tau() == pytest.approx(e.copula.tau(), abs=0.03)


def test_serialization(vine, tmp_path):
    v2 = RVine.load(vine.save(tmp_path / "v.json.gz"))
    np.testing.assert_allclose(v2.simulate(50, np.random.default_rng(0)), vine.simulate(50, np.random.default_rng(0)))
    assert v2.loglik() == pytest.approx(vine.loglik())
