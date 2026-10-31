"""Kopuła R-vine obcięta na zadanym piętrze: wybór struktury, estymacja i symulacja.

Wybór struktury – algorytm Dißmanna i in. (2013), drzewo po drzewie:
    T₁: graf pełny na zmiennych, wagi |τ̂ᵢⱼ|, maksymalne drzewo rozpinające;
    T_{k+1}: węzłami są krawędzie T_k; kandydatami krawędzie spełniające warunek
    bliskości (proximity) – łączone węzły mają wspólny węzeł w T_k; wagi |τ̂| obliczane
    na pseudo-obserwacjach warunkowych F(a | D), F(b | D) uzyskanych h-funkcjami z T_k.
Dla każdej krawędzi (a, b | D) dobierana jest kopuła parowa (``copulas.select``):
parametry z odwrócenia τ Kendalla (``itau``) lub sekwencyjnej MNW startującej z tego punktu
(``mle``), rodzina (Gauss, t, Clayton, Gumbel z obrotami, Frank) z AIC.

Struktury narzucone (``structure``), budowane tym samym algorytmem z innym wyborem drzew:
    cvine – C-vine: w drzewie T_k gwiazda wokół korzenia r_k maksymalizującego Σ_j |τ̂(r_k, j | r₁…r_{k−1})|
            (krawędzie (r_k, j | r₁, …, r_{k−1}));
    dvine – D-vine: T₁ jest ścieżką Hamiltona o maksymalnej Σ|τ̂| między sąsiadami (heurystyka zachłanna
            krawędziowa + poprawa 2-opt); kolejne drzewa wynikają z warunku bliskości (ścieżki).
Wyższe drzewa poza obcięciem (niezależność) uzupełniane są strukturą zgodną z typem vine'a.

Gęstość (``logpdf``): log c(u) = Σ_{e ∈ T₁…T_L} log c_e(F(a_e | D_e), F(b_e | D_e)), z argumentami
liczonymi rekurencyjnie h-funkcjami, jak przy estymacji.

Obcięcie (truncation) na piętrze L: kopuły w drzewach T_{L+1}, …, T_{d−1} są kopułami
niezależności. Struktura tych drzew jest jednak uzupełniana (dowolne drzewa spełniające
warunek bliskości), bo wyznacza porządek symulacji.

Symulacja: zmienne zdejmowane są kolejno z najwyższego drzewa (zmienna ze zbioru
warunkowanego jedynej krawędzi drzewa najwyższego występuje w zbiorze warunkowanym
dokładnie jednej krawędzi każdego drzewa – odpowiada to kolumnie macierzy R-vine).
Dla zmiennej a z partnerami p_j (drzewo j, zbiór warunkujący D_j = {p₁, …, p_{j−1}}):
    W ~ U(0, 1),   F(a | D_j) = h⁻¹_{a,p_j|D_j}(F(a | D_{j+1}) | F(p_j | D_j)),  j = L, …, 1,
a F(p_j | D_j) liczone są rekurencyjnie h-funkcjami z wartości już zasymulowanych.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pf.models.copulas import PairCopula, kendall_tau, select


@dataclass
class Edge:
    tree: int                      # numer drzewa (1, 2, …)
    a: int                         # zmienne warunkowane; kopuła C(F(a|D), F(b|D))
    b: int
    cond: frozenset                # zbiór warunkujący D
    copula: PairCopula = field(default_factory=PairCopula)
    tau: float = 0.0               # τ̂ Kendalla pseudo-obserwacji
    loglik: float = 0.0

    @property
    def full(self) -> frozenset:
        return self.cond | {self.a, self.b}

    def h_for(self, target: int, u_target, u_other) -> np.ndarray:
        """F(target | D ∪ {other}) z F(target | D) i F(other | D)."""
        if target == self.a:
            return self.copula.h(u_target, u_other)
        return self.copula.h1(u_other, u_target)

    def hinv_for(self, target: int, w, u_other) -> np.ndarray:
        if target == self.a:
            return self.copula.hinv(w, u_other)
        return self.copula.hinv1(w, u_other)

    def other(self, var: int) -> int:
        return self.b if var == self.a else self.a


# --------------------------------------------------------------------------- pomocnicze

def max_spanning_tree(n_nodes: int, edges: list[tuple[float, int, int]]) -> list[tuple[int, int]]:
    """Kruskal: maksymalne drzewo rozpinające z listy (waga, i, j)."""
    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    out = []
    for _, i, j in sorted(edges, key=lambda e: -e[0]):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            out.append((i, j))
            if len(out) == n_nodes - 1:
                break
    if len(out) != n_nodes - 1:
        raise RuntimeError("Graf kandydatów niespójny – brak drzewa rozpinającego")
    return out


_TAU_CACHE: dict[bytes, np.ndarray] = {}
_TAU_CACHE_SIZE = 8


def kendall_matrix(U: np.ndarray) -> np.ndarray:
    """Macierz τ̂ Kendalla kolumn U (z pamięcią podręczną – wspólna dla modeli estymowanych na tym samym U)."""
    U = np.ascontiguousarray(U, dtype=float)
    key = hashlib.sha1(U.tobytes()).digest() + np.array(U.shape).tobytes()
    if key not in _TAU_CACHE:
        d = U.shape[1]
        T = np.eye(d)
        for i in range(d):
            for j in range(i + 1, d):
                T[i, j] = T[j, i] = kendall_tau(U[:, i], U[:, j])[0]
        if len(_TAU_CACHE) >= _TAU_CACHE_SIZE:
            _TAU_CACHE.pop(next(iter(_TAU_CACHE)))
        _TAU_CACHE[key] = T
    return _TAU_CACHE[key].copy()


def best_star(n_nodes: int, edges: list[tuple[float, int, int]]) -> list[tuple[int, int]] | None:
    """Gwiazda (drzewo rozpinające o jednym centrum) o maksymalnej sumie wag; None, gdy żadna nie istnieje."""
    incident: dict[int, list[tuple[float, int, int]]] = {i: [] for i in range(n_nodes)}
    for w, i, j in edges:
        incident[i].append((w, i, j))
        incident[j].append((w, i, j))
    best, best_w = None, -np.inf
    for c in range(n_nodes):
        if len(incident[c]) != n_nodes - 1:
            continue
        total = sum(w for w, _, _ in incident[c])
        if total > best_w + 1e-15:
            best, best_w = [(i, j) for _, i, j in incident[c]], total
    return best


def max_hamiltonian_path(W: np.ndarray) -> list[int]:
    """Ścieżka Hamiltona o (przybliżenie) maksymalnej Σ W między sąsiadami: zachłanna + 2-opt."""
    d = len(W)
    if d <= 2:
        return list(range(d))
    parent = list(range(d))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = np.zeros(d, dtype=int)
    adj: dict[int, list[int]] = {i: [] for i in range(d)}
    order = sorted(((W[i, j], i, j) for i in range(d) for j in range(i + 1, d)), key=lambda e: -e[0])
    n_edges = 0
    for _, i, j in order:
        if deg[i] < 2 and deg[j] < 2 and find(i) != find(j):
            parent[find(i)] = find(j)
            deg[i] += 1
            deg[j] += 1
            adj[i].append(j)
            adj[j].append(i)
            n_edges += 1
            if n_edges == d - 1:
                break
    start = int(np.flatnonzero(deg == 1)[0])
    path, prev = [start], -1
    while len(path) < d:
        nxt = [k for k in adj[path[-1]] if k != prev][0]
        prev = path[-1]
        path.append(nxt)
    # 2-opt: odwrócenie odcinka path[i..j] zmienia krawędzie (p_{i−1}, p_i) i (p_j, p_{j+1})
    improved = True
    while improved:
        improved = False
        for i in range(d - 1):
            for j in range(i + 1, d):
                delta = 0.0
                if i > 0:
                    delta += W[path[i - 1], path[j]] - W[path[i - 1], path[i]]
                if j < d - 1:
                    delta += W[path[i], path[j + 1]] - W[path[j], path[j + 1]]
                if delta > 1e-12:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
    return path


def path_weight(W: np.ndarray, path: list[int]) -> float:
    return float(sum(W[a, b] for a, b in zip(path[:-1], path[1:])))


def first_tree(tau_abs: np.ndarray, structure: str) -> list[tuple[int, int]]:
    d = len(tau_abs)
    if structure == "cvine":
        root = int(np.argmax(tau_abs.sum(axis=1) - np.diag(tau_abs)))
        return [(root, j) for j in range(d) if j != root]
    if structure == "dvine":
        path = max_hamiltonian_path(tau_abs)
        return list(zip(path[:-1], path[1:]))
    if structure != "rvine":
        raise ValueError(f"Nieznana struktura vine: {structure}")
    return max_spanning_tree(d, [(tau_abs[i, j], i, j) for i in range(d) for j in range(i + 1, d)])


def next_tree(n_nodes: int, weighted: list[tuple[float, int, int]], structure: str) -> list[tuple[int, int]]:
    if structure == "cvine":
        star = best_star(n_nodes, weighted)
        if star is not None:
            return star
    return max_spanning_tree(n_nodes, weighted)


def proximity_candidates(prev: list[Edge]) -> list[tuple[int, int]]:
    """Pary krawędzi drzewa T_k (indeksy), które mają wspólny węzeł T_k (|full₁ ∩ full₂| = k)."""
    k = prev[0].tree
    out = []
    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            if len(prev[i].full & prev[j].full) == k:
                out.append((i, j))
    return out


def join(e1: Edge, e2: Edge, tree: int) -> tuple[int, int, frozenset]:
    cond = e1.full & e2.full
    (a,), (b,) = e1.full - cond, e2.full - cond
    return a, b, frozenset(cond)


# --------------------------------------------------------------------------- R-vine

@dataclass
class RVine:
    names: list[str]
    trees: list[list[Edge]]
    truncation: int
    n_obs: int = 0
    structure: str = "rvine"

    @property
    def d(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ estymacja
    @classmethod
    def fit(cls, U: np.ndarray, names: list[str] | None = None, truncation: int = 3,
            families=("gauss", "t", "clayton", "gumbel"), rotations: bool = True,
            criterion: str = "aic", indep_level: float | None = 0.05,
            t_df_bounds=(2.05, 30.0), estimation: str = "itau", structure: str = "rvine",
            tau_matrix: np.ndarray | None = None) -> "RVine":
        """Estymacja na pseudo-obserwacjach U (n × d) z (0, 1); ``structure``: rvine | cvine | dvine."""
        n, d = U.shape
        names = list(names) if names is not None else [str(i) for i in range(d)]
        kw = dict(families=families, rotations=rotations, criterion=criterion,
                  indep_level=indep_level, t_df_bounds=t_df_bounds, estimation=estimation)
        trees: list[list[Edge]] = []
        # pseudo[(var, D)] = F(var | D)
        pseudo: dict[tuple[int, frozenset], np.ndarray] = {(i, frozenset()): U[:, i] for i in range(d)}

        # T₁
        tau_abs = np.abs(tau_matrix if tau_matrix is not None else kendall_matrix(U))
        pairs = first_tree(tau_abs, structure)
        trees.append([cls._fit_edge(1, i, j, frozenset(), pseudo, kw) for i, j in pairs])

        for t in range(2, d):
            prev = trees[-1]
            cand_idx = proximity_candidates(prev)
            fitted = t <= truncation
            weighted = []
            for i, j in cand_idx:
                if fitted:
                    a, b, D = join(prev[i], prev[j], t)
                    w = abs(kendall_tau(pseudo[(a, D)], pseudo[(b, D)])[0])
                else:
                    w = 0.0
                weighted.append((w, i, j))
            sel = next_tree(len(prev), weighted, structure)
            level = []
            for i, j in sel:
                a, b, D = join(prev[i], prev[j], t)
                if fitted:
                    level.append(cls._fit_edge(t, a, b, D, pseudo, kw))
                else:
                    level.append(Edge(t, a, b, D))
            trees.append(level)
        return cls(names, trees, truncation, n_obs=n, structure=structure)

    @staticmethod
    def _fit_edge(t, a, b, D, pseudo, kw) -> Edge:
        u, v = pseudo[(a, D)], pseudo[(b, D)]
        cop, info = select(u, v, **kw)
        e = Edge(t, a, b, D, cop, info["tau"], info["loglik"])
        pseudo[(a, D | {b})] = cop.h(u, v)
        pseudo[(b, D | {a})] = cop.h1(u, v)
        return e

    # ------------------------------------------------------------------ podsumowania
    def edges(self, max_tree: int | None = None) -> list[Edge]:
        m = max_tree or self.truncation
        return [e for level in self.trees[:m] for e in level]

    def loglik(self) -> float:
        return float(sum(e.loglik for e in self.edges()))

    def n_params(self) -> int:
        return sum(e.copula.n_params for e in self.edges())

    def aic(self) -> float:
        return 2.0 * self.n_params() - 2.0 * self.loglik()

    def bic(self) -> float:
        return np.log(max(self.n_obs, 1)) * self.n_params() - 2.0 * self.loglik()

    def logpdf(self, U: np.ndarray) -> np.ndarray:
        """log c(u_t) dla wierszy U (n × d) – gęstość kopuły R-vine (drzewa 1…L)."""
        U = np.asarray(U, dtype=float)
        pseudo: dict[tuple[int, frozenset], np.ndarray] = {(i, frozenset()): U[:, i] for i in range(self.d)}
        out = np.zeros(len(U))
        top = min(self.truncation, len(self.trees))
        for k, level in enumerate(self.trees[:top], start=1):
            for e in level:
                u, v = pseudo[(e.a, e.cond)], pseudo[(e.b, e.cond)]
                if e.copula.family != "indep":
                    out += e.copula.logpdf(u, v)
                if k < top:
                    pseudo[(e.a, e.cond | {e.b})] = e.copula.h(u, v)
                    pseudo[(e.b, e.cond | {e.a})] = e.copula.h1(u, v)
        return out

    def summary(self) -> list[dict]:
        rows = []
        for e in self.edges():
            lam_l, lam_u = e.copula.tail_dependence()
            rows.append({
                "tree": e.tree, "a": self.names[e.a], "b": self.names[e.b],
                "cond": ",".join(sorted(self.names[c] for c in e.cond)),
                "family": e.copula.label(), "rotation": e.copula.rotation,
                "par": e.copula.params[0] if e.copula.params else np.nan,
                "par2": e.copula.params[1] if e.copula.n_params > 1 else np.nan,
                "tau_emp": e.tau, "tau_model": e.copula.tau(), "loglik": e.loglik,
                "lambda_L": lam_l, "lambda_U": lam_u,
            })
        return rows

    # ------------------------------------------------------------------ symulacja
    def sampling_plan(self) -> list[tuple[int, list[Edge]]]:
        """Kolejność zdejmowania zmiennych [(a, [krawędź w drzewie m−1, …, w drzewie 1])]."""
        remaining = [list(level) for level in self.trees]
        active = set(range(self.d))
        plan = []
        while len(active) > 1:
            m = len(active)
            top = remaining[m - 2]
            assert len(top) == 1, "niepoprawna struktura R-vine"
            a = top[0].a
            chain = []
            for t in range(m - 1, 0, -1):
                hits = [e for e in remaining[t - 1] if a in (e.a, e.b)]
                assert len(hits) == 1 and a not in hits[0].cond, "zmienna nie jest liściem struktury"
                chain.append(hits[0])
                remaining[t - 1].remove(hits[0])
            for level in remaining[:m - 1]:
                assert all(a not in e.full for e in level)
            plan.append((a, chain))
            active.remove(a)
        plan.append((active.pop(), []))
        return plan

    def simulate(self, n: int, rng: np.random.Generator | None = None, W: np.ndarray | None = None) -> np.ndarray:
        """n obserwacji z kopuły R-vine (n × d); ``W`` – opcjonalne wejściowe U(0,1) (n × d)."""
        rng = rng or np.random.default_rng()
        W = rng.random((n, self.d)) if W is None else W
        by_full = {e.full: e for e in self.edges()}
        U: dict[int, np.ndarray] = {}
        memo: dict[tuple[int, frozenset], np.ndarray] = {}

        def cond_cdf(v: int, D: frozenset) -> np.ndarray:
            if not D:
                return U[v]
            key = (v, D)
            if key not in memo:
                e = by_full.get(D | {v})
                if e is None:  # krawędź poza obcięciem – kopuła niezależności
                    raise KeyError(key)
                x = e.other(v)
                rest = D - {x}
                memo[key] = e.h_for(v, cond_cdf(v, rest), cond_cdf(x, rest))
            return memo[key]

        for a, chain in reversed(self.sampling_plan()):
            x = W[:, a]
            for e in chain:  # od najwyższego drzewa do T₁
                if e.tree > self.truncation or e.copula.family == "indep":
                    continue
                p = e.other(a)
                x = e.hinv_for(a, x, cond_cdf(p, e.cond))
            U[a] = x
        return np.column_stack([U[i] for i in range(self.d)])

    # ------------------------------------------------------------------ serializacja
    def to_dict(self) -> dict:
        return {"names": self.names, "truncation": self.truncation, "n_obs": self.n_obs, "structure": self.structure,
                "trees": [[{"a": e.a, "b": e.b, "cond": sorted(e.cond), "tau": e.tau, "loglik": e.loglik,
                            **e.copula.to_dict()} for e in level] for level in self.trees]}

    @classmethod
    def from_dict(cls, d: dict) -> "RVine":
        trees = [[Edge(t + 1, e["a"], e["b"], frozenset(e["cond"]), PairCopula.from_dict(e), e["tau"],
                       e["loglik"]) for e in level] for t, level in enumerate(d["trees"])]
        return cls(d["names"], trees, d["truncation"], d.get("n_obs", 0), d.get("structure", "rvine"))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def save(self, path: Path) -> Path:
        """Zapis JSON (``.json.gz`` – skompresowany)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_json().encode("utf-8")
        path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
        return path

    @classmethod
    def load(cls, path: Path) -> "RVine":
        raw = path.read_bytes()
        return cls.from_dict(json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw))


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Rangi skalowane n/(n+1) w kolumnach (empiryczna dystrybuanta)."""
    from scipy.stats import rankdata
    return rankdata(x, axis=0) / (x.shape[0] + 1.0)
