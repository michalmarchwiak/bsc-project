import numpy as np
from pf.opt.cvar import min_es, max_return_es, empirical_es
from pf.opt.utility import ce_and_grad
from pf.opt.dependence_penalty import dependence_matrix
from pf.opt.profiles import solve_ce
from pf.sim.scenarios import es_with_se

# 1. Tworzymy 10 000 sztucznych scenariuszy miesięcznych stóp zwrotu dla 3 aktywów:
# Kolumna 0: Obligacja (stałe +0.5%)
# Kolumna 1: ETF_A (umiarkowane ryzyko)
# Kolumna 2: ETF_B (wysokie ryzyko, krachy razem z ETF_A)
rng = np.random.default_rng(42)
S = 10_000

obl = np.full(S, 0.005)
szok_rynkowy = rng.standard_t(df=3, size=S) * 0.04  # wspólny czynnik krachu (grube ogony)
etf_a = 0.015 + szok_rynkowy + rng.normal(0, 0.02, S)
etf_b = 0.025 + 2.0 * szok_rynkowy + rng.normal(0, 0.04, S)

R = np.column_stack([obl, etf_a, etf_b])  # macierz scenariuszy (10000 x 3)
nazwy = ["OBL", "ETF_A", "ETF_B"]

print("=== 1. MIARY RYZYKA (VaR i ES 95%) DLA PORTFELA RÓWNYCH WAG ===")
w_ew = np.array([1/3, 1/3, 1/3])
straty_ew = -(R @ w_ew)  # strata = -zwrot
var_95, es_95, se_es = es_with_se(straty_ew, alpha=0.95)
print(f"VaR 95%: {var_95:.2%} | ES 95%: {es_95:.2%} | Błąd standardowy ES: {se_es:.4%}")

print("\n=== 2. OPTYMALIZACJA LP ROCKAFELLARA-URYASEVA (Minimalny ES) ===")
sol_min_es = min_es(R, alpha=0.95, w_max=0.8)  # max 80% w jedno aktywo
for nazwa, waga in zip(nazwy, sol_min_es.weights):
    print(f"  {nazwa}: {waga:.1%}")
print(f"Oczekiwany zwrot: {sol_min_es.expected_return:.2%}, Minimalny ES 95%: {sol_min_es.es:.2%}")

print("\n=== 3. MAX ZWROT PRZY LIMICIE ES <= 5% ===")
sol_max_ret = max_return_es(R, alpha=0.95, es_limit=0.05, w_max=0.8)
for nazwa, waga in zip(nazwy, sol_max_ret.weights):
    print(f"  {nazwa}: {waga:.1%}")
print(f"Oczekiwany zwrot: {sol_max_ret.expected_return:.2%}, Osiągnięty ES 95%: {sol_max_ret.es:.2%}")

print("\n=== 4. EKWIWALENT PEWNOŚCI (CE) Z KARĄ ZA ZALEŻNOŚĆ W DOLNYM OGONIE ===")
M_ogon = dependence_matrix(R, measure="lambda_L", q=0.05)
print("Macierz zależności w dolnym ogonie (5% najgorszych scenariuszy):")
print(np.round(M_ogon, 2))

# Porównanie portfela bez kary (kappa=0) i z karą za wspólne krachy (kappa=0.5)
w_bez_kary, _ = solve_ce(R, gamma=4.0, kappa=0.0, ub=0.8)
w_z_kara, _   = solve_ce(R, gamma=4.0, kappa=0.5, ub=0.8, M=M_ogon, sigma=straty_ew.std())

print("\nWagi CE bez kary za ogony (kappa=0):", np.round(w_bez_kary, 3))
print("Wagi CE z karą za ogony (kappa=0.5):", np.round(w_z_kara, 3))
