"""Parametryczna powierzchnia zmienności implikowanej S&P 500 odtwarzana z indeksów Cboe.

1. Struktura terminowa. Indeksy VIX9D, VIX, VIX3M, VIX6M, VIX1Y (tenory 9, 30, 93, 186, 365 dni)
   wyznaczają w dniu t punkty wariancji całkowitej w_i = σ_i² T_i. Dla dowolnego T
   wariancja całkowita jest interpolowana liniowo względem T, a σ_VIX(T) = √(w(T)/T);
   poza zakresem tenorów σ jest ekstrapolowana płasko. Ciąg w_i jest wymuszany jako
   niemalejący (brak arbitrażu kalendarzowego), co gwarantuje monotoniczność w(T).

2. Korekta VIX vs σ_ATM. VIX jest bezmodelową miarą oczekiwanej wariancji z całego pasma
   cen wykonania (replikacja kontraktu wariancyjnego), więc przy ujemnej skośności
   przewyższa zmienność ATM. Zmienność ATM (k = 0) modelowana jest afinicznie:
   σ_ATM(T) = a + b·VIX_T, gdzie a, b wynikają z kalibracji poniżej (zwykle b < 1).

3. Skośność. Dla k = ln(K/F), F = S e^{(r−q)T}:
       σ_imp(k, T) = a + b·VIX_T + c·k + d·k·VIX_T  [+ e·k·SKEW/100],
   VIX_T – σ_VIX(T) w ujęciu dziesiętnym. Nachylenie ∂σ/∂k = c + d·VIX_T (+ e·SKEW/100)
   zależy od poziomu zmienności. Parametry szacowane są MNK lub odpornie (Huber, IRLS)
   na kwotowaniach opcji OTM (puty dla K < F, calle dla K ≥ F) z DoltHub i zrzutów yfinance.

4. Walidacja poza próbą (podział czasowy): RMSE błędu ceny modelowej względem środka
   bid/ask, w % premii, osobno dla putów ATM, 5% OTM i 10% OTM.

Ograniczenia: powierzchnia jest odtwarzana modelowo, a nie obserwowana; liniowość w k
nie oddaje wypukłości uśmiechu dla skrzydeł (|k| > 0,15); kwotowania i indeksy nie są
idealnie zsynchronizowane w czasie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from pf.options import bs, iv

TENOR_COLUMNS = ["VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]
DEFAULT_TENORS_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 186, "VIX1Y": 365}


# --------------------------------------------------------------------------- struktura terminowa

def interp_total_variance(tenors: np.ndarray, vols: np.ndarray, T: np.ndarray) -> np.ndarray:
    """σ(T) z interpolacji liniowej wariancji całkowitej; ``tenors`` w latach, ``vols`` dziesiętnie."""
    tenors, vols = np.asarray(tenors, dtype=float), np.asarray(vols, dtype=float)
    ok = np.isfinite(vols) & (vols > 0)
    tenors, vols = tenors[ok], vols[ok]
    if tenors.size == 0:
        return np.full(np.shape(T), np.nan)
    order = np.argsort(tenors)
    tenors, vols = tenors[order], vols[order]
    w = np.maximum.accumulate(vols ** 2 * tenors)
    T = np.asarray(T, dtype=float)
    Tc = np.clip(T, tenors[0], tenors[-1])
    sigma = np.sqrt(np.interp(Tc, tenors, w) / Tc)
    # Płaska ekstrapolacja σ: poniżej najkrótszego i powyżej najdłuższego tenoru.
    sigma = np.where(T < tenors[0], np.sqrt(w[0] / tenors[0]), sigma)
    return np.where(T > tenors[-1], np.sqrt(w[-1] / tenors[-1]), sigma)


def total_variance(tenors, vols, T) -> np.ndarray:
    return interp_total_variance(tenors, vols, T) ** 2 * np.asarray(T, dtype=float)


def vix_at_tenor(vol_indices: pd.DataFrame, dates: pd.Series | np.ndarray, T: np.ndarray,
                 tenors_days: dict[str, int] = DEFAULT_TENORS_DAYS) -> np.ndarray:
    """σ_VIX(T) (dziesiętnie) dla par (data, T); poziomy indeksów w punktach procentowych."""
    cols = [c for c in tenors_days if c in vol_indices.columns]
    tenors = np.array([tenors_days[c] for c in cols]) / 365.0
    dates = pd.DatetimeIndex(dates)
    levels = vol_indices[cols].reindex(dates.unique()) / 100.0
    out = np.full(len(dates), np.nan)
    T = np.broadcast_to(np.asarray(T, dtype=float), (len(dates),))
    codes, uniques = pd.factorize(dates)
    for i, d in enumerate(uniques):
        idx = codes == i
        out[idx] = interp_total_variance(tenors, levels.loc[d].to_numpy(), T[idx])
    return out


# --------------------------------------------------------------------------- model skośności

@dataclass
class SkewParams:
    a: float
    b: float
    c: float
    d: float
    e: float = 0.0
    use_skew: bool = False
    min_sigma: float = 0.01
    fit_info: dict = field(default_factory=dict)

    def sigma(self, k, vix_T, skew=None) -> np.ndarray:
        k, vix_T = np.asarray(k, dtype=float), np.asarray(vix_T, dtype=float)
        s = self.a + self.b * vix_T + self.c * k + self.d * k * vix_T
        if self.use_skew:
            s = s + self.e * k * np.asarray(skew, dtype=float) / 100.0
        return np.maximum(s, self.min_sigma)

    def atm(self, vix_T) -> np.ndarray:
        """Korekta VIX → σ_ATM: σ_ATM(T) = a + b·VIX_T."""
        return np.maximum(self.a + self.b * np.asarray(vix_T, dtype=float), self.min_sigma)

    def slope(self, vix_T, skew=None) -> np.ndarray:
        s = self.c + self.d * np.asarray(vix_T, dtype=float)
        return s + self.e * np.asarray(skew, dtype=float) / 100.0 if self.use_skew else s

    def to_dict(self) -> dict:
        return asdict(self)


def design_matrix(k, vix_T, skew=None) -> np.ndarray:
    k, vix_T = np.asarray(k, dtype=float), np.asarray(vix_T, dtype=float)
    cols = [np.ones_like(k), vix_T, k, k * vix_T]
    if skew is not None:
        cols.append(k * np.asarray(skew, dtype=float) / 100.0)
    return np.column_stack(cols)


def huber_irls(X: np.ndarray, y: np.ndarray, c: float = 1.345, max_iter: int = 50,
               tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Estymator M Hubera metodą IRLS; skala reszt z MAD. Zwraca (β, wagi końcowe)."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    w = np.ones_like(y)
    for _ in range(max_iter):
        resid = y - X @ beta
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if scale <= 0:
            break
        u = np.abs(resid) / scale
        w = c / np.maximum(u, c)
        sw = np.sqrt(w)
        new = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta, w


def fit_skew(obs: pd.DataFrame, use_skew: bool = False, robust: bool = True, huber_c: float = 1.345,
             min_sigma: float = 0.01) -> SkewParams:
    """Kalibracja σ_imp = a + b·VIX_T + c·k + d·k·VIX_T [+ e·k·SKEW/100]; kolumny: iv, k, vix_T[, SKEW]."""
    cols = ["iv", "k", "vix_T"] + (["SKEW"] if use_skew else [])
    if "reject" in obs.columns:
        obs = obs[obs["reject"] == ""]
    data = obs[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = design_matrix(data["k"], data["vix_T"], data["SKEW"] if use_skew else None)
    y = data["iv"].to_numpy()
    if robust:
        beta, _ = huber_irls(X, y, huber_c)
    else:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    info = {"n_obs": int(len(y)), "rmse_iv": float(np.sqrt(np.mean(resid ** 2))),
            "r2": float(1 - resid.var() / y.var()) if y.var() > 0 else None,
            "method": "huber_irls" if robust else "ols"}
    if "quote_date" in obs.columns:
        info["first_date"] = str(obs["quote_date"].min().date())
        info["last_date"] = str(obs["quote_date"].max().date())
    e = float(beta[4]) if use_skew else 0.0
    return SkewParams(*map(float, beta[:4]), e=e, use_skew=use_skew, min_sigma=min_sigma, fit_info=info)


# --------------------------------------------------------------------------- powierzchnia

@dataclass
class VolSurface:
    """Powierzchnia w dniu t: poziomy indeksów zmienności (p.p.) + parametry skośności."""

    levels: dict[str, float]
    params: SkewParams
    tenors_days: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TENORS_DAYS))

    def vix_T(self, T) -> np.ndarray:
        cols = [c for c in self.tenors_days if c in self.levels]
        tenors = np.array([self.tenors_days[c] for c in cols]) / 365.0
        vols = np.array([self.levels[c] for c in cols], dtype=float) / 100.0
        return interp_total_variance(tenors, vols, T)

    def implied_vol(self, K, T, F) -> np.ndarray:
        k = np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float))
        return self.params.sigma(k, self.vix_T(T), self.levels.get("SKEW"))

    def price(self, S, K, T, r, q, is_call) -> np.ndarray:
        sigma = self.implied_vol(K, T, bs.forward(S, T, r, q))
        return bs.price(S, K, T, r, q, sigma, is_call)


# --------------------------------------------------------------------------- dane kalibracyjne

def rate_from_irx(irx_pct: pd.Series | np.ndarray | float) -> np.ndarray:
    """Stopa ciągła z rentowności ^IRX (% p.a.): r = ln(1 + y)."""
    return np.log1p(np.asarray(irx_pct, dtype=float) / 100.0)


def implied_forwards(quotes: pd.DataFrame, n_strikes: int = 5) -> pd.Series:
    """Forward z parytetu put–call F = K + e^{rT}(C − P) – mediana dla ``n_strikes`` cen
    wykonania najbliższych cenie bazowej; indeks (quote_date, expiration)."""
    q = quotes[(quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])]
    q = q.assign(mid=0.5 * (q["bid"] + q["ask"]))
    wide = q.pivot_table(index=["quote_date", "expiration", "strike"], columns="call_put",
                         values="mid", aggfunc="mean")
    if not {"C", "P"} <= set(wide.columns):
        return pd.Series(dtype=float)
    wide = wide.dropna(subset=["C", "P"]).reset_index()
    meta = q.groupby(["quote_date", "expiration"])[["underlying_price", "r", "T"]].first()
    wide = wide.join(meta, on=["quote_date", "expiration"])
    wide["dist"] = (wide["strike"] - wide["underlying_price"]).abs()
    wide["F"] = wide["strike"] + np.exp(wide["r"] * wide["T"]) * (wide["C"] - wide["P"])
    near = wide.sort_values("dist").groupby(["quote_date", "expiration"]).head(n_strikes)
    return near.groupby(["quote_date", "expiration"])["F"].median()


def prepare_quotes(quotes: pd.DataFrame, vol_indices: pd.DataFrame, irx: pd.Series | None,
                   cfg: dict, filters: dict) -> pd.DataFrame:
    """Kwotowania → obserwacje kalibracyjne (iv, k, vix_T, SKEW) dla opcji OTM.

    ``quotes``: quote_date, expiration, strike, call_put, bid, ask, underlying_price
    [, openInterest, lastTradeDate]. ``irx`` – szereg ^IRX (% p.a.) indeksowany datą.
    """
    df = quotes.copy()
    df["T"] = (df["expiration"] - df["quote_date"]).dt.days / 365.0
    if irx is not None and len(irx):
        r = irx.reindex(irx.index.union(df["quote_date"].unique())).ffill().reindex(df["quote_date"])
        df["r"] = np.nan_to_num(rate_from_irx(r.to_numpy()), nan=cfg["default_rate"])
    else:
        df["r"] = cfg["default_rate"]
    fwd = implied_forwards(df)
    df = df.join(fwd.rename("F"), on=["quote_date", "expiration"])
    fallback = pd.Series(bs.forward(df["underlying_price"], df["T"], df["r"], cfg["default_div_yield"]),
                         index=df.index)
    df["F"] = df["F"].where(df["F"] > 0, fallback)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["q"] = df["r"] - np.log(df["F"] / df["underlying_price"]) / df["T"]
    df["k"] = np.log(df["strike"] / df["F"])
    otm = ((df["call_put"] == "P") & (df["k"] < 0)) | ((df["call_put"] == "C") & (df["k"] >= 0))
    df = df[otm & (df["T"] > 0)]
    df = iv.implied_vols_from_quotes(df, filters, df["r"].to_numpy(), df["q"].to_numpy())
    df["vix_T"] = vix_at_tenor(vol_indices, df["quote_date"], df["T"].to_numpy(), cfg["tenors_days"])
    if "SKEW" in vol_indices.columns:
        df["SKEW"] = vol_indices["SKEW"].reindex(df["quote_date"]).to_numpy()
    return df


# --------------------------------------------------------------------------- walidacja OOS

def oos_report(obs: pd.DataFrame, params: SkewParams, buckets: dict[str, list[float]]) -> pd.DataFrame:
    """RMSE błędu ceny (model − mid)/mid w % premii dla putów w przedziałach K/S.

    Zwraca tabelę: bucket, tenor (≤ 60 dni, > 60 dni, wszystkie), n, rmse_price_pct,
    bias_price_pct, rmse_iv_pp (punkty procentowe zmienności).
    """
    d = obs[(obs["reject"] == "") & (obs["call_put"] == "P")].dropna(subset=["iv", "vix_T"]).copy()
    d["iv_model"] = params.sigma(d["k"], d["vix_T"], d["SKEW"] if params.use_skew else None)
    d["price_model"] = bs.price(d["underlying_price"], d["strike"], d["T"], d["r"], d["q"],
                                d["iv_model"], False)
    d["err_pct"] = 100.0 * (d["price_model"] - d["mid"]) / d["mid"]
    d["err_iv_pp"] = 100.0 * (d["iv_model"] - d["iv"])
    d["moneyness"] = d["strike"] / d["underlying_price"]
    d["tenor"] = np.where(d["T"] * 365 <= 60, "<=60d", ">60d")
    rows = []
    for name, (lo, hi) in buckets.items():
        b = d[(d["moneyness"] >= lo) & (d["moneyness"] < hi)]
        for tenor, g in [("all", b), *b.groupby("tenor")]:
            rows.append({"bucket": name, "tenor": tenor, "n": len(g),
                         "rmse_price_pct": float(np.sqrt(np.mean(g["err_pct"] ** 2))) if len(g) else np.nan,
                         "bias_price_pct": float(g["err_pct"].mean()) if len(g) else np.nan,
                         "rmse_iv_pp": float(np.sqrt(np.mean(g["err_iv_pp"] ** 2))) if len(g) else np.nan})
    return pd.DataFrame(rows)


def time_split(obs: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = pd.Timestamp(split)
    return obs[obs["quote_date"] < cut], obs[obs["quote_date"] >= cut]
