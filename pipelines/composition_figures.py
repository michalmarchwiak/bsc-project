"""Wykresy struktury funduszu do prezentacji: skład sub-funduszy (poziom 1), wagi sub-funduszy (poziom 2)
i ekspozycja look-through.

Wejście: ``artifacts/backtest/<run_id>/weights_tickers.parquet`` (kolumny: date, model, strategy, profile,
subfund, ticker, w_sf – waga tickera w SF, W_sf – waga SF w portfelu, x = W_sf·w_sf – waga look-through)
i ``config.json`` przebiegu (limity profili).

Wyjście (``figures/backtest/``):
  subfund_composition_<P>.png  – wagi tickerów w każdym SF w czasie (szereg główny, profil P);
  portfolio_W_profiles.png     – wagi SF w portfelu w czasie dla wybranych profili;
  portfolio_W_mean.png         – średni skład W w kalibracji i hold-out, P1–P5;
  lookthrough_max_x.png        – maxᵢ xᵢ w czasie wobec limitu c_LT profilu.

Uruchomienie:
    python -m pipelines.composition_figures                       # bieżący run_id
    python -m pipelines.composition_figures --run-id 3ababedef5d8 --profile P3
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pf import ROOT  # noqa: E402

log = logging.getLogger("composition_figures")
FIG_DIR = ROOT / "figures" / "backtest"
BACKTEST_DIR = ROOT / "artifacts" / "backtest"
SUBFUNDS = [f"SF{k}" for k in range(1, 11)]
SF_NAMES = {"SF1": "akcje PL", "SF2": "akcje USA", "SF3": "rynki rozw. ex-USA", "SF4": "rynki wschodzące",
            "SF5": "obligacje skarb. USA", "SF6": "obligacje IG", "SF7": "high yield",
            "SF8": "złoto i wydobycie", "SF9": "REIT", "SF10": "obligacje oszczędn."}
SF_COLORS = dict(zip(SUBFUNDS, plt.get_cmap("tab10").colors))
HOLDOUT_START = pd.Timestamp("2021-01-01")


def load(run_id: str, model: str, strategy: str) -> pd.DataFrame:
    path = BACKTEST_DIR / run_id / "weights_tickers.parquet"
    df = pd.read_parquet(path, filters=[("model", "==", model), ("strategy", "==", strategy)])
    for c in ("model", "strategy", "profile", "subfund", "ticker"):
        df[c] = df[c].astype(str)
    return df[df["subfund"].isin(SUBFUNDS)]


def _shade_holdout(ax):
    ax.axvspan(HOLDOUT_START, ax.get_xlim()[1], color="0.92", zorder=0)


def subfund_composition(df: pd.DataFrame, profile: str, top: int = 5) -> str:
    """Wagi tickerów w SF (poziom 1) w czasie; ``top`` tickerów o największej średniej wadze, reszta – „inne”."""
    d = df[df["profile"] == profile]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, sf in zip(axes.ravel(), SUBFUNDS):
        piv = d[d["subfund"] == sf].pivot_table(index="date", columns="ticker", values="w_sf",
                                                aggfunc="sum").fillna(0.0)
        if piv.empty:
            ax.set_axis_off()
            continue
        order = piv.mean().sort_values(ascending=False).index
        keep = list(order[:top])
        if len(order) > top:
            piv = piv[keep].assign(inne=piv[order[top:]].sum(axis=1))
        else:
            piv = piv[keep]
        labels = [c.replace(".WA", "").replace(".SP", "") for c in piv.columns]
        colors = list(plt.get_cmap("tab10").colors[: len(keep)]) + ["0.78"] * (piv.shape[1] - len(keep))
        ax.stackplot(piv.index, piv.T.to_numpy(), labels=labels, colors=colors, step="post", lw=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(piv.index.min(), piv.index.max())
        ax.axvline(HOLDOUT_START, color="k", ls="--", lw=1.0)
        ax.set_title(f"{sf}: {SF_NAMES[sf]}", fontsize=12, loc="left")
        ax.legend(fontsize=9.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1), frameon=False,
                  handlelength=1.0, columnspacing=0.8)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.tick_params(labelsize=10)
    return _save(fig, f"subfund_composition_{profile}.png")


def portfolio_W(df: pd.DataFrame, profiles: list[str]) -> str:
    """Wagi SF w portfelu (poziom 2) w czasie dla wybranych profili."""
    W = (df.drop_duplicates(["date", "profile", "subfund"])
           .pivot_table(index=["profile", "date"], columns="subfund", values="W_sf").fillna(0.0))
    fig, axes = plt.subplots(len(profiles), 1, figsize=(13, 1.8 * len(profiles) + 0.9), sharex=True,
                             constrained_layout=True)
    for ax, p in zip(np.atleast_1d(axes), profiles):
        piv = W.loc[p].reindex(columns=SUBFUNDS, fill_value=0.0)
        ax.stackplot(piv.index, piv.T.to_numpy(), labels=[f"{c} {SF_NAMES[c]}" for c in SUBFUNDS],
                     colors=[SF_COLORS[c] for c in SUBFUNDS], step="post", lw=0)
        ax.set_ylim(0, 1)
        ax.set_xlim(piv.index.min(), piv.index.max())
        ax.axvline(HOLDOUT_START, color="k", ls="--", lw=0.8)
        ax.set_ylabel(p, fontsize=13)
        ax.tick_params(labelsize=11)
    np.atleast_1d(axes)[0].legend(ncol=5, fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False)
    return _save(fig, "portfolio_W_profiles.png")


def portfolio_W_mean(df: pd.DataFrame, profiles: list[str]) -> str:
    """Średni skład W w kalibracji i hold-out, poziome słupki skumulowane."""
    d = df.drop_duplicates(["date", "profile", "subfund"]).copy()
    d["sample"] = np.where(d["date"] >= HOLDOUT_START, "hold-out", "kalibracja")
    n_dates = d.groupby(["profile", "sample"])["date"].nunique()
    sums = d.groupby(["profile", "sample", "subfund"])["W_sf"].sum().unstack("subfund").fillna(0.0)
    mean = sums.div(n_dates, axis=0).reindex(columns=SUBFUNDS, fill_value=0.0)
    rows = [(p, s) for p in profiles for s in ("kalibracja", "hold-out")]
    mean = mean.reindex(rows)
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    y = np.arange(len(rows))[::-1] + np.repeat(np.arange(len(profiles))[::-1] * 0.5, 2)
    left = np.zeros(len(rows))
    for sf in SUBFUNDS:
        v = mean[sf].to_numpy()
        ax.barh(y, v, left=left, color=SF_COLORS[sf], label=f"{sf} {SF_NAMES[sf]}", height=0.85)
        for yi, li, vi in zip(y, left, v):
            if vi >= 0.06:
                ax.text(li + vi / 2, yi, f"{100 * vi:.0f}", ha="center", va="center", fontsize=10, color="white")
        left += v
    ax.set_yticks(y, [f"{p} {s}" for p, s in rows], fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_xlabel("średnia waga sub-funduszu w portfelu", fontsize=11)
    ax.legend(ncol=4, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    return _save(fig, "portfolio_W_mean.png")


def lookthrough(df: pd.DataFrame, profiles: list[str], c_lt: dict[str, float]) -> str:
    """maxᵢ xᵢ (największa ekspozycja na pojedynczy instrument) wobec limitu c_LT."""
    mx = df.groupby(["profile", "date"])["x"].max()
    fig, axes = plt.subplots(1, len(profiles), figsize=(11, 2.9), constrained_layout=True)
    for ax, p in zip(np.atleast_1d(axes), profiles):
        s = mx.loc[p]
        ax.plot(s.index, s.to_numpy(), lw=0.9, color="C0", label=r"$\max_i x_i$")
        ax.axhline(c_lt[p], color="C3", ls="--", lw=1.0, label=r"$c_{LT}$")
        ax.axvline(HOLDOUT_START, color="k", ls=":", lw=0.8)
        share = (s >= c_lt[p] - 1e-4).mean()
        ax.set_title(f"{p}: $c_{{LT}}$ = {c_lt[p]:.2f}\nwiąże w {100 * share:.0f}% dat", fontsize=10)
        ax.set_ylim(0, 1.25 * c_lt[p])
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(6))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.tick_params(labelsize=9)
    np.atleast_1d(axes)[0].legend(fontsize=9, loc="lower left")
    return _save(fig, "lookthrough_max_x.png")


def _save(fig, name: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None, help="katalog artifacts/backtest/<run_id> (domyślnie bieżący)")
    ap.add_argument("--model", default="rvine")
    ap.add_argument("--strategy", default="ce_dep")
    ap.add_argument("--profile", default="P3", help="profil wykresu składu sub-funduszy")
    ap.add_argument("--profiles-w", default="P1,P3,P5", help="profile wykresu W w czasie")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_id = args.run_id
    if run_id is None:
        from pipelines.backtest import current_run_id
        run_id = current_run_id()
    cfg = json.loads((BACKTEST_DIR / run_id / "config.json").read_text(encoding="utf-8"))
    prof_cfg = cfg["profiles"]["profiles"] if "profiles" in cfg.get("profiles", {}) else cfg["profiles"]
    profiles = sorted(prof_cfg)
    c_lt = {p: float(prof_cfg[p]["c_lt"]) for p in profiles}

    df = load(run_id, args.model, args.strategy)
    files = [subfund_composition(df, args.profile),
             portfolio_W(df, args.profiles_w.split(",")),
             portfolio_W_mean(df, profiles),
             lookthrough(df, profiles, c_lt)]
    for f in files:
        log.info("zapisano %s", f)


if __name__ == "__main__":
    main()
