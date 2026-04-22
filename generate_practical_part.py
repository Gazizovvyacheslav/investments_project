#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


ASSETS = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "XLV", "TLT", "GLD"]
RF_TICKER = "^IRX"  # 13-week T-bill yield proxy
YEARS = 5
WINDOW_GRID = [126, 252, 504]
REBALANCE_GRID = ["ME", "QE"]  # month-end, quarter-end
RISK_CAP_GRID = [0.10, 0.12, 0.15, 0.18]  # annualized volatility caps
SHORT_BOUNDS = (-0.3, 0.5)
CALIBRATION_SPLIT = 0.6
TRADING_DAYS = 252

OUT_DIR = Path("results")
FIG_DIR = OUT_DIR / "figures"


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    config: dict


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            return df["Close"]
        if "Adj Close" in df.columns.get_level_values(0):
            return df["Adj Close"]
        raise ValueError("Cannot locate Close/Adj Close in downloaded multi-index columns.")
    return df


def download_data(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.Series]:
    raw_assets = yf.download(
        ASSETS,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    prices = _extract_close_prices(raw_assets)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=ASSETS[0])
    prices = prices.loc[:, [c for c in ASSETS if c in prices.columns]].copy()
    missing_assets = [a for a in ASSETS if a not in prices.columns]
    if missing_assets:
        raise ValueError(f"Missing assets in data download: {missing_assets}")

    raw_rf = yf.download(
        RF_TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    rf_df = _extract_close_prices(raw_rf)
    if isinstance(rf_df, pd.DataFrame):
        if RF_TICKER in rf_df.columns:
            rf = rf_df[RF_TICKER]
        else:
            rf = rf_df.iloc[:, 0]
    else:
        rf = rf_df
    rf.name = RF_TICKER

    return prices, rf


def clean_and_align(prices: pd.DataFrame, rf_annual_percent: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    prices = prices.sort_index().ffill().dropna(how="any")
    returns = prices.pct_change().dropna()

    rf_daily = (rf_annual_percent / 100.0) / TRADING_DAYS
    rf_daily = rf_daily.sort_index().reindex(returns.index).ffill().bfill()
    rf_daily.name = "rf_daily"

    return returns, rf_daily


def nearest_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    groups = pd.Series(index=index, data=index).groupby(pd.Grouper(freq=freq))
    dates = [grp.iloc[-1] for _, grp in groups if not grp.empty]
    return dates


def safe_weights(w: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    s = w.sum()
    if abs(s) < 1e-12:
        return np.repeat(1.0 / len(w), len(w))
    return w / s


def solve_max_return_under_risk(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_cap_daily: float,
    bounds: Iterable[tuple[float, float]],
) -> np.ndarray:
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: risk_cap_daily**2 - float(w @ cov @ w)},
    ]

    result = minimize(
        lambda w: -float(w @ mu),
        x0=x0,
        method="SLSQP",
        bounds=list(bounds),
        constraints=constraints,
        options={"maxiter": 600, "ftol": 1e-12},
    )
    if result.success:
        return safe_weights(result.x)

    fallback = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(bounds),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
    )
    if fallback.success:
        return safe_weights(fallback.x)

    return x0


def solve_min_vol_for_target_return(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    bounds: Iterable[tuple[float, float]],
) -> np.ndarray | None:
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w, m=mu, t=target_return: float(w @ m) - t},
    ]
    res = minimize(
        lambda w: float(w @ cov @ w),
        x0=x0,
        method="SLSQP",
        bounds=list(bounds),
        constraints=constraints,
        options={"maxiter": 600, "ftol": 1e-12},
    )
    if not res.success:
        return None
    return safe_weights(res.x)


def solve_tangency(mu: np.ndarray, cov: np.ndarray, rf_daily: float, bounds: Iterable[tuple[float, float]]) -> np.ndarray:
    n = len(mu)
    x0 = np.repeat(1.0 / n, n)

    def objective(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-16)))
        return -((ret - rf_daily) / vol)

    res = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=list(bounds),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 600, "ftol": 1e-12},
    )
    if res.success:
        return safe_weights(res.x)
    return x0


def calc_max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def calc_metrics(returns: pd.Series, rf_daily: pd.Series, market_returns: pd.Series | None = None) -> dict:
    returns = returns.dropna()
    rf = rf_daily.reindex(returns.index).ffill().bfill()
    n = len(returns)
    if n == 0:
        return {}

    cumulative = float((1.0 + returns).prod() - 1.0)
    annualized_return = float((1.0 + cumulative) ** (TRADING_DAYS / n) - 1.0)
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    excess = returns - rf
    sharpe = float((excess.mean() * TRADING_DAYS) / annualized_vol) if annualized_vol > 1e-12 else np.nan
    mdd = calc_max_drawdown(returns)

    q95 = float(np.quantile(returns, 0.05))
    q99 = float(np.quantile(returns, 0.01))
    var95 = -q95
    var99 = -q99
    es95 = float(-returns[returns <= q95].mean())
    es99 = float(-returns[returns <= q99].mean())

    beta = np.nan
    if market_returns is not None:
        aligned = pd.concat([returns, market_returns.reindex(returns.index)], axis=1).dropna()
        if len(aligned) > 3 and aligned.iloc[:, 1].var(ddof=1) > 1e-12:
            beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var(ddof=1))

    raroc = float(annualized_return / es95) if es95 > 1e-12 else np.nan

    return {
        "n_obs": n,
        "cumulative_return": cumulative,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "VaR_95": var95,
        "VaR_99": var99,
        "ES_95": es95,
        "ES_99": es99,
        "beta_to_spy": beta,
        "RAROC": raroc,
    }


def run_backtest(
    returns: pd.DataFrame,
    rf_daily: pd.Series,
    window: int,
    rebalance_freq: str,
    risk_cap_annual: float,
    bounds: Iterable[tuple[float, float]],
) -> BacktestResult:
    idx = returns.index
    rb_dates = nearest_rebalance_dates(idx, rebalance_freq)
    rb_dates = [d for d in rb_dates if d in idx]

    strat_ret = pd.Series(index=idx, dtype=float)
    weights_log: list[pd.Series] = []

    for i, d in enumerate(rb_dates):
        loc = idx.get_loc(d)
        if loc < window:
            continue

        train = returns.iloc[loc - window : loc]
        mu = train.mean().values
        cov = train.cov().values
        risk_cap_daily = risk_cap_annual / np.sqrt(TRADING_DAYS)
        w = solve_max_return_under_risk(mu, cov, risk_cap_daily, bounds=bounds)

        if i + 1 < len(rb_dates):
            next_d = rb_dates[i + 1]
            next_loc = idx.get_loc(next_d)
            hold = returns.iloc[loc + 1 : next_loc + 1]
        else:
            hold = returns.iloc[loc + 1 :]

        if not hold.empty:
            strat_ret.loc[hold.index] = hold.values @ w
            weights_log.append(pd.Series(w, index=returns.columns, name=d))

    weights_df = pd.DataFrame(weights_log)
    strat_ret = strat_ret.dropna()
    return BacktestResult(
        returns=strat_ret,
        weights=weights_df,
        config={
            "window": window,
            "rebalance_freq": rebalance_freq,
            "risk_cap_annual": risk_cap_annual,
        },
    )


def split_validation_test(returns: pd.Series, ratio: float = CALIBRATION_SPLIT) -> tuple[pd.Series, pd.Series]:
    n = len(returns)
    split_idx = int(n * ratio)
    return returns.iloc[:split_idx], returns.iloc[split_idx:]


def calibrate_config(
    returns: pd.DataFrame,
    rf_daily: pd.Series,
    market_returns: pd.Series,
    bounds: Iterable[tuple[float, float]],
    name: str,
) -> tuple[BacktestResult, pd.DataFrame]:
    grid_rows = []
    best: BacktestResult | None = None
    best_score = -np.inf

    for w in WINDOW_GRID:
        for f in REBALANCE_GRID:
            for c in RISK_CAP_GRID:
                bt = run_backtest(returns, rf_daily, window=w, rebalance_freq=f, risk_cap_annual=c, bounds=bounds)
                if len(bt.returns) < 80:
                    continue
                val_ret, _ = split_validation_test(bt.returns)
                if len(val_ret) < 40:
                    continue

                ew_val = returns.mean(axis=1).reindex(val_ret.index).dropna()
                spy_val = market_returns.reindex(val_ret.index).dropna()
                aligned_idx = val_ret.index.intersection(ew_val.index).intersection(spy_val.index)
                val_ret = val_ret.reindex(aligned_idx)
                ew_val = ew_val.reindex(aligned_idx)
                spy_val = spy_val.reindex(aligned_idx)

                m_val = calc_metrics(val_ret, rf_daily, market_returns)
                m_ew = calc_metrics(ew_val, rf_daily, market_returns)
                m_spy = calc_metrics(spy_val, rf_daily, market_returns)
                if not m_val:
                    continue

                es_ok = m_val["ES_95"] <= min(m_ew["ES_95"], m_spy["ES_95"]) * 1.10
                mdd_ok = abs(m_val["max_drawdown"]) <= min(abs(m_ew["max_drawdown"]), abs(m_spy["max_drawdown"])) * 1.15
                gate = es_ok and mdd_ok

                score = m_val["sharpe_ratio"] if np.isfinite(m_val["sharpe_ratio"]) else -999
                if not gate:
                    score -= 1.0

                grid_rows.append(
                    {
                        "strategy_type": name,
                        "window": w,
                        "rebalance_freq": f,
                        "risk_cap_annual": c,
                        "val_sharpe": m_val["sharpe_ratio"],
                        "val_ES95": m_val["ES_95"],
                        "val_MDD": m_val["max_drawdown"],
                        "gate_passed": gate,
                        "score": score,
                    }
                )

                if score > best_score:
                    best_score = score
                    best = bt

    if best is None:
        raise RuntimeError(f"Calibration failed for strategy {name}.")

    return best, pd.DataFrame(grid_rows).sort_values("score", ascending=False)


def build_frontier_and_cml(
    train_returns: pd.DataFrame,
    rf_daily_scalar: float,
    bounds: Iterable[tuple[float, float]],
) -> tuple[pd.DataFrame, pd.Series, float]:
    mu = train_returns.mean().values
    cov = train_returns.cov().values

    min_ret, max_ret = float(mu.min()), float(mu.max())
    targets = np.linspace(min_ret, max_ret, 45)
    points = []
    for t in targets:
        w = solve_min_vol_for_target_return(mu, cov, t, bounds)
        if w is None:
            continue
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        points.append({"target_ret_daily": t, "ret_daily": ret, "vol_daily": vol})
    frontier = pd.DataFrame(points)
    if frontier.empty:
        raise RuntimeError("Could not build efficient frontier.")

    w_tan = solve_tangency(mu, cov, rf_daily_scalar, bounds=bounds)
    ret_t = float(w_tan @ mu)
    vol_t = float(np.sqrt(max(w_tan @ cov @ w_tan, 1e-16)))
    slope = (ret_t - rf_daily_scalar) / vol_t

    tangent = pd.Series(w_tan, index=train_returns.columns, name="tangency_weights")
    return frontier, tangent, float(slope)


def make_plots(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    frontier: pd.DataFrame,
    cml_slope: float,
    rf_daily_scalar: float,
    tangency_point: tuple[float, float],
    weights: pd.DataFrame,
    equity_curves: pd.DataFrame,
    drawdowns: pd.DataFrame,
) -> None:
    # Normalized prices
    norm_prices = prices / prices.iloc[0]
    plt.figure(figsize=(12, 6))
    for c in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[c], label=c, linewidth=1.2)
    plt.title("Нормированные цены активов (t0=1)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_normalized_prices.png", dpi=150)
    plt.close()

    # Daily returns sample (rolling mean)
    plt.figure(figsize=(12, 5))
    rolling = returns.mean(axis=1).rolling(20).mean()
    plt.plot(rolling.index, rolling, label="Средняя доходность пула (20d MA)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Динамика доходностей (сглаженная)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_returns_trend.png", dpi=150)
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    ax.set_title("Корреляционная матрица доходностей")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_corr_matrix.png", dpi=150)
    plt.close()

    # Frontier + CML
    plt.figure(figsize=(10, 6))
    f = frontier.copy()
    f["vol_ann"] = f["vol_daily"] * np.sqrt(TRADING_DAYS)
    f["ret_ann"] = f["ret_daily"] * TRADING_DAYS
    plt.plot(f["vol_ann"], f["ret_ann"], label="Efficient frontier", color="tab:blue")
    vol_t_ann = tangency_point[0] * np.sqrt(TRADING_DAYS)
    ret_t_ann = tangency_point[1] * TRADING_DAYS
    plt.scatter([vol_t_ann], [ret_t_ann], color="tab:red", label="Тангенциальный портфель")

    x = np.linspace(0, max(f["vol_ann"].max(), vol_t_ann) * 1.05, 100)
    rf_ann = rf_daily_scalar * TRADING_DAYS
    cml_ann_slope = cml_slope * np.sqrt(TRADING_DAYS)
    y = rf_ann + cml_ann_slope * x
    plt.plot(x, y, "--", color="tab:green", label="CML")
    plt.xlabel("Волатильность (годовая)")
    plt.ylabel("Доходность (годовая)")
    plt.title("Граница эффективных портфелей и CML")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_frontier_cml.png", dpi=150)
    plt.close()

    # Weights dynamics
    if not weights.empty:
        plt.figure(figsize=(12, 6))
        weights.plot.area(ax=plt.gca(), linewidth=0)
        plt.title("Динамика весов оптимального no-short портфеля")
        plt.ylabel("Вес")
        plt.xlabel("Дата ребалансировки")
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "05_weights_dynamics.png", dpi=150)
        plt.close()

    # Equity curves
    plt.figure(figsize=(12, 6))
    for c in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[c], label=c, linewidth=1.4)
    plt.title("Накопленная доходность: стратегия и бенчмарки")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_equity_curves.png", dpi=150)
    plt.close()

    # Drawdown curves
    plt.figure(figsize=(12, 6))
    for c in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[c], label=c, linewidth=1.3)
    plt.title("Просадки (drawdown)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_drawdowns.png", dpi=150)
    plt.close()


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def num(x: float) -> str:
    return f"{x:.4f}"


def render_report(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf_daily: pd.Series,
    best_no_short: BacktestResult,
    best_short: BacktestResult,
    metrics_test: pd.DataFrame,
    calibration_table: pd.DataFrame,
    frontier: pd.DataFrame,
    tangency: pd.Series,
    cml_slope: float,
    out_file: Path,
) -> None:
    test_start = metrics_test.attrs["test_start"]
    test_end = metrics_test.attrs["test_end"]
    start_date = prices.index.min().date()
    end_date = prices.index.max().date()

    ns_cfg = best_no_short.config
    s_cfg = best_short.config

    top_cal = calibration_table.head(6).copy()
    top_cal["val_sharpe"] = top_cal["val_sharpe"].map(lambda x: f"{x:.3f}")
    top_cal["val_ES95"] = top_cal["val_ES95"].map(lambda x: f"{x:.4f}")
    top_cal["val_MDD"] = top_cal["val_MDD"].map(lambda x: f"{x:.4f}")

    best_weights = best_no_short.weights.mean().sort_values(ascending=False).head(5)
    best_weights_text = ", ".join([f"{k}: {v:.2%}" for k, v in best_weights.items()])

    # Tangency summary
    mu = returns.mean().values
    cov = returns.cov().values
    w_t = tangency.values
    ret_t = float(w_t @ mu)
    vol_t = float(np.sqrt(w_t @ cov @ w_t))

    def mrow(name: str) -> str:
        r = metrics_test.loc[name]
        return (
            f"| {name} | {pct(r['cumulative_return'])} | {pct(r['annualized_return'])} | {pct(r['annualized_volatility'])} "
            f"| {r['sharpe_ratio']:.3f} | {pct(r['max_drawdown'])} | {pct(r['VaR_95'])} | {pct(r['ES_95'])} | {r['beta_to_spy']:.3f} | {r['RAROC']:.3f} |"
        )


def main() -> None:
    ensure_dirs()

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=YEARS)

    prices, rf_raw = download_data(start, end)
    returns, rf_daily = clean_and_align(prices, rf_raw)
    market_returns = returns["SPY"]

    # Calibrate no-short
    no_short_bounds = [(0.0, 1.0)] * len(ASSETS)
    best_ns, grid_ns = calibrate_config(
        returns=returns,
        rf_daily=rf_daily,
        market_returns=market_returns,
        bounds=no_short_bounds,
        name="no_short",
    )

    # Calibrate short-allowed
    short_bounds = [SHORT_BOUNDS] * len(ASSETS)
    best_short, grid_short = calibrate_config(
        returns=returns,
        rf_daily=rf_daily,
        market_returns=market_returns,
        bounds=short_bounds,
        name="short_allowed",
    )

    # Build test set metrics
    _, ns_test = split_validation_test(best_ns.returns)
    _, short_test = split_validation_test(best_short.returns)

    common_idx = ns_test.index.intersection(short_test.index)
    ns_test = ns_test.reindex(common_idx)
    short_test = short_test.reindex(common_idx)
    ew_test = returns.mean(axis=1).reindex(common_idx)
    spy_test = returns["SPY"].reindex(common_idx)

    metrics = {
        "Opt No-Short": calc_metrics(ns_test, rf_daily, market_returns=returns["SPY"]),
        "Opt Short": calc_metrics(short_test, rf_daily, market_returns=returns["SPY"]),
        "Equal Weight": calc_metrics(ew_test, rf_daily, market_returns=returns["SPY"]),
        "SPY": calc_metrics(spy_test, rf_daily, market_returns=returns["SPY"]),
    }
    metrics_test = pd.DataFrame(metrics).T
    metrics_test.attrs["test_start"] = common_idx.min()
    metrics_test.attrs["test_end"] = common_idx.max()

    # Efficient frontier and CML from latest training window
    train_window = max(best_ns.config["window"], 252)
    train_slice = returns.iloc[-train_window:]
    rf_scalar = float(rf_daily.reindex(train_slice.index).mean())
    frontier, tangency, cml_slope = build_frontier_and_cml(train_slice, rf_scalar, no_short_bounds)

    # Build curves for plots
    equity = pd.DataFrame(
        {
            "Opt No-Short": (1.0 + ns_test).cumprod(),
            "Opt Short": (1.0 + short_test).cumprod(),
            "Equal Weight": (1.0 + ew_test).cumprod(),
            "SPY": (1.0 + spy_test).cumprod(),
        }
    )
    drawdowns = equity.divide(equity.cummax()).subtract(1.0)

    # Tangency point for plotting
    mu = train_slice.mean().values
    cov = train_slice.cov().values
    wt = tangency.values
    tangency_point = (float(np.sqrt(wt @ cov @ wt)), float(wt @ mu))

    make_plots(
        prices=prices,
        returns=returns,
        corr=returns.corr(),
        frontier=frontier,
        cml_slope=cml_slope,
        rf_daily_scalar=rf_scalar,
        tangency_point=tangency_point,
        weights=best_ns.weights,
        equity_curves=equity,
        drawdowns=drawdowns,
    )

    calibration_table = pd.concat([grid_ns, grid_short], ignore_index=True)
    calibration_table.to_csv(OUT_DIR / "calibration_grid.csv", index=False)
    metrics_test.to_csv(OUT_DIR / "metrics_test.csv")
    frontier.to_csv(OUT_DIR / "efficient_frontier.csv", index=False)
    best_ns.weights.to_csv(OUT_DIR / "weights_no_short.csv")
    best_short.weights.to_csv(OUT_DIR / "weights_short.csv")

    render_report(
        prices=prices,
        returns=returns,
        rf_daily=rf_daily,
        best_no_short=best_ns,
        best_short=best_short,
        metrics_test=metrics_test,
        calibration_table=calibration_table,
        frontier=frontier,
        tangency=tangency,
        cml_slope=cml_slope,
        out_file=OUT_DIR / "practical_part.md",
    )

    print("Done. Outputs:")
    print(f"- {OUT_DIR / 'practical_part.md'}")
    print(f"- {OUT_DIR / 'metrics_test.csv'}")
    print(f"- {OUT_DIR / 'calibration_grid.csv'}")
    print(f"- {FIG_DIR}")


if __name__ == "__main__":
    main()
