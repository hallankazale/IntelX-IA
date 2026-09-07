from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

POSITION_FRACTION = 0.25
FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0002
ROUND_TRIP_COST_PCT = 2.0 * (FEE_RATE + SLIPPAGE_RATE) * 100.0
COST_SAFETY_MARGIN_PCT = 0.03
MIN_ENTRY_EDGE_PCT = ROUND_TRIP_COST_PCT + COST_SAFETY_MARGIN_PCT
STOP_LOSS = 0.03
TAKE_PROFIT = 0.06
MAX_HOLD_BARS = 48
HORIZONS = (3, 6, 12)
MAX_HORIZON = max(HORIZONS)
WARMUP = 80
OUTER_GAP_BARS = 14 * 24 * 12
INNER_GAP_BARS = 3 * 24 * 12
MIN_TUNING_TRADES = 5
MIN_VALIDATION_TRADES = 5
ENTRY_QUANTILES = (0.85, 0.90, 0.925, 0.95, 0.975, 0.99)
EXIT_QUANTILES = (0.20, 0.30, 0.40, 0.50)
FEATURE_NAMES = (
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ema9_21", "ema21_50", "ema9_slope3", "ema21_slope6", "ema50_slope12",
    "dist_ema21", "dist_ema50", "rsi14", "atr_pct", "vol_ratio20",
    "rv_12", "rv_48", "range_pct", "hour_sin", "hour_cos",
)


@dataclass(frozen=True)
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FoldSpec:
    name: str
    train_start: int
    train_end: int
    val_start: int
    val_end: int


@dataclass(frozen=True)
class Protocol:
    dev_start: int
    dev_end: int
    holdout_start: int
    holdout_end: int
    gap_bars: int
    folds: Tuple[FoldSpec, ...]


class MarketData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self.data_dir / "btcusdt_5m_180d.csv"

    def load_or_download(self, days: int = 180, max_age_hours: int = 6) -> List[Candle]:
        if self.cache.exists() and time.time() - self.cache.stat().st_mtime < max_age_hours * 3600:
            rows = self._load_csv()
            if len(rows) > 10_000:
                return rows
        rows = self._download(days)
        self._save_csv(rows)
        return rows

    def _load_csv(self) -> List[Candle]:
        out: List[Candle] = []
        with self.cache.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(Candle(
                    int(r["open_time"]), float(r["open"]), float(r["high"]),
                    float(r["low"]), float(r["close"]), float(r["volume"])
                ))
        return out

    def _save_csv(self, rows: List[Candle]) -> None:
        tmp = self.cache.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["open_time", "open", "high", "low", "close", "volume"])
            for c in rows:
                w.writerow([c.open_time, c.open, c.high, c.low, c.close, c.volume])
        tmp.replace(self.cache)

    def _download(self, days: int) -> List[Candle]:
        interval_ms = 5 * 60 * 1000
        end = int(time.time() * 1000)
        start = end - days * 24 * 60 * 60 * 1000
        urls = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
            "https://api1.binance.com/api/v3/klines",
        ]
        out: List[Candle] = []
        cursor = start
        while cursor < end:
            params = {"symbol": "BTCUSDT", "interval": "5m", "startTime": cursor, "endTime": end, "limit": 1000}
            data = None
            last_err: Exception | None = None
            for url in urls:
                try:
                    r = requests.get(url, params=params, timeout=20)
                    r.raise_for_status()
                    candidate = r.json()
                    if isinstance(candidate, list) and candidate:
                        data = candidate
                        break
                except Exception as e:
                    last_err = e
            if not data:
                if out:
                    break
                raise RuntimeError(f"Falha ao baixar candles públicos: {last_err}")
            for x in data:
                out.append(Candle(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])))
            nxt = int(data[-1][0]) + interval_ms
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.04)
        dedup = {c.open_time: c for c in out}
        rows = [dedup[k] for k in sorted(dedup)]
        if len(rows) < 20_000:
            raise RuntimeError(f"Poucos candles baixados: {len(rows)}")
        return rows


def ema(v: np.ndarray, p: int) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    out[0] = v[0]
    a = 2.0 / (p + 1)
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1.0 - a) * out[i - 1]
    return out


def sma(v: np.ndarray, p: int) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    s = 0.0
    for i, x in enumerate(v):
        s += x
        if i >= p:
            s -= v[i - p]
        out[i] = s / max(1, min(p, i + 1))
    return out


def rolling_std(v: np.ndarray, p: int) -> np.ndarray:
    out = np.zeros(len(v), dtype=float)
    for i in range(len(v)):
        j = max(0, i - p + 1)
        x = v[j:i + 1]
        out[i] = float(np.std(x)) if len(x) > 1 else 0.0
    return out


def rsi(close: np.ndarray, p: int = 14) -> np.ndarray:
    out = np.full(len(close), 50.0)
    g = l = 0.0
    for i in range(1, len(close)):
        ch = close[i] - close[i - 1]
        gain, loss = max(0.0, ch), max(0.0, -ch)
        if i <= p:
            g += gain; l += loss
            if i == p:
                g /= p; l /= p
        else:
            g = ((g * (p - 1)) + gain) / p
            l = ((l * (p - 1)) + loss) / p
        if i >= p:
            out[i] = 100.0 if l < 1e-12 else 100 - 100 / (1 + g / l)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, p: int = 14) -> np.ndarray:
    tr = np.empty(len(close), dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    out = np.empty(len(close), dtype=float)
    avg = tr[0]; out[0] = avg
    for i in range(1, len(tr)):
        avg = ((avg * i) + tr[i]) / (i + 1) if i < p else ((avg * (p - 1)) + tr[i]) / p
        out[i] = avg
    return out


def safe_log_return(close: np.ndarray, lag: int) -> np.ndarray:
    out = np.zeros(len(close), dtype=float)
    if lag < len(close):
        out[lag:] = np.log(np.maximum(close[lag:], 1e-12) / np.maximum(close[:-lag], 1e-12))
    return out


class FeatureDataset:
    def __init__(self, candles: List[Candle]):
        self.candles = candles
        close = np.array([c.close for c in candles], dtype=float)
        high = np.array([c.high for c in candles], dtype=float)
        low = np.array([c.low for c in candles], dtype=float)
        vol = np.array([c.volume for c in candles], dtype=float)
        e9, e21, e50 = ema(close, 9), ema(close, 21), ema(close, 50)
        rr, aa, v20 = rsi(close, 14), atr(high, low, close, 14), sma(vol, 20)
        ret1 = safe_log_return(close, 1)

        def slope(arr: np.ndarray, lag: int) -> np.ndarray:
            out = np.zeros(len(arr), dtype=float)
            out[lag:] = np.log(np.maximum(arr[lag:], 1e-12) / np.maximum(arr[:-lag], 1e-12))
            return out

        hours = np.array([(c.open_time // 3_600_000) % 24 for c in candles], dtype=float)
        angle = hours / 24.0 * 2.0 * math.pi
        X = np.column_stack([
            ret1, safe_log_return(close, 3), safe_log_return(close, 6), safe_log_return(close, 12),
            np.log(np.maximum(e9, 1e-12) / np.maximum(e21, 1e-12)),
            np.log(np.maximum(e21, 1e-12) / np.maximum(e50, 1e-12)),
            slope(e9, 3), slope(e21, 6), slope(e50, 12),
            np.log(np.maximum(close, 1e-12) / np.maximum(e21, 1e-12)),
            np.log(np.maximum(close, 1e-12) / np.maximum(e50, 1e-12)),
            (rr - 50.0) / 50.0, aa / np.maximum(close, 1e-12),
            vol / np.maximum(v20, 1e-12) - 1.0,
            rolling_std(ret1, 12), rolling_std(ret1, 48),
            (high - low) / np.maximum(close, 1e-12), np.sin(angle), np.cos(angle),
        ])
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.close = close
        self.targets: Dict[int, np.ndarray] = {}
        for h in HORIZONS:
            y = np.full(len(close), np.nan, dtype=float)
            y[:-h] = np.log(np.maximum(close[h:], 1e-12) / np.maximum(close[:-h], 1e-12))
            self.targets[h] = y

    @property
    def feature_count(self) -> int:
        return self.X.shape[1]


def build_protocol(n: int) -> Protocol:
    if n < 20_000:
        raise ValueError("Histórico insuficiente para protocolo v0.7.1")
    holdout_len = max(2_500, int(n * 0.15))
    holdout_start = n - holdout_len
    gap = min(OUTER_GAP_BARS, max(1_000, int(n * 0.08)))
    dev_start = WARMUP
    dev_end = holdout_start - gap
    if dev_end - dev_start < 10_000:
        raise ValueError("Faixa de desenvolvimento insuficiente")
    span = dev_end - dev_start
    val_len = max(1_000, int(span * 0.10))
    folds: List[FoldSpec] = []
    for idx, ratio in enumerate((0.42, 0.58, 0.74), start=1):
        train_end = dev_start + int(span * ratio)
        val_start = train_end + gap
        val_end = min(val_start + val_len, dev_end)
        if val_end - val_start < 700:
            raise ValueError("Fold muito curto após embargo temporal")
        folds.append(FoldSpec(f"Fold {idx}", dev_start, train_end, val_start, val_end))
    return Protocol(dev_start, dev_end, holdout_start, n, gap, tuple(folds))


def _new_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=140,
        max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=2.5,
        max_bins=127, early_stopping=False, random_state=1337,
    )


def fit_models(ds: FeatureDataset, indices: np.ndarray) -> Dict[int, HistGradientBoostingRegressor]:
    if len(indices) < 1_000:
        raise ValueError("Poucas amostras para treinar modelo")
    models: Dict[int, HistGradientBoostingRegressor] = {}
    for h in HORIZONS:
        valid = indices[np.isfinite(ds.targets[h][indices])]
        m = _new_model(); m.fit(ds.X[valid], ds.targets[h][valid]); models[h] = m
    return models


def predict_score(models: Dict[int, HistGradientBoostingRegressor], X: np.ndarray) -> np.ndarray:
    preds = []
    for h in HORIZONS:
        p = np.asarray(models[h].predict(X), dtype=float)
        preds.append((p / h) * 6.0)
    return np.median(np.vstack(preds), axis=0)


class Portfolio:
    def __init__(self, cash: float = 1000.0):
        self.cash = cash; self.asset = 0.0; self.entry = 0.0; self.cost_basis = 0.0
        self.bars_held = 0; self.trades = 0; self.wins = 0
        self.gross_profit = 0.0; self.gross_loss = 0.0

    def in_position(self) -> bool: return self.asset > 1e-12
    def equity(self, px: float) -> float: return self.cash + self.asset * px

    def buy(self, px: float) -> bool:
        if self.in_position(): return False
        alloc = self.cash * POSITION_FRACTION
        if alloc < 1e-9: return False
        exe = px * (1 + SLIPPAGE_RATE); fee = alloc * FEE_RATE
        self.asset = (alloc - fee) / exe; self.cash -= alloc
        self.entry = exe; self.cost_basis = alloc; self.bars_held = 0
        return True

    def sell(self, px: float) -> bool:
        if not self.in_position(): return False
        exe = px * (1 - SLIPPAGE_RATE); gross = self.asset * exe
        fee = gross * FEE_RATE; net = gross - fee; pnl = net - self.cost_basis
        self.cash += net; self.asset = 0.0; self.entry = 0.0; self.cost_basis = 0.0; self.bars_held = 0
        self.trades += 1
        if pnl >= 0: self.wins += 1; self.gross_profit += pnl
        else: self.gross_loss += -pnl
        return True

    def apply_risk(self, next_candle: Candle) -> bool:
        if not self.in_position(): return False
        self.bars_held += 1
        stop, take = self.entry * (1 - STOP_LOSS), self.entry * (1 + TAKE_PROFIT)
        if next_candle.low <= stop: return self.sell(stop)
        if next_candle.high >= take: return self.sell(take)
        if self.bars_held >= MAX_HOLD_BARS: return self.sell(next_candle.close)
        return False


def _daily_returns(curve: List[float], bars_per_day: int = 288) -> List[float]:
    if len(curve) < bars_per_day + 1: return []
    pts = [curve[0]] + [curve[i] for i in range(bars_per_day, len(curve), bars_per_day)]
    if pts[-1] != curve[-1]: pts.append(curve[-1])
    return [pts[i] / pts[i - 1] - 1.0 for i in range(1, len(pts)) if pts[i - 1] > 0]


def _sharpe_daily(xs: List[float]) -> float:
    if len(xs) < 3: return 0.0
    a = np.asarray(xs); sd = float(np.std(a, ddof=1))
    return 0.0 if sd < 1e-12 else float(np.mean(a) / sd * math.sqrt(365))


def _sortino_daily(xs: List[float]) -> float:
    if len(xs) < 3: return 0.0
    a = np.asarray(xs); d = a[a < 0]
    if len(d) < 2: return 0.0
    sd = float(np.std(d, ddof=1))
    return 0.0 if sd < 1e-12 else float(np.mean(a) / sd * math.sqrt(365))


def _cvar_95_daily(xs: List[float]) -> float:
    if not xs: return 0.0
    a = np.sort(np.asarray(xs)); k = max(1, int(math.ceil(len(a) * 0.05)))
    return float(np.mean(a[:k]) * 100.0)


def fair_benchmark(candles: List[Candle], start: int, end_exclusive: int) -> float:
    first, last = candles[start].close, candles[end_exclusive - 1].close
    p = Portfolio(1000.0); p.buy(first); p.sell(last)
    return (p.equity(last) / 1000.0 - 1.0) * 100.0


def backtest(ds: FeatureDataset, indices: np.ndarray, scores: np.ndarray, entry_threshold_pct: float, exit_threshold_pct: float) -> Dict[str, Any]:
    if len(indices) != len(scores): raise ValueError("indices e scores incompatíveis")
    if len(indices) < 10: raise ValueError("Janela de backtest muito curta")
    p = Portfolio(1000.0); peak = 1000.0; max_dd = 0.0; curve = [1000.0]; exposure_bars = 0; entry_signals = 0
    entry, exit_ = entry_threshold_pct / 100.0, exit_threshold_pct / 100.0
    for pos, i in enumerate(indices):
        if i + 1 >= len(ds.candles): break
        cur, nxt, score = ds.candles[i], ds.candles[i + 1], float(scores[pos])
        if not p.in_position():
            if score >= entry:
                entry_signals += 1; p.buy(cur.close)
        else:
            exposure_bars += 1
            if score <= exit_: p.sell(cur.close)
            else: p.apply_risk(nxt)
        eq = p.equity(nxt.close); peak = max(peak, eq)
        max_dd = max(max_dd, 0.0 if peak <= 0 else (peak - eq) / peak); curve.append(eq)
    final_i = int(indices[-1]); final_px = ds.candles[final_i].close
    p.sell(final_px); final_eq = p.equity(final_px); curve[-1] = final_eq
    ret = (final_eq / 1000.0 - 1.0) * 100.0
    benchmark = fair_benchmark(ds.candles, int(indices[0]), int(indices[-1]) + 1)
    pf = (99.0 if p.gross_profit > 0 else 0.0) if p.gross_loss < 1e-12 else p.gross_profit / p.gross_loss
    wr = 0.0 if p.trades == 0 else p.wins * 100.0 / p.trades
    daily = _daily_returns(curve)
    return {
        "return_pct": ret, "benchmark_pct": benchmark, "alpha_pct": ret - benchmark,
        "max_drawdown_pct": max_dd * 100.0, "profit_factor": pf, "win_rate_pct": wr,
        "sharpe_daily": _sharpe_daily(daily), "sortino_daily": _sortino_daily(daily),
        "cvar95_daily_pct": _cvar_95_daily(daily), "trades": p.trades,
        "entry_signals": entry_signals,
        "exposure_pct": exposure_bars * 100.0 / max(1, len(indices)),
        "entry_threshold_pct": entry_threshold_pct, "exit_threshold_pct": exit_threshold_pct,
        "final_equity": final_eq,
    }


def score_distribution(scores: np.ndarray) -> Dict[str, float]:
    pct = np.asarray(scores, dtype=float) * 100.0
    return {
        "mean_pct": float(np.mean(pct)), "max_pct": float(np.max(pct)), "min_pct": float(np.min(pct)),
        "p85_pct": float(np.quantile(pct, 0.85)), "p90_pct": float(np.quantile(pct, 0.90)),
        "p95_pct": float(np.quantile(pct, 0.95)), "p975_pct": float(np.quantile(pct, 0.975)),
        "p99_pct": float(np.quantile(pct, 0.99)),
    }


def prediction_diagnostics(ds: FeatureDataset, indices: np.ndarray, score: np.ndarray, entry_threshold_pct: float) -> Dict[str, float]:
    actual = ds.targets[6][indices]; valid = np.isfinite(actual)
    if not np.any(valid):
        return {"direction_accuracy_pct": 0.0, "signal_direction_accuracy_pct": 0.0, "mae_bps": 0.0}
    a, s = actual[valid], score[valid]
    global_acc = float(np.mean(np.sign(s) == np.sign(a)) * 100.0)
    high = s >= entry_threshold_pct / 100.0
    high_acc = float(np.mean(a[high] > 0) * 100.0) if np.any(high) else 0.0
    return {"direction_accuracy_pct": global_acc, "signal_direction_accuracy_pct": high_acc, "mae_bps": float(np.mean(np.abs(s - a)) * 10_000.0)}


def select_thresholds(ds: FeatureDataset, indices: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    dist = score_distribution(scores)
    entries: List[Tuple[float, float]] = []
    for q in ENTRY_QUANTILES:
        raw = float(np.quantile(scores, q) * 100.0)
        entries.append((q, max(MIN_ENTRY_EDGE_PCT, raw)))
    dedup: Dict[float, float] = {}
    for q, e in entries: dedup[round(e, 6)] = q
    entries = [(q, e) for e, q in [(k, v) for k, v in dedup.items()]]

    exits = sorted(set(min(0.0, float(np.quantile(scores, q) * 100.0)) for q in EXIT_QUANTILES))
    if not exits: exits = [0.0]

    best: Dict[str, Any] | None = None; best_score = -1e18
    tested = 0
    for q, entry in entries:
        for exit_ in exits:
            tested += 1
            m = backtest(ds, indices, scores, entry, exit_)
            if m["trades"] < MIN_TUNING_TRADES: continue
            if m["trades"] > max(80, len(indices) // 12): continue
            if m["exposure_pct"] > 65.0: continue
            score = (
                m["return_pct"] + 0.7 * m["alpha_pct"] - 0.8 * m["max_drawdown_pct"]
                + 0.8 * min(2.0, m["profit_factor"]) - 0.008 * max(0, m["trades"] - 30)
            )
            if score > best_score:
                best_score = score
                best = {"entry_quantile": q, "entry_pct": entry, "exit_pct": exit_, "tuning": m}

    if best is None:
        entry = MIN_ENTRY_EDGE_PCT
        exit_ = min(0.0, float(np.quantile(scores, 0.35) * 100.0))
        probe = backtest(ds, indices, scores, entry, exit_)
        return {
            "signal_sufficient": False, "reason": "Nenhuma calibração cost-aware atingiu o mínimo de operações.",
            "entry_quantile": None, "entry_pct": entry, "exit_pct": exit_, "tuning": probe,
            "distribution": dist, "tested_configs": tested,
            "cost_floor_pct": MIN_ENTRY_EDGE_PCT, "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
        }

    return {
        "signal_sufficient": True, "reason": "Calibração válida encontrada somente na janela de tuning.",
        **best, "distribution": dist, "tested_configs": tested,
        "cost_floor_pct": MIN_ENTRY_EDGE_PCT, "round_trip_cost_pct": ROUND_TRIP_COST_PCT,
    }


def run_fold(ds: FeatureDataset, fold: FoldSpec) -> Dict[str, Any]:
    train_len = fold.train_end - fold.train_start
    inner_cut = fold.train_start + int(train_len * 0.78)
    inner_train_end = max(fold.train_start + 1_500, inner_cut)
    tune_start = inner_train_end + INNER_GAP_BARS
    if fold.train_end - tune_start < 700:
        tune_start = inner_train_end + MAX_HORIZON
    inner_train_idx = np.arange(fold.train_start, inner_train_end, dtype=int)
    tune_idx = np.arange(tune_start, fold.train_end, dtype=int)
    if len(tune_idx) < 500: raise ValueError(f"{fold.name}: tuning insuficiente")

    inner_models = fit_models(ds, inner_train_idx)
    tune_scores = predict_score(inner_models, ds.X[tune_idx])
    calibration = select_thresholds(ds, tune_idx, tune_scores)

    outer_train_idx = np.arange(fold.train_start, fold.train_end, dtype=int)
    outer_models = fit_models(ds, outer_train_idx)
    val_idx = np.arange(fold.val_start, fold.val_end, dtype=int)
    val_scores = predict_score(outer_models, ds.X[val_idx])
    entry, exit_ = float(calibration["entry_pct"]), float(calibration["exit_pct"])
    metrics = backtest(ds, val_idx, val_scores, entry, exit_)
    metrics.update(prediction_diagnostics(ds, val_idx, val_scores, entry))
    metrics["score_distribution"] = score_distribution(val_scores)

    eligible = bool(calibration["signal_sufficient"] and metrics["trades"] >= MIN_VALIDATION_TRADES)
    return {
        "name": fold.name, "train_samples": len(outer_train_idx), "validation_samples": len(val_idx),
        "tuning_samples": len(tune_idx), "outer_gap_bars": fold.val_start - fold.train_end,
        "inner_gap_bars": tune_start - inner_train_end,
        "thresholds": {"entry_pct": entry, "exit_pct": exit_, "entry_quantile": calibration["entry_quantile"]},
        "calibration": calibration, "validation": metrics, "eligible": eligible,
    }


def summarize_folds(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [r["validation"] for r in results]
    eligible_results = [r for r in results if r["eligible"]]
    eligible_vals = [r["validation"] for r in eligible_results]

    def med(source: List[Dict[str, Any]], key: str) -> float:
        return float(np.median([float(v[key]) for v in source])) if source else 0.0

    positive = sum(1 for v in eligible_vals if v["return_pct"] > 0)
    alpha_positive = sum(1 for v in eligible_vals if v["alpha_pct"] > 0)
    source = eligible_vals if eligible_vals else vals
    summary = {
        "folds": len(vals), "eligible_folds": len(eligible_vals), "positive_folds": positive,
        "alpha_positive_folds": alpha_positive, "median_return_pct": med(source, "return_pct"),
        "median_alpha_pct": med(source, "alpha_pct"), "median_profit_factor": med(source, "profit_factor"),
        "median_drawdown_pct": med(source, "max_drawdown_pct"), "median_sharpe_daily": med(source, "sharpe_daily"),
        "median_sortino_daily": med(source, "sortino_daily"), "median_cvar95_daily_pct": med(source, "cvar95_daily_pct"),
        "median_direction_accuracy_pct": med(source, "direction_accuracy_pct"),
        "median_signal_direction_accuracy_pct": med(source, "signal_direction_accuracy_pct"),
        "total_trades": int(sum(v["trades"] for v in eligible_vals)),
        "all_trades_observed": int(sum(v["trades"] for v in vals)),
        "signal_insufficient_folds": sum(1 for r in results if not r["calibration"]["signal_sufficient"]),
    }
    worst_ret = min((v["return_pct"] for v in eligible_vals), default=-999.0)
    summary["candidate_walk_forward"] = bool(
        len(eligible_vals) == len(vals)
        and positive >= 2 and alpha_positive >= 2
        and summary["median_return_pct"] > 0 and summary["median_alpha_pct"] > 0
        and summary["median_profit_factor"] >= 1.05 and summary["median_drawdown_pct"] <= 10.0
        and summary["total_trades"] >= 20 and worst_ret > -1.0
    )
    return summary
