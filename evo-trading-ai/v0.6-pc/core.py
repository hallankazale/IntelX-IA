from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, json, math, random, time
from statistics import median
from typing import List, Dict, Any, Tuple

import numpy as np
import requests

ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

POSITION_FRACTION = 0.25
FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0002
STOP_LOSS = 0.03
TAKE_PROFIT = 0.06
MAX_HOLD_BARS = 48
EPISODE_BARS = 512
WARMUP = 180
DATA_DAYS = 180
HOLDOUT_GAP_DAYS = 14


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketData:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache = self.data_dir / f"btcusdt_5m_{DATA_DAYS}d_gap{HOLDOUT_GAP_DAYS}.csv"

    def load_or_download(self, days: int = DATA_DAYS, gap_days: int = HOLDOUT_GAP_DAYS,
                         max_age_hours: int = 24) -> List[Candle]:
        if self.cache.exists() and time.time() - self.cache.stat().st_mtime < max_age_hours * 3600:
            rows = self._load_csv()
            if len(rows) > 10000:
                return rows
        rows = self._download(days, gap_days)
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

    def _download(self, days: int, gap_days: int) -> List[Candle]:
        interval_ms = 5 * 60 * 1000
        end = int((time.time() - gap_days * 86400) * 1000)
        start = end - days * 86400 * 1000
        urls = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
            "https://api1.binance.com/api/v3/klines",
        ]
        out: List[Candle] = []
        cursor = start
        while cursor < end:
            params = {"symbol": "BTCUSDT", "interval": "5m", "startTime": cursor,
                      "endTime": end, "limit": 1000}
            data = None
            last_err = None
            for url in urls:
                try:
                    r = requests.get(url, params=params, timeout=25)
                    r.raise_for_status()
                    candidate = r.json()
                    if isinstance(candidate, list):
                        data = candidate
                        break
                except Exception as exc:
                    last_err = exc
            if not data:
                if out:
                    break
                raise RuntimeError(f"Falha ao baixar candles da Binance: {last_err}")
            for x in data:
                out.append(Candle(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])))
            nxt = int(data[-1][0]) + interval_ms
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.03)
        dedup = {c.open_time: c for c in out}
        rows = [dedup[k] for k in sorted(dedup)]
        if len(rows) < 30000:
            raise RuntimeError(f"Poucos candles baixados: {len(rows)}")
        return rows


def ema(v: np.ndarray, p: int) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    out[0] = v[0]
    a = 2.0 / (p + 1.0)
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


def rsi(close: np.ndarray, p: int = 14) -> np.ndarray:
    out = np.full(len(close), 50.0)
    g = l = 0.0
    for i in range(1, len(close)):
        ch = close[i] - close[i - 1]
        gain, loss = max(0.0, ch), max(0.0, -ch)
        if i <= p:
            g += gain
            l += loss
            if i == p:
                g /= p
                l /= p
        else:
            g = ((g * (p - 1)) + gain) / p
            l = ((l * (p - 1)) + loss) / p
        if i >= p:
            out[i] = 100.0 if l < 1e-12 else 100.0 - 100.0 / (1.0 + g / l)
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, p: int = 14) -> np.ndarray:
    tr = np.empty(len(close), dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    out = np.empty(len(close), dtype=float)
    avg = tr[0]
    out[0] = avg
    for i in range(1, len(tr)):
        avg = ((avg * i) + tr[i]) / (i + 1) if i < p else ((avg * (p - 1)) + tr[i]) / p
        out[i] = avg
    return out


def _bucket3(x: float, lo: float, hi: float) -> int:
    return 0 if x < lo else (2 if x > hi else 1)


def higher_timeframe_trend(close: np.ndarray, factor: int, fast: int, slow: int,
                           threshold: float = 0.0015) -> np.ndarray:
    """Trend using only completed higher-timeframe candles; avoids partial-bar lookahead."""
    n = len(close)
    completed_ends = list(range(factor - 1, n, factor))
    agg = np.array([close[i] for i in completed_ends], dtype=float)
    if len(agg) == 0:
        return np.ones(n, dtype=np.int8)
    ef, es = ema(agg, fast), ema(agg, slow)
    trend_agg = np.ones(len(agg), dtype=np.int8)
    for k in range(len(agg)):
        px = max(agg[k], 1e-9)
        d = (ef[k] - es[k]) / px
        trend_agg[k] = 2 if d > threshold else (0 if d < -threshold else 1)
    out = np.ones(n, dtype=np.int8)
    for i in range(n):
        k = (i + 1) // factor - 1
        if k >= 0:
            out[i] = trend_agg[min(k, len(trend_agg) - 1)]
    return out


class PreparedMarket:
    """Discretized multi-timeframe features. Decisions are made after each 5m candle closes."""
    def __init__(self, candles: List[Candle]):
        if len(candles) < 5000:
            raise ValueError("Poucos candles para preparar mercado v0.6")
        self.candles = candles
        self.warmup = WARMUP
        close = np.array([c.close for c in candles], dtype=float)
        high = np.array([c.high for c in candles], dtype=float)
        low = np.array([c.low for c in candles], dtype=float)
        vol = np.array([c.volume for c in candles], dtype=float)

        e9, e21 = ema(close, 9), ema(close, 21)
        rr = rsi(close, 14)
        aa = atr(high, low, close, 14)
        vs = sma(vol, 20)
        trend15 = higher_timeframe_trend(close, 3, 6, 14)
        trend1h = higher_timeframe_trend(close, 12, 4, 12)

        base = np.zeros(len(candles), dtype=np.int32)
        radices = (3, 3, 2, 2, 3, 3, 3, 3)
        for i in range(len(candles)):
            px = max(close[i], 1e-9)
            td = (e9[i] - e21[i]) / px
            trend5 = _bucket3(td, -0.001, 0.001)
            mom = _bucket3(rr[i], 42.0, 58.0)
            vola = 1 if aa[i] / px > 0.006 else 0
            hv = 1 if vol[i] > max(vs[i], 1e-9) * 1.20 else 0
            ret3 = 0.0 if i < 3 else close[i] / max(close[i - 3], 1e-9) - 1.0
            ret12 = 0.0 if i < 12 else close[i] / max(close[i - 12], 1e-9) - 1.0
            r3 = _bucket3(ret3, -0.003, 0.003)
            r12 = _bucket3(ret12, -0.006, 0.006)
            vals = (trend5, mom, vola, hv, int(trend15[i]), int(trend1h[i]), r3, r12)
            code = 0
            for value, radix in zip(vals, radices):
                code = code * radix + int(value)
            base[i] = code
        self.base_states = base
        self.base_state_count = int(np.prod(radices))

    def state_at(self, i: int, portfolio: "Portfolio") -> int:
        base = int(self.base_states[i])
        if not portfolio.in_position():
            pos_code = 0
        else:
            held = 0 if portfolio.bars_held <= 12 else (1 if portfolio.bars_held <= 30 else 2)
            px = self.candles[i].close
            pnl = px / max(portfolio.entry, 1e-9) - 1.0
            pnl_bin = _bucket3(pnl, -0.01, 0.01)
            pos_code = 1 + held * 3 + pnl_bin
        return base * 10 + pos_code


class Portfolio:
    def __init__(self, cash: float = 1000.0):
        self.cash = cash
        self.asset = 0.0
        self.entry = 0.0
        self.cost_basis = 0.0
        self.bars_held = 0
        self.trades = 0
        self.wins = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.turnover_events = 0

    def in_position(self) -> bool:
        return self.asset > 1e-12

    def equity(self, px: float) -> float:
        return self.cash + self.asset * px

    def buy(self, px: float) -> bool:
        if self.in_position():
            return False
        alloc = self.cash * POSITION_FRACTION
        if alloc < 1e-9:
            return False
        exe = px * (1 + SLIPPAGE_RATE)
        fee = alloc * FEE_RATE
        net = alloc - fee
        self.asset = net / exe
        self.cash -= alloc
        self.entry = exe
        self.cost_basis = alloc
        self.bars_held = 0
        self.turnover_events += 1
        return True

    def sell(self, px: float) -> bool:
        if not self.in_position():
            return False
        exe = px * (1 - SLIPPAGE_RATE)
        gross = self.asset * exe
        fee = gross * FEE_RATE
        net = gross - fee
        pnl = net - self.cost_basis
        self.cash += net
        self.asset = 0.0
        self.entry = 0.0
        self.cost_basis = 0.0
        self.bars_held = 0
        self.trades += 1
        self.turnover_events += 1
        if pnl >= 0:
            self.wins += 1
            self.gross_profit += pnl
        else:
            self.gross_loss += -pnl
        return True

    def apply(self, action: int, px: float) -> bool:
        if action == ACTION_BUY:
            return self.buy(px)
        if action == ACTION_SELL:
            return self.sell(px)
        return False

    def risk(self, next_c: Candle) -> bool:
        if not self.in_position():
            return False
        self.bars_held += 1
        stop = self.entry * (1 - STOP_LOSS)
        take = self.entry * (1 + TAKE_PROFIT)
        if next_c.low <= stop:
            return self.sell(stop)
        if next_c.high >= take:
            return self.sell(take)
        if self.bars_held >= MAX_HOLD_BARS:
            return self.sell(next_c.close)
        return False


class SparseQModel:
    def __init__(self, q: Dict[str, List[float]] | None = None,
                 visits: Dict[str, List[int]] | None = None,
                 episodes: int = 0, seed: int = 1337):
        self.q: Dict[int, np.ndarray] = {}
        if q:
            for k, row in q.items():
                self.q[int(k)] = np.array(row, dtype=float)
        self.visits: Dict[int, List[int]] = {}
        if visits:
            for k, row in visits.items():
                vals = [int(x) for x in row[:3]]
                while len(vals) < 3:
                    vals.append(0)
                self.visits[int(k)] = vals
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.rng = random.Random(seed + episodes)

    def _row(self, state: int) -> np.ndarray:
        row = self.q.get(state)
        if row is None:
            row = np.zeros(3, dtype=float)
            self.q[state] = row
        return row

    def _visits(self, state: int) -> List[int]:
        return self.visits.setdefault(state, [0, 0, 0])

    def epsilon(self) -> float:
        return max(0.06, 0.40 * math.exp(-self.episodes / 3000.0))

    @staticmethod
    def valid_actions(in_position: bool) -> Tuple[int, int]:
        return (ACTION_HOLD, ACTION_SELL) if in_position else (ACTION_HOLD, ACTION_BUY)

    def greedy(self, state: int, in_position: bool, deterministic: bool = False) -> int:
        valid = self.valid_actions(in_position)
        row = self._row(state)
        mx = max(row[a] for a in valid)
        best = [a for a in valid if abs(row[a] - mx) < 1e-12]
        if len(best) == 1:
            return best[0]
        if deterministic:
            idx = ((state * 1103515245 + self.seed * 12345) & 0x7FFFFFFF) % len(best)
            return best[idx]
        return self.rng.choice(best)

    def choose(self, state: int, in_position: bool, explore: bool = True) -> int:
        valid = self.valid_actions(in_position)
        counts = self._visits(state)
        if explore:
            untried = [a for a in valid if counts[a] == 0]
            if untried:
                action = self.rng.choice(untried)
                counts[action] += 1
                return action
            if self.rng.random() < self.epsilon():
                action = self.rng.choice(valid)
                counts[action] += 1
                return action
        action = self.greedy(state, in_position, deterministic=False)
        counts[action] += 1
        return action

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_in_position: bool) -> None:
        alpha = max(0.018, 0.10 / math.sqrt(1.0 + self.episodes / 1800.0))
        gamma = 0.965
        row = self._row(state)
        next_row = self._row(next_state)
        next_best = max(next_row[a] for a in self.valid_actions(next_in_position))
        row[action] += alpha * (reward + gamma * next_best - row[action])

    def export_q(self) -> Dict[str, List[float]]:
        return {str(k): [float(x) for x in v] for k, v in self.q.items()}

    def export_visits(self) -> Dict[str, List[int]]:
        return {str(k): list(v) for k, v in self.visits.items()}

    def clone_q(self) -> Dict[str, List[float]]:
        return self.export_q()


class Trainer:
    def __init__(self, market: PreparedMarket, seed: int = 1337):
        self.market = market
        self.rng = random.Random(seed)

    def train_episode(self, model: SparseQModel, start: int, end_exclusive: int) -> Dict[str, int]:
        min_start = max(self.market.warmup, start)
        latest = end_exclusive - EPISODE_BARS - 2
        if latest <= min_start:
            raise ValueError("Faixa de treino insuficiente")
        i0 = self.rng.randint(min_start, latest)
        end = min(end_exclusive - 1, i0 + EPISODE_BARS)
        p = Portfolio(1000.0)
        peak = 1000.0
        counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
        for i in range(i0, end):
            cur, nxt = self.market.candles[i], self.market.candles[i + 1]
            before = max(1e-9, p.equity(cur.close))
            inpos = p.in_position()
            state = self.market.state_at(i, p)
            action = model.choose(state, inpos, True)
            counts[ACTION_NAMES[action]] += 1
            traded = p.apply(action, cur.close)
            risk_closed = p.risk(nxt)
            after = max(1e-9, p.equity(nxt.close))
            peak = max(peak, after)
            dd = max(0.0, (peak - after) / peak)
            agent_ret = 100.0 * math.log(after / before)
            benchmark_ret = POSITION_FRACTION * 100.0 * math.log(max(1e-9, nxt.close) / max(1e-9, cur.close))
            alpha_step = agent_ret - benchmark_ret
            turnover_penalty = 0.0025 * (int(traded) + int(risk_closed))
            risk_penalty = 0.002 * (dd * 100.0)
            reward = agent_ret + 0.15 * alpha_step - risk_penalty - turnover_penalty
            next_state = self.market.state_at(i + 1, p)
            model.update(state, action, reward, next_state, p.in_position())
        p.sell(self.market.candles[end].close)
        model.episodes += 1
        return counts


def _daily_returns(times: List[int], equities: List[float]) -> List[float]:
    if not times or not equities:
        return []
    daily_ends: List[float] = []
    current_day = times[0] // 86400000
    last_eq = equities[0]
    for ts, eq in zip(times, equities):
        day = ts // 86400000
        if day != current_day:
            daily_ends.append(last_eq)
            current_day = day
        last_eq = eq
    daily_ends.append(last_eq)
    if len(daily_ends) < 2:
        return []
    return [daily_ends[i] / max(daily_ends[i - 1], 1e-9) - 1.0 for i in range(1, len(daily_ends))]


def sharpe_daily(xs: List[float]) -> float:
    if len(xs) < 5:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var)
    return 0.0 if sd < 1e-12 else (m / sd) * math.sqrt(365.0)


def sortino_daily(xs: List[float]) -> float:
    if len(xs) < 5:
        return 0.0
    m = sum(xs) / len(xs)
    downside = [min(0.0, x) for x in xs]
    dd = math.sqrt(sum(x * x for x in downside) / max(1, len(downside)))
    return 0.0 if dd < 1e-12 else (m / dd) * math.sqrt(365.0)


def cvar95_daily_pct(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    k = max(1, math.ceil(len(ordered) * 0.05))
    return 100.0 * sum(ordered[:k]) / k


def benchmark_with_costs(start_px: float, end_px: float) -> float:
    cash = 1000.0
    alloc = cash * POSITION_FRACTION
    entry_exec = start_px * (1 + SLIPPAGE_RATE)
    entry_fee = alloc * FEE_RATE
    asset = (alloc - entry_fee) / entry_exec
    cash -= alloc
    exit_exec = end_px * (1 - SLIPPAGE_RATE)
    gross = asset * exit_exec
    exit_fee = gross * FEE_RATE
    cash += gross - exit_fee
    return (cash / 1000.0 - 1.0) * 100.0


def evaluate(market: PreparedMarket, q: Dict[str, List[float]], start: int,
             end_exclusive: int, seed: int = 7) -> Dict[str, Any]:
    model = SparseQModel(q=q, episodes=999999, seed=seed)
    p = Portfolio(1000.0)
    first = max(market.warmup, start)
    last = min(end_exclusive - 1, len(market.candles) - 2)
    if last <= first + 5:
        return {"return_pct": 0, "benchmark_pct": 0, "alpha_pct": 0, "max_drawdown_pct": 0,
                "profit_factor": 0, "win_rate_pct": 0, "sharpe": 0, "sortino": 0,
                "cvar95_daily_pct": 0, "trades": 0, "actions": {"HOLD": 0, "BUY": 0, "SELL": 0}}
    peak = 1000.0
    maxdd = 0.0
    counts = {"HOLD": 0, "BUY": 0, "SELL": 0}
    times: List[int] = []
    equities: List[float] = []
    for i in range(first, last + 1):
        cur, nxt = market.candles[i], market.candles[i + 1]
        state = market.state_at(i, p)
        action = model.greedy(state, p.in_position(), deterministic=True)
        counts[ACTION_NAMES[action]] += 1
        p.apply(action, cur.close)
        p.risk(nxt)
        eq = p.equity(nxt.close)
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak)
        times.append(nxt.open_time)
        equities.append(eq)
    final_px = market.candles[last + 1].close
    p.sell(final_px)
    final_eq = p.equity(final_px)
    ret = (final_eq / 1000.0 - 1.0) * 100.0
    benchmark = benchmark_with_costs(market.candles[first].close, final_px)
    pf = 99.0 if p.gross_loss < 1e-9 and p.gross_profit > 0 else (0.0 if p.gross_loss < 1e-9 else p.gross_profit / p.gross_loss)
    wr = 0.0 if p.trades == 0 else p.wins * 100.0 / p.trades
    daily = _daily_returns(times, equities)
    return {
        "return_pct": ret,
        "benchmark_pct": benchmark,
        "alpha_pct": ret - benchmark,
        "max_drawdown_pct": maxdd * 100.0,
        "profit_factor": pf,
        "win_rate_pct": wr,
        "sharpe": sharpe_daily(daily),
        "sortino": sortino_daily(daily),
        "cvar95_daily_pct": cvar95_daily_pct(daily),
        "trades": p.trades,
        "actions": counts,
        "final_equity": final_eq,
        "turnover_events": p.turnover_events,
    }


def walk_forward_summary(market: PreparedMarket, fold_models: List[SparseQModel],
                         folds: List[Tuple[int, int]]) -> Dict[str, Any]:
    metrics = []
    for idx, (val_start, val_end) in enumerate(folds):
        metrics.append(evaluate(market, fold_models[idx].export_q(), val_start, val_end, seed=70 + idx))
    returns = [m["return_pct"] for m in metrics]
    alphas = [m["alpha_pct"] for m in metrics]
    pfs = [m["profit_factor"] for m in metrics]
    dds = [m["max_drawdown_pct"] for m in metrics]
    positive = sum(1 for m in metrics if m["return_pct"] > 0 and m["profit_factor"] > 1.0)
    return {
        "folds": metrics,
        "median_return_pct": median(returns),
        "median_alpha_pct": median(alphas),
        "median_profit_factor": median(pfs),
        "max_fold_drawdown_pct": max(dds),
        "positive_folds": positive,
    }


def walk_forward_score(summary: Dict[str, Any]) -> float:
    return (
        summary["median_return_pct"]
        + 0.6 * summary["median_alpha_pct"]
        + min(3.0, summary["median_profit_factor"]) * 1.5
        - 0.8 * summary["max_fold_drawdown_pct"]
        + 1.0 * summary["positive_folds"]
    )


def dump_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
