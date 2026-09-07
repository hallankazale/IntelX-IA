from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier

POSITION_FRACTION = 0.25
FEE_RATE = 0.0010
SLIPPAGE_RATE = 0.0002
ROUND_TRIP_COST_PCT = 2.0 * (FEE_RATE + SLIPPAGE_RATE) * 100.0

STOP_LOSS = 0.0065
TAKE_PROFIT = 0.0150
MAX_HOLD_BARS = 48

HORIZONS = (12, 24, 48)
MAX_HORIZON = max(HORIZONS)
BARRIERS = {
    12: (0.0075, 0.0035),
    24: (0.0100, 0.0045),
    48: (0.0150, 0.0065),
}
NEUTRAL_CLOSE_EDGE_PCT = ROUND_TRIP_COST_PCT + 0.06

WARMUP = 240
OUTER_GAP_BARS = 14 * 24 * 12
INNER_GAP_BARS = 3 * 24 * 12
MIN_TUNING_TRADES = 8
MIN_VALIDATION_TRADES = 8

PROB_QUANTILES = (0.75, 0.80, 0.85, 0.90, 0.925, 0.95)
EDGE_QUANTILES = (0.75, 0.80, 0.85, 0.90, 0.95)
EXIT_DOWN_PROBS = (0.45, 0.50, 0.55)

FEATURE_NAMES = (
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48", "ret_96", "ret_288",
    "ema9_21", "ema21_50", "ema50_200",
    "ema9_slope3", "ema21_slope6", "ema50_slope12", "ema200_slope48",
    "dist_ema21", "dist_ema50", "dist_ema200",
    "rsi14", "atr_pct",
    "vol_ratio20", "vol_ratio96",
    "rv_12", "rv_48", "rv_96", "rv_288",
    "range_pct",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
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
        self.cache = self.data_dir / "btcusdt_5m_365d.csv"

    def load_or_download(self, days: int = 365, max_age_hours: int = 6) -> List[Candle]:
        if self.cache.exists() and time.time() - self.cache.stat().st_mtime < max_age_hours * 3600:
            rows = self._load_csv()
            if len(rows) > 70_000:
                return rows
        rows = self._download(days)
        self._save_csv(rows)
        return rows

    def _load_csv(self) -> List[Candle]:
        out: List[Candle] = []
        with self.cache.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(Candle(int(r["open_time"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])))
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
        urls = ["https://data-api.binance.vision/api/v3/klines", "https://api.binance.com/api/v3/klines", "https://api1.binance.com/api/v3/klines"]
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
            time.sleep(0.03)
        dedup = {c.open_time: c for c in out}
        rows = [dedup[k] for k in sorted(dedup)]
        if len(rows) < 70_000:
            raise RuntimeError(f"Poucos candles baixados para 365 dias: {len(rows)}")
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
        e9, e21, e50, e200 = ema(close, 9), ema(close, 21), ema(close, 50), ema(close, 200)
        rr, aa = rsi(close, 14), atr(high, low, close, 14)
        v20, v96 = sma(vol, 20), sma(vol, 96)
        ret1 = safe_log_return(close, 1)
        def slope(arr: np.ndarray, lag: int) -> np.ndarray:
            out = np.zeros(len(arr), dtype=float)
            out[lag:] = np.log(np.maximum(arr[lag:], 1e-12) / np.maximum(arr[:-lag], 1e-12))
            return out
        hours = np.array([(c.open_time // 3_600_000) % 24 for c in candles], dtype=float)
        days = np.array([(c.open_time // 86_400_000 + 4) % 7 for c in candles], dtype=float)
        hour_angle = hours / 24.0 * 2.0 * math.pi
        dow_angle = days / 7.0 * 2.0 * math.pi
        X = np.column_stack([
            ret1, safe_log_return(close, 3), safe_log_return(close, 6), safe_log_return(close, 12), safe_log_return(close, 24), safe_log_return(close, 48), safe_log_return(close, 96), safe_log_return(close, 288),
            np.log(np.maximum(e9, 1e-12) / np.maximum(e21, 1e-12)), np.log(np.maximum(e21, 1e-12) / np.maximum(e50, 1e-12)), np.log(np.maximum(e50, 1e-12) / np.maximum(e200, 1e-12)),
            slope(e9, 3), slope(e21, 6), slope(e50, 12), slope(e200, 48),
            np.log(np.maximum(close, 1e-12) / np.maximum(e21, 1e-12)), np.log(np.maximum(close, 1e-12) / np.maximum(e50, 1e-12)), np.log(np.maximum(close, 1e-12) / np.maximum(e200, 1e-12)),
            (rr - 50.0) / 50.0, aa / np.maximum(close, 1e-12), vol / np.maximum(v20, 1e-12) - 1.0, vol / np.maximum(v96, 1e-12) - 1.0,
            rolling_std(ret1, 12), rolling_std(ret1, 48), rolling_std(ret1, 96), rolling_std(ret1, 288),
            (high - low) / np.maximum(close, 1e-12), np.sin(hour_angle), np.cos(hour_angle), np.sin(dow_angle), np.cos(dow_angle),
        ])
        self.X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.close = close
        self.labels: Dict[int, np.ndarray] = {h: self._build_event_labels(h) for h in HORIZONS}

    @property
    def feature_count(self) -> int:
        return self.X.shape[1]

    def _build_event_labels(self, horizon: int) -> np.ndarray:
        up_pct, down_pct = BARRIERS[horizon]
        n = len(self.candles)
        y = np.full(n, np.nan, dtype=float)
        for i in range(WARMUP, n - horizon - 1):
            entry = self.candles[i + 1].open
            up = entry * (1.0 + up_pct); down = entry * (1.0 - down_pct)
            label = 0.0; resolved = False
            for j in range(i + 1, i + horizon + 1):
                c = self.candles[j]
                hit_down = c.low <= down; hit_up = c.high >= up
                if hit_down and hit_up:
                    label = -1.0; resolved = True; break
                if hit_down:
                    label = -1.0; resolved = True; break
                if hit_up:
                    label = 1.0; resolved = True; break
            if not resolved:
                final_close = self.candles[i + horizon].close
                gross_pct = (final_close / entry - 1.0) * 100.0
                if gross_pct >= NEUTRAL_CLOSE_EDGE_PCT: label = 1.0
                elif gross_pct <= -NEUTRAL_CLOSE_EDGE_PCT: label = -1.0
                else: label = 0.0
            y[i] = label
        return y

def build_protocol(n: int) -> Protocol:
    if n < 70_000: raise ValueError("Histórico insuficiente para protocolo v0.8")
    holdout_len = max(8_000, int(n * 0.15)); holdout_start = n - holdout_len
    gap = min(OUTER_GAP_BARS, max(2_000, int(n * 0.05)))
    dev_start = WARMUP; dev_end = holdout_start - gap
    if dev_end - dev_start < 40_000: raise ValueError("Faixa de desenvolvimento insuficiente")
    span = dev_end - dev_start; val_len = max(4_000, int(span * 0.10))
    folds: List[FoldSpec] = []
    for idx, ratio in enumerate((0.42, 0.58, 0.74), start=1):
        train_end = dev_start + int(span * ratio); val_start = train_end + gap; val_end = min(val_start + val_len, dev_end)
        if val_end - val_start < 3_000: raise ValueError("Fold muito curto após embargo temporal")
        folds.append(FoldSpec(f"Fold {idx}", dev_start, train_end, val_start, val_end))
    return Protocol(dev_start, dev_end, holdout_start, n, gap, tuple(folds))

def _new_model():
    return HistGradientBoostingClassifier(loss="log_loss", learning_rate=0.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=3.0, max_bins=127, early_stopping=False, random_state=1337)

def fit_models(ds: FeatureDataset, indices: np.ndarray) -> Dict[int, Any]:
    if len(indices) < 2_000: raise ValueError("Poucas amostras para treinar classificador")
    models: Dict[int, Any] = {}
    for h in HORIZONS:
        valid = indices[np.isfinite(ds.labels[h][indices])]; y = ds.labels[h][valid].astype(int); classes = np.unique(y)
        m = DummyClassifier(strategy="prior") if len(classes) < 2 else _new_model()
        m.fit(ds.X[valid], y); models[h] = m
    return models

def _class_prob(model: Any, X: np.ndarray, cls: int) -> np.ndarray:
    probs = np.asarray(model.predict_proba(X), dtype=float); classes = list(map(int, model.classes_))
    if cls not in classes: return np.zeros(len(X), dtype=float)
    return probs[:, classes.index(cls)]

def predict_signal(models: Dict[int, Any], X: np.ndarray) -> Dict[str, np.ndarray]:
    ups, downs = [], []
    for h in HORIZONS:
        m = models[h]; ups.append(_class_prob(m, X, 1)); downs.append(_class_prob(m, X, -1))
    p_up = np.median(np.vstack(ups), axis=0); p_down = np.median(np.vstack(downs), axis=0)
    return {"p_up": p_up, "p_down": p_down, "edge": p_up - p_down}

class Portfolio:
    def __init__(self, cash: float = 1000.0):
        self.cash=cash; self.asset=0.0; self.entry=0.0; self.cost_basis=0.0; self.bars_held=0; self.trades=0; self.wins=0; self.gross_profit=0.0; self.gross_loss=0.0
    def in_position(self)->bool: return self.asset > 1e-12
    def equity(self, px: float)->float: return self.cash + self.asset*px
    def buy(self, px: float)->bool:
        if self.in_position(): return False
        alloc=self.cash*POSITION_FRACTION
        if alloc<1e-9: return False
        exe=px*(1+SLIPPAGE_RATE); fee=alloc*FEE_RATE; self.asset=(alloc-fee)/exe; self.cash-=alloc; self.entry=exe; self.cost_basis=alloc; self.bars_held=0; return True
    def sell(self, px: float)->bool:
        if not self.in_position(): return False
        exe=px*(1-SLIPPAGE_RATE); gross=self.asset*exe; fee=gross*FEE_RATE; net=gross-fee; pnl=net-self.cost_basis; self.cash+=net; self.asset=0.0; self.entry=0.0; self.cost_basis=0.0; self.bars_held=0; self.trades+=1
        if pnl>=0: self.wins+=1; self.gross_profit+=pnl
        else: self.gross_loss+=-pnl
        return True
    def apply_risk(self, candle:Candle)->bool:
        if not self.in_position(): return False
        self.bars_held+=1; stop=self.entry*(1-STOP_LOSS); take=self.entry*(1+TAKE_PROFIT); hit_stop=candle.low<=stop; hit_take=candle.high>=take
        if hit_stop and hit_take: return self.sell(stop)
        if hit_stop: return self.sell(stop)
        if hit_take: return self.sell(take)
        if self.bars_held>=MAX_HOLD_BARS: return self.sell(candle.close)
        return False

def _daily_returns(curve:List[float], bars_per_day:int=288)->List[float]:
    if len(curve)<bars_per_day+1:return []
    pts=[curve[0]]+[curve[i] for i in range(bars_per_day,len(curve),bars_per_day)]
    if pts[-1]!=curve[-1]:pts.append(curve[-1])
    return [pts[i]/pts[i-1]-1 for i in range(1,len(pts)) if pts[i-1]>0]

def _sharpe_daily(xs:List[float])->float:
    if len(xs)<3:return 0.0
    a=np.asarray(xs,dtype=float); sd=float(np.std(a,ddof=1)); return 0.0 if sd<1e-12 else float(np.mean(a)/sd*math.sqrt(365))

def _sortino_daily(xs:List[float])->float:
    if len(xs)<3:return 0.0
    a=np.asarray(xs,dtype=float); d=a[a<0]
    if len(d)<2:return 0.0
    sd=float(np.std(d,ddof=1)); return 0.0 if sd<1e-12 else float(np.mean(a)/sd*math.sqrt(365))

def _cvar_95_daily(xs:List[float])->float:
    if not xs:return 0.0
    a=np.sort(np.asarray(xs,dtype=float)); k=max(1,int(math.ceil(len(a)*0.05))); return float(np.mean(a[:k])*100.0)

def fair_benchmark(candles:List[Candle], start:int, end_exclusive:int)->float:
    first=candles[start].open; last=candles[end_exclusive-1].close; p=Portfolio(1000.0); p.buy(first); p.sell(last); return (p.equity(last)/1000.0-1)*100.0

def backtest(ds:FeatureDataset, indices:np.ndarray, signal:Dict[str,np.ndarray], entry_prob:float, entry_edge:float, exit_down_prob:float)->Dict[str,Any]:
    n=len(indices)
    if any(len(signal[k])!=n for k in ("p_up","p_down","edge")):raise ValueError("sinais e índices incompatíveis")
    if n<10:raise ValueError("Janela de backtest muito curta")
    p=Portfolio(1000.0); peak=1000.0; max_dd=0.0; curve=[1000.0]; exposure_bars=0; entry_signals=0
    for pos,i in enumerate(indices):
        if i+1>=len(ds.candles):break
        nxt=ds.candles[i+1]; pu=float(signal["p_up"][pos]); pd=float(signal["p_down"][pos]); edge=float(signal["edge"][pos])
        if not p.in_position():
            if pu>=entry_prob and edge>=entry_edge:
                entry_signals+=1; p.buy(nxt.open)
                if p.in_position(): exposure_bars+=1; p.apply_risk(nxt)
        else:
            exposure_bars+=1
            if pd>=exit_down_prob or edge<=0.0: p.sell(nxt.open)
            else: p.apply_risk(nxt)
        eq=p.equity(nxt.close); peak=max(peak,eq); max_dd=max(max_dd,0.0 if peak<=0 else (peak-eq)/peak); curve.append(eq)
    final_i=min(int(indices[-1])+1,len(ds.candles)-1); final_px=ds.candles[final_i].close; p.sell(final_px); final_eq=p.equity(final_px); curve[-1]=final_eq
    ret=(final_eq/1000.0-1)*100.0; benchmark=fair_benchmark(ds.candles,int(indices[0])+1,min(int(indices[-1])+2,len(ds.candles)))
    pf=(99.0 if p.gross_profit>0 else 0.0) if p.gross_loss<1e-12 else p.gross_profit/p.gross_loss; wr=0.0 if p.trades==0 else p.wins*100.0/p.trades; daily=_daily_returns(curve)
    return {"return_pct":ret,"benchmark_pct":benchmark,"alpha_pct":ret-benchmark,"max_drawdown_pct":max_dd*100.0,"profit_factor":pf,"win_rate_pct":wr,"sharpe_daily":_sharpe_daily(daily),"sortino_daily":_sortino_daily(daily),"cvar95_daily_pct":_cvar_95_daily(daily),"trades":p.trades,"entry_signals":entry_signals,"exposure_pct":exposure_bars*100.0/max(1,n),"entry_prob":entry_prob,"entry_edge":entry_edge,"exit_down_prob":exit_down_prob,"final_equity":final_eq}

def signal_distribution(signal:Dict[str,np.ndarray])->Dict[str,float]:
    pu=np.asarray(signal["p_up"],dtype=float); pd=np.asarray(signal["p_down"],dtype=float); edge=np.asarray(signal["edge"],dtype=float)
    return {"p_up_mean":float(np.mean(pu)),"p_up_p85":float(np.quantile(pu,0.85)),"p_up_p90":float(np.quantile(pu,0.90)),"p_up_p95":float(np.quantile(pu,0.95)),"p_up_max":float(np.max(pu)),"p_down_mean":float(np.mean(pd)),"edge_mean":float(np.mean(edge)),"edge_p85":float(np.quantile(edge,0.85)),"edge_p90":float(np.quantile(edge,0.90)),"edge_p95":float(np.quantile(edge,0.95)),"edge_max":float(np.max(edge))}

def event_diagnostics(ds:FeatureDataset, indices:np.ndarray, signal:Dict[str,np.ndarray], entry_prob:float, entry_edge:float)->Dict[str,float]:
    labels=ds.labels[48][indices]; valid=np.isfinite(labels)
    if not np.any(valid):return {"base_up_rate_pct":0.0,"signal_precision_pct":0.0,"signal_lift":0.0,"signal_samples":0.0}
    y=labels[valid]; pu=signal["p_up"][valid]; edge=signal["edge"][valid]; selected=(pu>=entry_prob)&(edge>=entry_edge); base=float(np.mean(y==1)*100.0)
    if not np.any(selected):return {"base_up_rate_pct":base,"signal_precision_pct":0.0,"signal_lift":0.0,"signal_samples":0.0}
    precision=float(np.mean(y[selected]==1)*100.0); lift=0.0 if base<1e-12 else precision/base
    return {"base_up_rate_pct":base,"signal_precision_pct":precision,"signal_lift":lift,"signal_samples":float(np.sum(selected))}

def select_policy(ds:FeatureDataset, indices:np.ndarray, signal:Dict[str,np.ndarray])->Dict[str,Any]:
    pu=np.asarray(signal["p_up"],dtype=float); edge=np.asarray(signal["edge"],dtype=float); prob_candidates=sorted(set(float(np.quantile(pu,q)) for q in PROB_QUANTILES)); edge_candidates=sorted(set(max(0.02,float(np.quantile(edge,q))) for q in EDGE_QUANTILES))
    best=None; best_score=-1e18; tested=0
    for ep in prob_candidates:
        for ee in edge_candidates:
            for xd in EXIT_DOWN_PROBS:
                tested+=1; m=backtest(ds,indices,signal,ep,ee,xd)
                if m["trades"]<MIN_TUNING_TRADES:continue
                if m["trades"]>max(120,len(indices)//10):continue
                if m["exposure_pct"]>60.0:continue
                diag=event_diagnostics(ds,indices,signal,ep,ee)
                if diag["signal_samples"]<10:continue
                s=m["return_pct"]+0.7*m["alpha_pct"]-0.9*m["max_drawdown_pct"]+0.9*min(2.0,m["profit_factor"])+0.5*max(0.0,diag["signal_lift"]-1.0)-0.006*max(0,m["trades"]-50)
                if s>best_score:
                    best_score=s; best={"entry_prob":ep,"entry_edge":ee,"exit_down_prob":xd,"tuning":m,"tuning_event":diag}
    dist=signal_distribution(signal)
    if best is None:
        return {"policy_sufficient":False,"reason":"Nenhuma política probabilística gerou amostra mínima com risco aceitável.","tested_configs":tested,"distribution":dist,"tuning":None,"tuning_event":None,"entry_prob":float(np.quantile(pu,0.95)),"entry_edge":max(0.02,float(np.quantile(edge,0.95))),"exit_down_prob":0.50}
    return {"policy_sufficient":True,"reason":"Política calibrada somente na janela de tuning.","tested_configs":tested,"distribution":dist,**best}

def run_fold(ds:FeatureDataset, fold:FoldSpec)->Dict[str,Any]:
    train_len=fold.train_end-fold.train_start; inner_cut=fold.train_start+int(train_len*0.78); inner_train_end=max(fold.train_start+4000,inner_cut); tune_start=inner_train_end+INNER_GAP_BARS
    if fold.train_end-tune_start<2000:tune_start=inner_train_end+MAX_HORIZON
    inner_train_idx=np.arange(fold.train_start,inner_train_end,dtype=int); tune_idx=np.arange(tune_start,fold.train_end,dtype=int)
    if len(tune_idx)<1500:raise ValueError(f"{fold.name}: tuning insuficiente")
    inner_models=fit_models(ds,inner_train_idx); tune_signal=predict_signal(inner_models,ds.X[tune_idx]); policy=select_policy(ds,tune_idx,tune_signal)
    outer_train_idx=np.arange(fold.train_start,fold.train_end,dtype=int); outer_models=fit_models(ds,outer_train_idx); val_idx=np.arange(fold.val_start,fold.val_end,dtype=int); val_signal=predict_signal(outer_models,ds.X[val_idx])
    ep=float(policy["entry_prob"]); ee=float(policy["entry_edge"]); xd=float(policy["exit_down_prob"]); metrics=backtest(ds,val_idx,val_signal,ep,ee,xd); event=event_diagnostics(ds,val_idx,val_signal,ep,ee)
    eligible=bool(policy["policy_sufficient"] and metrics["trades"]>=MIN_VALIDATION_TRADES and event["signal_samples"]>=10)
    return {"name":fold.name,"train_samples":len(outer_train_idx),"validation_samples":len(val_idx),"tuning_samples":len(tune_idx),"outer_gap_bars":fold.val_start-fold.train_end,"inner_gap_bars":tune_start-inner_train_end,"policy":policy,"validation":metrics,"event":event,"validation_distribution":signal_distribution(val_signal),"eligible":eligible}

def summarize_folds(results:List[Dict[str,Any]])->Dict[str,Any]:
    vals=[r["validation"] for r in results]; eligible_results=[r for r in results if r["eligible"]]; eligible_vals=[r["validation"] for r in eligible_results]
    def med(source:List[Dict[str,Any]],key:str)->float:return float(np.median([float(v[key]) for v in source])) if source else 0.0
    def med_event(key:str)->float:return float(np.median([float(r["event"][key]) for r in eligible_results])) if eligible_results else 0.0
    positive=sum(1 for v in eligible_vals if v["return_pct"]>0); alpha_positive=sum(1 for v in eligible_vals if v["alpha_pct"]>0); source=eligible_vals if eligible_vals else vals; total_trades=int(sum(v["trades"] for v in eligible_vals)); worst_ret=min((v["return_pct"] for v in eligible_vals),default=-999.0)
    summary={"folds":len(vals),"eligible_folds":len(eligible_vals),"positive_folds":positive,"alpha_positive_folds":alpha_positive,"policy_insufficient_folds":sum(1 for r in results if not r["policy"]["policy_sufficient"]),"median_return_pct":med(source,"return_pct"),"median_alpha_pct":med(source,"alpha_pct"),"median_profit_factor":med(source,"profit_factor"),"median_drawdown_pct":med(source,"max_drawdown_pct"),"median_sharpe_daily":med(source,"sharpe_daily"),"median_sortino_daily":med(source,"sortino_daily"),"median_cvar95_daily_pct":med(source,"cvar95_daily_pct"),"median_signal_precision_pct":med_event("signal_precision_pct"),"median_base_up_rate_pct":med_event("base_up_rate_pct"),"median_signal_lift":med_event("signal_lift"),"total_trades":total_trades,"all_trades_observed":int(sum(v["trades"] for v in vals))}
    summary["candidate_walk_forward"]=bool(len(eligible_vals)==len(vals) and positive>=2 and alpha_positive>=2 and summary["median_return_pct"]>0 and summary["median_alpha_pct"]>0 and summary["median_profit_factor"]>=1.10 and summary["median_drawdown_pct"]<=8.0 and summary["median_signal_lift"]>=1.05 and total_trades>=30 and worst_ret>-1.5)
    return summary
