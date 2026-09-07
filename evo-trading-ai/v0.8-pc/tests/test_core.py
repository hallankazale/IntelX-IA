import math
import numpy as np

from core import (
    Candle, FeatureDataset, build_protocol, fit_models, predict_signal,
    backtest, event_diagnostics, summarize_folds, HORIZONS
)


def synthetic_candles(n=9000):
    rows=[]
    px=100.0
    for i in range(n):
        drift=0.00005 + 0.0012*math.sin(i/70.0) + 0.00045*math.sin(i/13.0)
        px*=math.exp(drift)
        spread=px*(0.003 + 0.001*abs(math.sin(i/17.0)))
        o=px*(1-0.0002*math.sin(i/5.0))
        h=max(o,px)+spread
        l=min(o,px)-spread
        rows.append(Candle(i*300_000,o,h,l,px,1000+250*math.sin(i/11.0)))
    return rows


def test_features_and_labels_are_causal_shapes():
    ds=FeatureDataset(synthetic_candles(1800))
    assert ds.X.shape[1] >= 25
    assert np.isfinite(ds.X).all()
    for h in HORIZONS:
        assert len(ds.labels[h]) == len(ds.candles)
        assert np.isfinite(ds.labels[h][400:800]).all()
        assert set(np.unique(ds.labels[h][400:800])).issubset({-1.0,0.0,1.0})


def test_protocol_has_14d_gap_and_locked_holdout():
    p=build_protocol(105_000)
    assert p.dev_end < p.holdout_start
    assert p.holdout_start - p.dev_end == p.gap_bars
    assert p.gap_bars >= 14*24*12
    assert len(p.folds)==3
    for f in p.folds:
        assert f.val_start - f.train_end == p.gap_bars
        assert f.val_end <= p.dev_end


def test_classifier_signal_smoke():
    ds=FeatureDataset(synthetic_candles(7000))
    train=np.arange(300,5000)
    val=np.arange(5100,6500)
    models=fit_models(ds,train)
    sig=predict_signal(models,ds.X[val])
    for k in ("p_up","p_down","edge"):
        assert len(sig[k])==len(val)
        assert np.isfinite(sig[k]).all()
    assert np.all((sig["p_up"]>=0)&(sig["p_up"]<=1))
    assert np.all((sig["p_down"]>=0)&(sig["p_down"]<=1))


def test_backtest_executes_next_open_and_returns_metrics():
    ds=FeatureDataset(synthetic_candles(3500))
    idx=np.arange(500,2500)
    sig={"p_up":np.full(len(idx),0.80),"p_down":np.full(len(idx),0.10),"edge":np.full(len(idx),0.70)}
    m=backtest(ds,idx,sig,0.70,0.30,0.55)
    for key in ("return_pct","benchmark_pct","alpha_pct","max_drawdown_pct","profit_factor","trades","entry_signals"):
        assert key in m
    assert math.isfinite(m["return_pct"])
    assert m["entry_signals"] > 0


def test_event_diagnostics_and_summary():
    ds=FeatureDataset(synthetic_candles(3500))
    idx=np.arange(500,2500)
    sig={"p_up":np.full(len(idx),0.80),"p_down":np.full(len(idx),0.10),"edge":np.full(len(idx),0.70)}
    e=event_diagnostics(ds,idx,sig,0.70,0.30)
    assert "signal_lift" in e
    fake=[]
    for n in range(3):
        fake.append({
            "eligible":True,
            "policy":{"policy_sufficient":True},
            "validation":{"return_pct":1+n*0.1,"alpha_pct":0.5,"profit_factor":1.3,"max_drawdown_pct":1.0,"sharpe_daily":1.0,"sortino_daily":1.0,"cvar95_daily_pct":-0.5,"trades":12},
            "event":{"signal_precision_pct":55.0,"base_up_rate_pct":45.0,"signal_lift":1.22}
        })
    s=summarize_folds(fake)
    assert s["candidate_walk_forward"] is True
