import math
import numpy as np

from core import Candle, FeatureDataset, build_protocol, MAX_HORIZON, fit_models, predict_score, backtest


def synthetic_candles(n=2500):
    rows=[]
    px=100.0
    for i in range(n):
        drift=0.00008 + 0.0006*math.sin(i/37.0)
        px*=math.exp(drift)
        spread=px*0.002
        rows.append(Candle(i*300_000, px*(1-0.0003), px+spread, px-spread, px, 1000+100*math.sin(i/11.0)))
    return rows


def test_target_alignment_and_features_finite():
    ds=FeatureDataset(synthetic_candles(1200))
    i=200
    expected=math.log(ds.candles[i+6].close/ds.candles[i].close)
    assert abs(ds.targets[6][i]-expected) < 1e-12
    assert ds.X.shape[1] >= 15
    assert np.isfinite(ds.X).all()


def test_protocol_has_gaps_and_locked_holdout():
    p=build_protocol(52_000)
    assert p.dev_end < p.holdout_start
    assert p.holdout_end == 52_000
    assert p.holdout_start - p.dev_end == p.gap_bars
    assert len(p.folds) == 3
    for f in p.folds:
        assert f.val_start - f.train_end >= MAX_HORIZON
        assert f.val_end <= p.dev_end


def test_predictive_model_and_backtest_smoke():
    ds=FeatureDataset(synthetic_candles(2600))
    train=np.arange(100,1800)
    val=np.arange(1812,2400)
    models=fit_models(ds,train)
    scores=predict_score(models,ds.X[val])
    assert len(scores)==len(val)
    assert np.isfinite(scores).all()
    m=backtest(ds,val,scores,0.20,-0.10)
    for key in ("return_pct","benchmark_pct","alpha_pct","max_drawdown_pct","profit_factor","trades"):
        assert key in m
    assert math.isfinite(m["return_pct"])
