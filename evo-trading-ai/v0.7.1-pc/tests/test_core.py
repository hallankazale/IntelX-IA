import math
import numpy as np

from core import (
    Candle, FeatureDataset, build_protocol, OUTER_GAP_BARS, fit_models,
    predict_score, backtest, select_thresholds, MIN_ENTRY_EDGE_PCT,
)


def synthetic_candles(n=2600):
    rows=[]; px=100.0
    for i in range(n):
        drift=0.00010 + 0.0007*math.sin(i/31.0)
        px*=math.exp(drift); spread=px*0.002
        rows.append(Candle(i*300_000, px*(1-0.0003), px+spread, px-spread, px, 1000+120*math.sin(i/9.0)))
    return rows


def test_features_and_target_alignment():
    ds=FeatureDataset(synthetic_candles(1400)); i=250
    assert ds.X.shape[1] == 19 and np.isfinite(ds.X).all()
    assert abs(ds.targets[6][i]-math.log(ds.candles[i+6].close/ds.candles[i].close)) < 1e-12


def test_protocol_uses_real_outer_embargo_and_locked_holdout():
    p=build_protocol(52_000)
    assert p.dev_end < p.holdout_start
    assert p.holdout_start-p.dev_end == p.gap_bars
    assert len(p.folds)==3
    for fold in p.folds:
        assert fold.val_start-fold.train_end == p.gap_bars
        assert fold.val_start-fold.train_end >= min(OUTER_GAP_BARS, int(52_000*0.08))
        assert fold.val_end <= p.dev_end


def test_predictive_model_smoke():
    ds=FeatureDataset(synthetic_candles(2800))
    train=np.arange(100,1800); val=np.arange(1900,2500)
    models=fit_models(ds,train); scores=predict_score(models,ds.X[val])
    assert np.isfinite(scores).all()
    m=backtest(ds,val,scores,MIN_ENTRY_EDGE_PCT,0.0)
    assert math.isfinite(m['return_pct']) and 'entry_signals' in m


def test_quantile_calibration_never_goes_below_cost_floor():
    ds=FeatureDataset(synthetic_candles(2800)); idx=np.arange(1800,2500)
    scores=np.linspace(-0.004,0.008,len(idx)) + 0.001*np.sin(np.arange(len(idx))/5)
    c=select_thresholds(ds,idx,scores)
    assert c['entry_pct'] >= MIN_ENTRY_EDGE_PCT-1e-9
    assert 'distribution' in c and 'p95_pct' in c['distribution']
    assert c['tested_configs'] > 0


def test_weak_predictions_report_signal_insufficient_not_fixed_fallback():
    ds=FeatureDataset(synthetic_candles(2800)); idx=np.arange(1800,2500)
    scores=np.zeros(len(idx))+0.0001
    c=select_thresholds(ds,idx,scores)
    assert c['signal_sufficient'] is False
    assert c['entry_pct'] == MIN_ENTRY_EDGE_PCT
    assert 'mínimo de operações' in c['reason']
