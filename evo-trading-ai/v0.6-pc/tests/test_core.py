from pathlib import Path
import sys, math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import Candle, PreparedMarket, Portfolio, SparseQModel, Trainer, evaluate, benchmark_with_costs, higher_timeframe_trend
import numpy as np


def candles(n=7000):
    out=[]
    px=30000.0
    for i in range(n):
        drift=0.00015*math.sin(i/80.0)+0.00005
        px*=1+drift
        out.append(Candle(i*300000, px*0.999, px*1.002, px*0.998, px, 100+i%17))
    return out


def test_multitimeframe_no_future_shape():
    close=np.arange(1,301,dtype=float)
    t=higher_timeframe_trend(close,3,6,14)
    assert len(t)==len(close)
    assert int(t[0])==1


def test_valid_actions_and_forced_exploration():
    m=SparseQModel(seed=1)
    state=123
    a1=m.choose(state,False,True)
    a2=m.choose(state,False,True)
    assert {a1,a2}=={0,1}
    p=Portfolio()
    assert m.valid_actions(p.in_position())==(0,1)
    p.buy(100)
    assert m.valid_actions(p.in_position())==(0,2)


def test_train_and_evaluate_runs():
    market=PreparedMarket(candles())
    model=SparseQModel(seed=2)
    trainer=Trainer(market,seed=3)
    for _ in range(12):
        trainer.train_episode(model,market.warmup,5000)
    r=evaluate(market,model.export_q(),5000,6500)
    assert r['trades']>=0
    assert 'sortino' in r and 'cvar95_daily_pct' in r
    assert len(model.q)>0


def test_benchmark_includes_costs():
    flat=benchmark_with_costs(100.0,100.0)
    assert flat<0
