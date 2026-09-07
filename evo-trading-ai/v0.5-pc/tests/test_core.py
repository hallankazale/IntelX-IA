import unittest, random, math
from core import Candle, PreparedMarket, QModel, Trainer, evaluate, ACTION_BUY, ACTION_SELL

def synthetic(n=5000):
    out=[]; px=100.0; rng=random.Random(42)
    for i in range(n):
        drift=0.0005 if (i//700)%2==0 else -0.00035
        px*=math.exp(drift+rng.gauss(0,0.003))
        hi=px*(1+abs(rng.gauss(0,0.002))); lo=px*(1-abs(rng.gauss(0,0.002)))
        out.append(Candle(i*300000,px,hi,lo,px,100+rng.random()*20))
    return out

class CoreTests(unittest.TestCase):
    def test_action_masking(self):
        m=QModel(seed=1)
        for _ in range(100):
            self.assertIn(m.choose(0,False,True),(0,ACTION_BUY))
            self.assertIn(m.choose(1,True,True),(0,ACTION_SELL))
    def test_training_explores_buy_sell(self):
        market=PreparedMarket(synthetic()); model=QModel(seed=4); t=Trainer(market,seed=5)
        total={"HOLD":0,"BUY":0,"SELL":0}
        for _ in range(80):
            c=t.train_episode(model,market.warmup,int(len(market.candles)*0.7))
            for k,v in c.items():total[k]+=v
        self.assertGreater(total["BUY"],0); self.assertGreater(total["SELL"],0)
    def test_evaluation_returns_metrics(self):
        market=PreparedMarket(synthetic()); model=QModel(seed=9); t=Trainer(market,seed=10); end=int(len(market.candles)*0.7)
        for _ in range(30):t.train_episode(model,market.warmup,end)
        m=evaluate(market,model.q,end,int(len(market.candles)*0.85))
        self.assertIn("alpha_pct",m); self.assertIn("trades",m)

if __name__=="__main__": unittest.main()
