from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, json, math, random, time
from typing import List, Dict, Any, Tuple
import requests
import numpy as np

ACTION_HOLD=0
ACTION_BUY=1
ACTION_SELL=2
ACTION_NAMES={0:"HOLD",1:"BUY",2:"SELL"}
POSITION_FRACTION=0.25
FEE_RATE=0.0010
SLIPPAGE_RATE=0.0002
STOP_LOSS=0.03
TAKE_PROFIT=0.06
MAX_HOLD_BARS=48
STATE_COUNT=72
EPISODE_BARS=512

@dataclass
class Candle:
    open_time:int
    open:float
    high:float
    low:float
    close:float
    volume:float

class MarketData:
    def __init__(self, data_dir:Path):
        self.data_dir=data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache=self.data_dir/"btcusdt_5m.csv"

    def load_or_download(self, days:int=60, max_age_hours:int=6)->List[Candle]:
        if self.cache.exists() and time.time()-self.cache.stat().st_mtime < max_age_hours*3600:
            rows=self._load_csv()
            if len(rows)>1000:
                return rows
        rows=self._download(days)
        self._save_csv(rows)
        return rows

    def _load_csv(self)->List[Candle]:
        out=[]
        with self.cache.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.append(Candle(int(r["open_time"]),float(r["open"]),float(r["high"]),float(r["low"]),float(r["close"]),float(r["volume"])))
        return out

    def _save_csv(self, rows:List[Candle])->None:
        tmp=self.cache.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["open_time","open","high","low","close","volume"])
            for c in rows:
                w.writerow([c.open_time,c.open,c.high,c.low,c.close,c.volume])
        tmp.replace(self.cache)

    def _download(self, days:int)->List[Candle]:
        interval_ms=5*60*1000
        end=int(time.time()*1000)
        start=end-days*24*60*60*1000
        urls=[
            "https://api.binance.com/api/v3/klines",
            "https://data-api.binance.vision/api/v3/klines",
        ]
        out=[]
        cursor=start
        while cursor < end:
            params={"symbol":"BTCUSDT","interval":"5m","startTime":cursor,"endTime":end,"limit":1000}
            last_err=None
            data=None
            for url in urls:
                try:
                    r=requests.get(url, params=params, timeout=20)
                    r.raise_for_status()
                    data=r.json()
                    if isinstance(data,list):
                        break
                except Exception as e:
                    last_err=e
                    data=None
            if not data:
                if out:
                    break
                raise RuntimeError(f"Falha ao baixar candles da Binance: {last_err}")
            for x in data:
                out.append(Candle(int(x[0]),float(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5])))
            nxt=int(data[-1][0])+interval_ms
            if nxt<=cursor:
                break
            cursor=nxt
            time.sleep(0.05)
        dedup={c.open_time:c for c in out}
        rows=[dedup[k] for k in sorted(dedup)]
        if len(rows)<2000:
            raise RuntimeError(f"Poucos candles baixados: {len(rows)}")
        return rows

def ema(v:np.ndarray,p:int)->np.ndarray:
    out=np.empty_like(v,dtype=float)
    out[0]=v[0]
    a=2/(p+1)
    for i in range(1,len(v)):
        out[i]=a*v[i]+(1-a)*out[i-1]
    return out

def sma(v:np.ndarray,p:int)->np.ndarray:
    out=np.empty_like(v,dtype=float)
    s=0.0
    for i,x in enumerate(v):
        s+=x
        if i>=p:s-=v[i-p]
        out[i]=s/max(1,min(p,i+1))
    return out

def rsi(close:np.ndarray,p:int=14)->np.ndarray:
    out=np.full(len(close),50.0)
    g=l=0.0
    for i in range(1,len(close)):
        ch=close[i]-close[i-1]
        gain=max(0.0,ch); loss=max(0.0,-ch)
        if i<=p:
            g+=gain;l+=loss
            if i==p:g/=p;l/=p
        else:
            g=((g*(p-1))+gain)/p
            l=((l*(p-1))+loss)/p
        if i>=p:
            out[i]=100.0 if l<1e-12 else 100-100/(1+g/l)
    return out

def atr(high:np.ndarray,low:np.ndarray,close:np.ndarray,p:int=14)->np.ndarray:
    tr=np.empty(len(close),dtype=float)
    tr[0]=high[0]-low[0]
    for i in range(1,len(close)):
        tr[i]=max(high[i]-low[i],abs(high[i]-close[i-1]),abs(low[i]-close[i-1]))
    out=np.empty(len(close),dtype=float)
    avg=tr[0];out[0]=avg
    for i in range(1,len(tr)):
        avg=((avg*i)+tr[i])/(i+1) if i<p else ((avg*(p-1))+tr[i])/p
        out[i]=avg
    return out

class PreparedMarket:
    def __init__(self,candles:List[Candle]):
        self.candles=candles
        self.warmup=30
        close=np.array([c.close for c in candles],float)
        high=np.array([c.high for c in candles],float)
        low=np.array([c.low for c in candles],float)
        vol=np.array([c.volume for c in candles],float)
        e9,e21=ema(close,9),ema(close,21)
        rr=rsi(close,14); aa=atr(high,low,close,14); vs=sma(vol,20)
        base=np.zeros(len(candles),dtype=np.int16)
        for i in range(len(candles)):
            px=max(close[i],1e-9)
            td=(e9[i]-e21[i])/px
            trend=2 if td>0.001 else (0 if td<-0.001 else 1)
            mom=2 if rr[i]>58 else (0 if rr[i]<42 else 1)
            vola=1 if aa[i]/px>0.006 else 0
            hv=1 if vol[i] > max(vs[i],1e-9)*1.20 else 0
            base[i]=(((trend*3+mom)*2+vola)*2+hv)
        self.base_states=base

    def state_at(self,i:int,in_position:bool)->int:
        return int(self.base_states[i])*2+(1 if in_position else 0)

class Portfolio:
    def __init__(self,cash:float=1000.0):
        self.cash=cash; self.asset=0.0; self.entry=0.0; self.cost_basis=0.0
        self.bars_held=0; self.trades=0; self.wins=0; self.gross_profit=0.0; self.gross_loss=0.0

    def in_position(self)->bool:
        return self.asset>1e-12
    def equity(self,px:float)->float:
        return self.cash+self.asset*px

    def buy(self,px:float)->None:
        if self.in_position(): return
        alloc=self.cash*POSITION_FRACTION
        if alloc<1e-9:return
        exe=px*(1+SLIPPAGE_RATE)
        fee=alloc*FEE_RATE
        net=alloc-fee
        self.asset=net/exe; self.cash-=alloc; self.entry=exe; self.cost_basis=alloc; self.bars_held=0

    def sell(self,px:float)->None:
        if not self.in_position(): return
        exe=px*(1-SLIPPAGE_RATE)
        gross=self.asset*exe; fee=gross*FEE_RATE; net=gross-fee
        pnl=net-self.cost_basis
        self.cash+=net; self.asset=0; self.entry=0; self.cost_basis=0; self.bars_held=0; self.trades+=1
        if pnl>=0:self.wins+=1;self.gross_profit+=pnl
        else:self.gross_loss+=-pnl

    def apply(self,action:int,px:float)->None:
        if action==ACTION_BUY:self.buy(px)
        elif action==ACTION_SELL:self.sell(px)

    def risk(self,next_c:Candle)->None:
        if not self.in_position():return
        self.bars_held+=1
        stop=self.entry*(1-STOP_LOSS); take=self.entry*(1+TAKE_PROFIT)
        if next_c.low<=stop:self.sell(stop)
        elif next_c.high>=take:self.sell(take)
        elif self.bars_held>=MAX_HOLD_BARS:self.sell(next_c.close)

class QModel:
    def __init__(self,q=None,episodes:int=0,seed:int=1337):
        self.q=np.zeros((STATE_COUNT,3),dtype=float) if q is None else np.array(q,dtype=float,copy=True)
        self.episodes=episodes
        self.rng=random.Random(seed+episodes)

    def epsilon(self)->float:
        return max(0.04,0.35*math.exp(-self.episodes/2500))
    @staticmethod
    def valid_actions(in_position:bool)->Tuple[int,int]:
        return (ACTION_HOLD,ACTION_SELL) if in_position else (ACTION_HOLD,ACTION_BUY)

    def greedy(self,state:int,in_position:bool)->int:
        valid=self.valid_actions(in_position)
        vals=[self.q[state,a] for a in valid]
        mx=max(vals)
        best=[a for a in valid if abs(self.q[state,a]-mx)<1e-12]
        return self.rng.choice(best)

    def choose(self,state:int,in_position:bool,explore:bool=True)->int:
        valid=self.valid_actions(in_position)
        if explore and self.rng.random()<self.epsilon():
            return self.rng.choice(valid)
        return self.greedy(state,in_position)

    def update(self,state:int,action:int,reward:float,next_state:int,next_in_position:bool)->None:
        alpha=max(0.02,0.11/math.sqrt(1+self.episodes/1600))
        gamma=0.97
        next_best=max(self.q[next_state,a] for a in self.valid_actions(next_in_position))
        self.q[state,action]+=alpha*(reward+gamma*next_best-self.q[state,action])

class Trainer:
    def __init__(self,market:PreparedMarket,seed:int=1337):
        self.market=market
        self.rng=random.Random(seed)

    def train_episode(self,model:QModel,start:int,end_exclusive:int)->Dict[str,int]:
        min_start=max(self.market.warmup,start)
        latest=end_exclusive-EPISODE_BARS-2
        if latest<=min_start:raise ValueError("Faixa de treino insuficiente")
        i0=self.rng.randint(min_start,latest); end=min(end_exclusive-1,i0+EPISODE_BARS)
        p=Portfolio(1000); peak=1000; counts={"HOLD":0,"BUY":0,"SELL":0}
        for i in range(i0,end):
            cur=self.market.candles[i]; nxt=self.market.candles[i+1]
            before=max(1e-9,p.equity(cur.close))
            inpos=p.in_position(); state=self.market.state_at(i,inpos)
            action=model.choose(state,inpos,True)
            counts[ACTION_NAMES[action]]+=1
            p.apply(action,cur.close); p.risk(nxt)
            after=max(1e-9,p.equity(nxt.close))
            peak=max(peak,after)
            dd=max(0.0,(peak-after)/peak)
            agent_ret=100*math.log(after/before)
            benchmark_ret=POSITION_FRACTION*100*math.log(max(1e-9,nxt.close)/max(1e-9,cur.close))
            reward=(agent_ret-benchmark_ret)-0.08*dd
            nxt_in=p.in_position(); nxt_state=self.market.state_at(i+1,nxt_in)
            model.update(state,action,reward,nxt_state,nxt_in)
        p.sell(self.market.candles[end].close)
        model.episodes+=1
        return counts

def sharpe(xs:List[float])->float:
    if len(xs)<10:return 0.0
    m=sum(xs)/len(xs)
    var=sum((x-m)**2 for x in xs)/max(1,len(xs)-1)
    sd=math.sqrt(var)
    return 0.0 if sd<1e-12 else (m/sd)*math.sqrt(12*24*365)

def evaluate(market:PreparedMarket,q,start:int,end_exclusive:int,seed:int=7)->Dict[str,Any]:
    model=QModel(q=q,episodes=999999,seed=seed)
    p=Portfolio(1000)
    first=max(market.warmup,start); last=min(end_exclusive-1,len(market.candles)-2)
    if last<=first+5:return {"return_pct":0,"benchmark_pct":0,"alpha_pct":0,"max_drawdown_pct":0,"profit_factor":0,"win_rate_pct":0,"sharpe":0,"trades":0}
    peak=1000; maxdd=0.0; prev=1000; rets=[]
    counts={"HOLD":0,"BUY":0,"SELL":0}
    for i in range(first,last+1):
        cur=market.candles[i]; nxt=market.candles[i+1]
        inpos=p.in_position(); state=market.state_at(i,inpos)
        action=model.greedy(state,inpos)
        counts[ACTION_NAMES[action]]+=1
        p.apply(action,cur.close); p.risk(nxt)
        eq=p.equity(nxt.close); peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak)
        rets.append(eq/prev-1); prev=eq
    final_px=market.candles[last+1].close
    p.sell(final_px); final_eq=p.equity(final_px)
    ret=(final_eq/1000-1)*100
    bh=(1-POSITION_FRACTION)*1000 + POSITION_FRACTION*1000*(final_px/market.candles[first].close)
    benchmark=(bh/1000-1)*100
    pf=99.0 if p.gross_loss<1e-9 and p.gross_profit>0 else (0.0 if p.gross_loss<1e-9 else p.gross_profit/p.gross_loss)
    wr=0.0 if p.trades==0 else p.wins*100.0/p.trades
    return {
        "return_pct":ret,
        "benchmark_pct":benchmark,
        "alpha_pct":ret-benchmark,
        "max_drawdown_pct":maxdd*100,
        "profit_factor":pf,
        "win_rate_pct":wr,
        "sharpe":sharpe(rets),
        "trades":p.trades,
        "actions":counts,
        "final_equity":final_eq,
    }

def dump_checkpoint(path:Path,payload:Dict[str,Any])->None:
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)

def load_checkpoint(path:Path)->Dict[str,Any]|None:
    if not path.exists():return None
    try:return json.loads(path.read_text(encoding="utf-8"))
    except Exception:return None
