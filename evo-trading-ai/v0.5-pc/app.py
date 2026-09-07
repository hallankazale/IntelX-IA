from __future__ import annotations
from pathlib import Path
import copy, threading, time, webbrowser
from typing import Any, Dict
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from core import MarketData, PreparedMarket, QModel, Trainer, evaluate, dump_checkpoint, load_checkpoint

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"; DATA.mkdir(exist_ok=True)
CHECKPOINT=DATA/"checkpoint.json"
DASH=(BASE/"dashboard.html").read_text(encoding="utf-8")

app=FastAPI(title="EVO Trading AI PC v0.5")
lock=threading.RLock()
worker=None
stop_event=threading.Event()

state:Dict[str,Any]={
    "version":"0.5.0","running":False,"phase":"PARADO","error":"","started_at":0,"ends_at":0,
    "elapsed_sec":0,"episodes":0,"candles":0,"split":{"train":0,"validation":0,"test":0},
    "actions":{"HOLD":0,"BUY":0,"SELL":0},"latest_validation":None,"final_test":None,"reports":[],
    "continuity_pct":100.0,"best_score":-1e18,"best_q":None,
    "note":"Dados reais; ordens e dinheiro simulados."
}

def public_state():
    with lock:
        s=copy.deepcopy(state); s.pop("best_q",None); return s

def save_state(model:QModel|None=None):
    with lock:
        payload=copy.deepcopy(state)
        if model is not None: payload["q"]=model.q.tolist()
        if payload.get("best_q") is not None and isinstance(payload["best_q"],np.ndarray):
            payload["best_q"]=payload["best_q"].tolist()
        dump_checkpoint(CHECKPOINT,payload)

def score_metrics(m:Dict[str,Any])->float:
    return m["return_pct"]+0.7*m["alpha_pct"]-0.8*m["max_drawdown_pct"]+min(3.0,m["profit_factor"])*1.5+min(2.5,max(-2.5,m["sharpe"]))*0.5

def run_training(hours:float):
    global state
    start_wall=time.time(); stop_event.clear(); model=None
    try:
        with lock:
            state.update({"running":True,"phase":"BAIXANDO DADOS REAIS","error":"","started_at":start_wall,
                "ends_at":start_wall+hours*3600,"elapsed_sec":0,"final_test":None,"reports":[],
                "actions":{"HOLD":0,"BUY":0,"SELL":0},"latest_validation":None,"best_score":-1e18,"best_q":None})
        candles=MarketData(DATA).load_or_download(days=60)
        market=PreparedMarket(candles); n=len(candles); train_end=int(n*0.70); val_end=int(n*0.85)
        prior=load_checkpoint(CHECKPOINT)
        if prior and prior.get("version")=="0.5.0" and isinstance(prior.get("q"),list):
            model=QModel(q=prior["q"],episodes=int(prior.get("episodes",0)),seed=1337)
        else: model=QModel(seed=1337)
        trainer=Trainer(market,seed=20260907)
        with lock:
            state["candles"]=n; state["split"]={"train":train_end,"validation":val_end-train_end,"test":n-val_end}
            state["episodes"]=model.episodes; state["phase"]="TREINANDO v0.5"
        next_diag=time.time()+3600; next_save=time.time()+30; target_period=0.5
        while not stop_event.is_set() and time.time()<state["ends_at"]:
            t0=time.time(); counts=trainer.train_episode(model,market.warmup,train_end)
            with lock:
                state["episodes"]=model.episodes
                for k,v in counts.items(): state["actions"][k]+=v
                state["elapsed_sec"]=time.time()-start_wall
            now=time.time()
            if now>=next_diag:
                v=evaluate(market,model.q,train_end,val_end); sc=score_metrics(v)
                with lock:
                    state["latest_validation"]=v; state["reports"].insert(0,{"timestamp":now,"episodes":model.episodes,"validation":v}); state["reports"]=state["reports"][:24]
                    if v["trades"]>=10 and sc>state["best_score"]: state["best_score"]=sc; state["best_q"]=model.q.copy()
                save_state(model); next_diag=now+3600
            if now>=next_save: save_state(model); next_save=now+30
            dt=time.time()-t0
            if dt<target_period: stop_event.wait(target_period-dt)
        final_q=model.q
        with lock:
            if state["best_q"] is not None: final_q=np.array(state["best_q"],dtype=float)
        v=evaluate(market,final_q,train_end,val_end); t=evaluate(market,final_q,val_end,n)
        lab_ready=(model.episodes>=500 and v["trades"]>=20 and t["trades"]>=10 and v["profit_factor"]>=1.05 and t["profit_factor"]>=1.02 and v["max_drawdown_pct"]<=12 and t["max_drawdown_pct"]<=12 and t["return_pct"]>0)
        with lock:
            state["latest_validation"]=v; state["final_test"]=t; state["running"]=False
            state["phase"]="CONCLUÍDO" if not stop_event.is_set() else "PARADO PELO USUÁRIO"
            state["elapsed_sec"]=time.time()-start_wall; state["laboratory_ready"]=lab_ready
        save_state(model)
    except Exception as e:
        with lock:
            state["running"]=False; state["phase"]="ERRO"; state["error"]=f"{type(e).__name__}: {e}"; state["elapsed_sec"]=time.time()-start_wall
        if model is not None: save_state(model)

class StartBody(BaseModel): hours:float=12.0

@app.get("/",response_class=HTMLResponse)
def home(): return DASH

@app.get("/api/state")
def get_state(): return public_state()

@app.post("/api/start")
def start(body:StartBody):
    global worker
    with lock:
        if state["running"]: return {"ok":False,"error":"Treino já está rodando."}
        hours=max(0.05,min(24.0,float(body.hours))); worker=threading.Thread(target=run_training,args=(hours,),daemon=True); worker.start()
    return {"ok":True,"hours":hours}

@app.post("/api/stop")
def stop(): stop_event.set(); return {"ok":True}

@app.post("/api/reset")
def reset():
    with lock:
        if state["running"]: return {"ok":False,"error":"Pare o treino antes."}
        if CHECKPOINT.exists(): CHECKPOINT.unlink()
        state.update({"running":False,"phase":"PARADO","error":"","started_at":0,"ends_at":0,"elapsed_sec":0,"episodes":0,
            "actions":{"HOLD":0,"BUY":0,"SELL":0},"latest_validation":None,"final_test":None,"reports":[],"best_score":-1e18,"best_q":None,"laboratory_ready":False})
    return {"ok":True}

def open_browser(): time.sleep(1.3); webbrowser.open("http://127.0.0.1:8765")

if __name__=="__main__":
    import uvicorn
    threading.Thread(target=open_browser,daemon=True).start()
    uvicorn.run(app,host="127.0.0.1",port=8765,log_level="warning")
