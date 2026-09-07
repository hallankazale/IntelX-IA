from __future__ import annotations

from pathlib import Path
import copy
import json
import threading
import time
import webbrowser
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from core import (
    MarketData, FeatureDataset, build_protocol, run_fold, summarize_folds,
    FEATURE_NAMES, MIN_ENTRY_EDGE_PCT, ROUND_TRIP_COST_PCT,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"; DATA.mkdir(exist_ok=True)
RESULTS = DATA / "v071_results.json"
DASH = (BASE / "dashboard.html").read_text(encoding="utf-8")

app = FastAPI(title="EVO Trading AI PC v0.7.1")
lock = threading.RLock(); stop_event = threading.Event(); worker: threading.Thread | None = None
state: Dict[str, Any] = {
    "version": "0.7.1", "running": False, "phase": "PARADO", "error": "",
    "started_at": 0.0, "elapsed_sec": 0.0, "progress_pct": 0.0, "candles": 0,
    "feature_count": len(FEATURE_NAMES), "feature_names": list(FEATURE_NAMES),
    "model": "HistGradientBoostingRegressor · calibração por quantis · 3/6/12 candles",
    "round_trip_cost_pct": ROUND_TRIP_COST_PCT, "min_entry_edge_pct": MIN_ENTRY_EDGE_PCT,
    "fold_results": [], "summary": None, "protocol": None, "holdout_touched": False,
    "note": "Preços históricos reais. Ordens, saldo e lucro simulados.",
}


def public_state() -> Dict[str, Any]:
    with lock: return copy.deepcopy(state)


def save_results() -> None:
    with lock:
        payload = copy.deepcopy(state); payload["running"] = False
    tmp = RESULTS.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(RESULTS)


def run_lab() -> None:
    start = time.time(); stop_event.clear()
    try:
        with lock:
            state.update({"running": True, "phase": "CARREGANDO 180 DIAS DE DADOS REAIS", "error": "",
                          "started_at": start, "elapsed_sec": 0.0, "progress_pct": 2.0, "candles": 0,
                          "fold_results": [], "summary": None, "protocol": None, "holdout_touched": False})
        candles = MarketData(DATA).load_or_download(days=180)
        if stop_event.is_set(): return
        ds = FeatureDataset(candles); protocol = build_protocol(len(candles))
        with lock:
            state["candles"] = len(candles)
            state["protocol"] = {
                "development_start": protocol.dev_start, "development_end": protocol.dev_end,
                "development_samples": protocol.dev_end - protocol.dev_start,
                "gap_bars": protocol.gap_bars, "gap_days": protocol.gap_bars / 288.0,
                "holdout_start": protocol.holdout_start, "holdout_end": protocol.holdout_end,
                "holdout_samples": protocol.holdout_end - protocol.holdout_start,
                "folds": [{"name": f.name, "train_start": f.train_start, "train_end": f.train_end,
                           "val_start": f.val_start, "val_end": f.val_end,
                           "embargo_days": (f.val_start - f.train_end) / 288.0} for f in protocol.folds],
            }
            state["phase"] = "TREINANDO + CALIBRANDO SEM TOCAR NO HOLDOUT"; state["progress_pct"] = 10.0
        results = []
        for idx, fold in enumerate(protocol.folds):
            if stop_event.is_set():
                with lock:
                    state["phase"] = "PARADO PELO USUÁRIO"; state["running"] = False; state["elapsed_sec"] = time.time() - start
                save_results(); return
            with lock:
                state["phase"] = f"{fold.name.upper()} · MODELO + QUANTIS + VALIDAÇÃO"
                state["progress_pct"] = 10.0 + idx * 28.0
            result = run_fold(ds, fold); results.append(result)
            with lock:
                state["fold_results"] = copy.deepcopy(results); state["progress_pct"] = 10.0 + (idx + 1) * 28.0
                state["elapsed_sec"] = time.time() - start
        summary = summarize_folds(results)
        with lock:
            state["summary"] = summary; state["progress_pct"] = 100.0
            state["phase"] = "CONCLUÍDO · HOLDOUT FINAL PRESERVADO"; state["running"] = False
            state["elapsed_sec"] = time.time() - start; state["holdout_touched"] = False
        save_results()
    except Exception as e:
        with lock:
            state["running"] = False; state["phase"] = "ERRO"; state["error"] = f"{type(e).__name__}: {e}"
            state["elapsed_sec"] = time.time() - start
        try: save_results()
        except Exception: pass


@app.get("/", response_class=HTMLResponse)
def home() -> str: return DASH

@app.get("/api/state")
def get_state() -> Dict[str, Any]: return public_state()

@app.post("/api/start")
def start_lab() -> Dict[str, Any]:
    global worker
    with lock:
        if state["running"]: return {"ok": False, "error": "Avaliação já está rodando."}
        worker = threading.Thread(target=run_lab, daemon=True); worker.start()
    return {"ok": True}

@app.post("/api/stop")
def stop_lab() -> Dict[str, Any]: stop_event.set(); return {"ok": True}

@app.post("/api/reset")
def reset_lab() -> Dict[str, Any]:
    with lock:
        if state["running"]: return {"ok": False, "error": "Pare a avaliação antes."}
        if RESULTS.exists(): RESULTS.unlink()
        state.update({"running": False, "phase": "PARADO", "error": "", "started_at": 0.0, "elapsed_sec": 0.0,
                      "progress_pct": 0.0, "candles": 0, "fold_results": [], "summary": None,
                      "protocol": None, "holdout_touched": False})
    return {"ok": True}


def open_browser() -> None:
    time.sleep(1.2); webbrowser.open("http://127.0.0.1:8768")

if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8768, log_level="warning")
