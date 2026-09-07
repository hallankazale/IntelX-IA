from __future__ import annotations
from pathlib import Path
import copy, threading, time, webbrowser
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core import (
    MarketData, PreparedMarket, SparseQModel, Trainer, evaluate,
    walk_forward_summary, walk_forward_score, dump_checkpoint, load_checkpoint,
    DATA_DAYS, HOLDOUT_GAP_DAYS,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
CHECKPOINT = DATA / "checkpoint_v06.json"
DASH = (BASE / "dashboard.html").read_text(encoding="utf-8")

app = FastAPI(title="EVO Trading AI PC v0.6")
lock = threading.RLock()
worker = None
stop_event = threading.Event()

state: Dict[str, Any] = {
    "version": "0.6.0", "running": False, "phase": "PARADO", "error": "",
    "started_at": 0, "ends_at": 0, "elapsed_sec": 0, "episodes": 0,
    "final_model_episodes": 0, "candles": 0,
    "split": {"development": 0, "test": 0},
    "folds": [], "actions": {"HOLD": 0, "BUY": 0, "SELL": 0},
    "walk_forward": None, "final_test": None, "reports": [],
    "best_score": -1e18, "best_final_q": None, "laboratory_ready": False,
    "data_days": DATA_DAYS, "holdout_gap_days": HOLDOUT_GAP_DAYS,
    "holdout_locked": True,
    "note": "Dados reais; ordens e dinheiro simulados. Holdout separado da v0.5."
}


def public_state():
    with lock:
        s = copy.deepcopy(state)
        s.pop("best_final_q", None)
        return s


def _serialize_model(model: SparseQModel) -> Dict[str, Any]:
    return {"q": model.export_q(), "visits": model.export_visits(), "episodes": model.episodes}


def save_state(models: List[SparseQModel] | None = None) -> None:
    with lock:
        payload = copy.deepcopy(state)
        if models is not None:
            payload["models"] = [_serialize_model(m) for m in models]
        dump_checkpoint(CHECKPOINT, payload)


def _restore_models(prior: Dict[str, Any] | None, seeds: List[int]) -> List[SparseQModel]:
    if prior and prior.get("version") == "0.6.0" and isinstance(prior.get("models"), list):
        items = prior["models"]
        if len(items) == len(seeds):
            out = []
            for item, seed in zip(items, seeds):
                out.append(SparseQModel(q=item.get("q"), visits=item.get("visits"),
                                        episodes=int(item.get("episodes", 0)), seed=seed))
            return out
    return [SparseQModel(seed=s) for s in seeds]


def _layout(n: int) -> Tuple[List[Tuple[int, int]], int, List[int]]:
    test_start = int(n * 0.85)
    train_ends = [int(n * 0.55), int(n * 0.65), int(n * 0.75)]
    val_ends = [int(n * 0.65), int(n * 0.75), test_start]
    folds = list(zip(train_ends, val_ends))
    return folds, test_start, train_ends


def run_training(hours: float):
    global state
    start_wall = time.time()
    stop_event.clear()
    models: List[SparseQModel] = []
    try:
        with lock:
            state.update({
                "running": True, "phase": "BAIXANDO 180 DIAS DE DADOS REAIS", "error": "",
                "started_at": start_wall, "ends_at": start_wall + hours * 3600,
                "elapsed_sec": 0, "final_test": None, "walk_forward": None, "reports": [],
                "actions": {"HOLD": 0, "BUY": 0, "SELL": 0}, "best_score": -1e18,
                "best_final_q": None, "laboratory_ready": False,
            })
        candles = MarketData(DATA).load_or_download(days=DATA_DAYS, gap_days=HOLDOUT_GAP_DAYS)
        market = PreparedMarket(candles)
        n = len(candles)
        folds, test_start, train_ends = _layout(n)
        prior = load_checkpoint(CHECKPOINT)
        seeds = [1001, 1002, 1003, 2001]
        models = _restore_models(prior, seeds)
        trainers = [Trainer(market, seed=20260910 + i) for i in range(4)]
        train_ranges = train_ends + [test_start]

        with lock:
            state["candles"] = n
            state["split"] = {"development": test_start, "test": n - test_start}
            state["folds"] = [
                {"train_end": train_ends[i], "validation_start": folds[i][0], "validation_end": folds[i][1]}
                for i in range(3)
            ]
            state["episodes"] = sum(m.episodes for m in models)
            state["final_model_episodes"] = models[3].episodes
            state["phase"] = "TREINANDO v0.6 · WALK-FORWARD"

        next_diag = time.time() + 120
        next_save = time.time() + 30
        target_period = 0.20
        model_index = 0

        while not stop_event.is_set() and time.time() < state["ends_at"]:
            t0 = time.time()
            idx = model_index % 4
            model_index += 1
            counts = trainers[idx].train_episode(models[idx], market.warmup, train_ranges[idx])
            with lock:
                state["episodes"] = sum(m.episodes for m in models)
                state["final_model_episodes"] = models[3].episodes
                for k, v in counts.items():
                    state["actions"][k] += v
                state["elapsed_sec"] = time.time() - start_wall

            now = time.time()
            if now >= next_diag:
                wf = walk_forward_summary(market, models[:3], folds)
                sc = walk_forward_score(wf)
                with lock:
                    state["walk_forward"] = wf
                    state["reports"].insert(0, {"timestamp": now, "episodes": state["episodes"], "walk_forward": wf})
                    state["reports"] = state["reports"][:30]
                    if sc > state["best_score"]:
                        state["best_score"] = sc
                        state["best_final_q"] = models[3].clone_q()
                save_state(models)
                next_diag = now + 120

            if now >= next_save:
                save_state(models)
                next_save = now + 30

            dt = time.time() - t0
            if dt < target_period:
                stop_event.wait(target_period - dt)

        wf = walk_forward_summary(market, models[:3], folds)
        current_score = walk_forward_score(wf)
        with lock:
            if state["best_final_q"] is None or current_score > state["best_score"]:
                state["best_score"] = current_score
                state["best_final_q"] = models[3].clone_q()
            final_q = copy.deepcopy(state["best_final_q"])

        completed = (not stop_event.is_set()) and time.time() >= state["ends_at"]
        reveal_holdout = completed and hours >= 6.0
        test = evaluate(market, final_q, test_start, n, seed=99) if reveal_holdout else None
        ready = False
        if test is not None:
            ready = (
                models[3].episodes >= 500
                and wf["positive_folds"] >= 2
                and wf["median_profit_factor"] >= 1.05
                and wf["median_return_pct"] > 0
                and wf["max_fold_drawdown_pct"] <= 10
                and test["trades"] >= 12
                and test["return_pct"] > 0
                and test["profit_factor"] >= 1.05
                and test["max_drawdown_pct"] <= 10
            )
        with lock:
            state["walk_forward"] = wf
            state["final_test"] = test
            state["holdout_locked"] = test is None
            state["running"] = False
            if stop_event.is_set():
                state["phase"] = "PARADO PELO USUÁRIO · HOLDOUT PRESERVADO"
            elif reveal_holdout:
                state["phase"] = "CONCLUÍDO · HOLDOUT AVALIADO"
            else:
                state["phase"] = "DESENVOLVIMENTO CONCLUÍDO · HOLDOUT PRESERVADO"
            state["elapsed_sec"] = time.time() - start_wall
            state["laboratory_ready"] = ready
        save_state(models)
    except Exception as exc:
        with lock:
            state["running"] = False
            state["phase"] = "ERRO"
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["elapsed_sec"] = time.time() - start_wall
        if models:
            save_state(models)


class StartBody(BaseModel):
    hours: float = 12.0


@app.get("/", response_class=HTMLResponse)
def home():
    return DASH


@app.get("/api/state")
def get_state():
    return public_state()


@app.post("/api/start")
def start(body: StartBody):
    global worker
    with lock:
        if state["running"]:
            return {"ok": False, "error": "Treino já está rodando."}
        hours = max(0.05, min(24.0, float(body.hours)))
        worker = threading.Thread(target=run_training, args=(hours,), daemon=True)
        worker.start()
    return {"ok": True, "hours": hours}


@app.post("/api/stop")
def stop():
    stop_event.set()
    return {"ok": True}


@app.post("/api/reset")
def reset():
    with lock:
        if state["running"]:
            return {"ok": False, "error": "Pare o treino antes."}
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
        state.update({
            "running": False, "phase": "PARADO", "error": "", "started_at": 0, "ends_at": 0,
            "elapsed_sec": 0, "episodes": 0, "final_model_episodes": 0,
            "actions": {"HOLD": 0, "BUY": 0, "SELL": 0}, "walk_forward": None,
            "final_test": None, "reports": [], "best_score": -1e18, "best_final_q": None,
            "laboratory_ready": False,
        })
    return {"ok": True}


def open_browser():
    time.sleep(1.3)
    webbrowser.open("http://127.0.0.1:8766")


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")
