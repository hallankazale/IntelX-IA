from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
USER_DATA = ROOT / "user_data"
DIAG_DIR = ROOT / "diagnostics"
CONFIG = USER_DATA / "config_evo.json"
DB = USER_DATA / "tradesv3.dryrun.sqlite"
LOG = USER_DATA / "logs" / "freqtrade.log"

SENSITIVE_KEYS = {
    "key", "secret", "password", "token", "ws_token", "jwt_secret_key",
    "api_key", "api_secret", "passphrase",
}


def redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_KEYS:
                out[k] = "***REDACTED***" if v else ""
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, list):
        return [redact(x) for x in obj]
    return obj


def sanitize_text(text: str) -> str:
    patterns = [
        r'(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+',
        r'(?i)(secret\s*[=:]\s*)[^\s,;]+',
        r'(?i)(password\s*[=:]\s*)[^\s,;]+',
        r'(?i)(token\s*[=:]\s*)[^\s,;]+',
    ]
    for pat in patterns:
        text = re.sub(pat, r'\1***REDACTED***', text)
    return text


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        "branch": run("git", "branch", "--show-current"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def docker_info() -> dict[str, Any]:
    try:
        out = subprocess.check_output(
            ["docker", "compose", "ps", "--format", "json"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        )
        return {"compose_ps": out.strip()}
    except Exception as e:
        return {"compose_ps_error": str(e)}


def export_trades(db_path: Path, target_csv: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"database_present": db_path.exists(), "trades": 0}
    if not db_path.exists():
        return summary
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        summary["tables"] = sorted(tables)
        if "trades" not in tables:
            return summary
        cols = [r[1] for r in con.execute("PRAGMA table_info(trades)")]
        rows = con.execute("SELECT * FROM trades ORDER BY id").fetchall()
        summary["trades"] = len(rows)
        with target_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)

        col_idx = {name: i for i, name in enumerate(cols)}
        closed_col = "is_open" if "is_open" in col_idx else None
        profit_abs_col = "close_profit_abs" if "close_profit_abs" in col_idx else None
        profit_col = "close_profit" if "close_profit" in col_idx else None
        closed_rows = rows
        if closed_col:
            closed_rows = [r for r in rows if not bool(r[col_idx[closed_col]])]
        summary["closed_trades"] = len(closed_rows)
        if profit_abs_col:
            vals = [r[col_idx[profit_abs_col]] for r in closed_rows if r[col_idx[profit_abs_col]] is not None]
            summary["closed_profit_abs_sum"] = float(sum(vals)) if vals else 0.0
        if profit_col:
            vals = [r[col_idx[profit_col]] for r in closed_rows if r[col_idx[profit_col]] is not None]
            summary["closed_profit_ratio_sum"] = float(sum(vals)) if vals else 0.0
    finally:
        con.close()
    return summary


def copy_sanitized_log(src: Path, dst: Path, max_bytes: int = 20 * 1024 * 1024) -> dict[str, Any]:
    if not src.exists():
        return {"present": False}
    size = src.stat().st_size
    with src.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
        raw = f.read().decode("utf-8", errors="replace")
    dst.write_text(sanitize_text(raw), encoding="utf-8")
    return {"present": True, "source_size": size, "exported_bytes": dst.stat().st_size, "tail_only": size > max_bytes}


def model_inventory() -> list[dict[str, Any]]:
    root = USER_DATA / "models"
    if not root.exists():
        return []
    items = []
    for p in root.rglob("*"):
        if p.is_file():
            items.append({
                "path": str(p.relative_to(USER_DATA)),
                "size": p.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
            })
    return sorted(items, key=lambda x: x["mtime_utc"], reverse=True)[:500]


def latest_backtest_files(limit: int = 10) -> list[Path]:
    root = USER_DATA / "backtest_results"
    if not root.exists():
        return []
    files = [p for p in root.rglob("*") if p.is_file() and p.stat().st_size <= 10 * 1024 * 1024]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def main() -> int:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_zip = DIAG_DIR / f"EVO-diagnostico-{stamp}.zip"

    with tempfile.TemporaryDirectory(prefix="evo_diag_") as tmp_name:
        tmp = Path(tmp_name)

        config_obj = {}
        if CONFIG.exists():
            config_obj = json.loads(CONFIG.read_text(encoding="utf-8"))
            (tmp / "config_sanitized.json").write_text(
                json.dumps(redact(config_obj), ensure_ascii=False, indent=2), encoding="utf-8"
            )

        trades_summary = export_trades(DB, tmp / "trades.csv")
        log_summary = copy_sanitized_log(LOG, tmp / "evo.log")
        models = model_inventory()
        (tmp / "models_inventory.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")

        backtest_dir = tmp / "backtest_results"
        backtest_dir.mkdir(exist_ok=True)
        copied_backtests = []
        for p in latest_backtest_files():
            target = backtest_dir / p.name
            target.write_bytes(p.read_bytes())
            copied_backtests.append(p.name)

        manifest = {
            "format": "evo-diagnostic-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evo_version": "1.0.0",
            "mode": "dry-run" if config_obj.get("dry_run") is True else "UNKNOWN_OR_NOT_DRY_RUN",
            "platform": platform.platform(),
            "python": sys.version,
            "git": git_info(),
            "docker": docker_info(),
            "config_sha256": sha256(CONFIG),
            "database_sha256": sha256(DB),
            "log": log_summary,
            "trade_summary": trades_summary,
            "model_files": len(models),
            "backtest_files": copied_backtests,
            "safety": {
                "secrets_redacted": True,
                "raw_exchange_keys_included": False,
            },
        }
        (tmp / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (tmp / "summary.json").write_text(json.dumps(trades_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in tmp.rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(tmp))

    print(str(out_zip.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
