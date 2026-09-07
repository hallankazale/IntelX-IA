import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "user_data" / "config_evo.json"
STRATEGY = ROOT / "user_data" / "strategies" / "EvoFreqAiStrategy.py"
EXPORTER = ROOT / "tools" / "export_diagnostics.py"


def test_config_is_dry_run_spot_and_small_stake():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["dry_run"] is True
    assert cfg["trading_mode"] == "spot"
    assert cfg["max_open_trades"] == 1
    assert float(cfg["stake_amount"]) <= 50
    assert cfg["exchange"]["key"] == ""
    assert cfg["exchange"]["secret"] == ""
    assert cfg["force_entry_enable"] is False


def test_strategy_syntax_and_long_only():
    source = STRATEGY.read_text(encoding="utf-8")
    ast.parse(source)
    assert "can_short = False" in source
    assert "round_trip_cost = 0.0024" in source
    assert "min_predicted_net_return = 0.0030" in source


def test_diagnostic_exporter_syntax():
    ast.parse(EXPORTER.read_text(encoding="utf-8"))


def test_diagnostic_exporter_runs_without_database():
    p = subprocess.run([sys.executable, str(EXPORTER)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    path = Path(p.stdout.strip().splitlines()[-1])
    assert path.exists()
    assert path.suffix == ".zip"
    path.unlink()
