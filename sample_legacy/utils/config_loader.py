"""The homegrown config loader every module imports (predates dotenv)."""
import json
import os

_DEFAULTS = {
    "currency": "USD",
    "tax_table": {"CA": 725, "NY": 880, "TX": 625, "EU": 2000},
    "warehouse_url": "tcp://wh01:9911",
    "ledger_spool": r"\\LEDGER01\spool\orders.dat",
}


def load_config() -> dict:
    """Merge defaults with an optional orderflow.json next to the cwd.
    Environment variables win over both (ORDERFLOW_CURRENCY etc.)."""
    config = dict(_DEFAULTS)
    if os.path.exists("orderflow.json"):
        with open("orderflow.json", encoding="utf-8") as fh:
            config.update(json.load(fh))
    for key in list(config):
        env = os.environ.get(f"ORDERFLOW_{key.upper()}")
        if env:
            config[key] = env
    return config
