"""Shared activity configuration for both halves."""
import json
from pathlib import Path
COMMON=json.loads(Path(__file__).with_name("common_config.json").read_text())
