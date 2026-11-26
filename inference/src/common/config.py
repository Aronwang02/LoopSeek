"""YAML config with `a.b.c=value` command-line overrides; the resolved config is saved per run."""
import copy
import json
from pathlib import Path

import yaml

from .utils import PROJECT_ROOT, read_json


def _parse_value(text):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def deep_update(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section in override: {dotted}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key in override: {dotted}")
    node[keys[-1]] = value


def load_config(paths, overrides=()):
    """Later files override earlier ones; overrides are `key.sub=value` strings.

    Unknown override keys raise, so a typo cannot silently fall back to a default.
    """
    cfg = {}
    for p in paths:
        deep_update(cfg, yaml.safe_load(Path(p).read_text()) or {})
    for item in overrides:
        key, _, value = item.partition("=")
        set_dotted(cfg, key.strip(), _parse_value(value))
    resolve_model_paths(cfg)
    return cfg


def resolve_model_paths(cfg):
    """Fill `models.<name>.path` from configs/model_paths.json (written by download_models.py)."""
    record_path = PROJECT_ROOT / "configs" / "model_paths.json"
    if not record_path.exists():
        return
    record = read_json(record_path)
    for name, spec in cfg.get("models", {}).items():
        if isinstance(spec, dict) and name in record:
            if not spec.get("path"):
                spec["path"] = record[name]["path"]
            if not spec.get("revision"):
                spec["revision"] = record[name]["revision"]


def dump_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


def config_json(cfg):
    return json.loads(json.dumps(cfg))
