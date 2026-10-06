"""Small utilities: seeding, JSON logging, hashing."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def to_json(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_default, **kw)


def write_json(path: os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(to_json(obj, indent=2))


def append_jsonl(path: os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(to_json(obj) + "\n")


def read_jsonl(path: os.PathLike) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_of(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(to_json(p, sort_keys=True).encode())
    return h.hexdigest()


def stable_hash_int(s: str) -> int:
    """Process-independent string hash (Python's hash() is salted)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")
