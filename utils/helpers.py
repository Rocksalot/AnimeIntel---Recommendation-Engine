import math
import time
from typing import Any, Dict, Iterable, List, Optional


def safe_get(d: dict, *keys, default=None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def normalize(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def flatten(nested: Iterable) -> List:
    result = []
    for item in nested:
        if isinstance(item, (list, tuple, set)):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def deduplicate_by(items: List[dict], key: str) -> List[dict]:
    seen = set()
    out = []
    for item in items:
        v = item.get(key)
        if v not in seen:
            seen.add(v)
            out.append(item)
    return out


def timer(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"  ⏱  {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def pretty_table(rows: List[Dict], columns: List[str], col_widths: Optional[List[int]] = None):
    if col_widths is None:
        col_widths = [max(len(str(r.get(c, ""))) for r in [{"": c}] + rows) + 2 for c in columns]

    header = "  ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    separator = "  ".join("-" * w for w in col_widths)
    print(header)
    print(separator)
    for row in rows:
        line = "  ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, col_widths))
        print(line)
