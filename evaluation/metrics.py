import time
import sys
import os
import math
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import jaccard

def intra_list_diversity(recommendations: List[dict]) -> float:
    n = len(recommendations)
    if n < 2:
        return 1.0

    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            g_i = set(recommendations[i].get("genres", []))
            g_j = set(recommendations[j].get("genres", []))
            sim = jaccard(g_i, g_j)
            total += 1.0 - sim
            pairs += 1

    return total / pairs if pairs else 0.0

def average_score(recommendations: List[dict]) -> float:
    scores = [a.get("score", 0.0) for a in recommendations]
    return sum(scores) / len(scores) if scores else 0.0


def average_normalized_score(recommendations: List[dict]) -> float:
    scores = [a.get("normalized_score", 0.0) for a in recommendations]
    return sum(scores) / len(scores) if scores else 0.0

def genre_coverage(recommendations: List[dict], all_genres: List[str]) -> float:
    if not all_genres:
        return 0.0
    covered = set()
    for a in recommendations:
        covered.update(a.get("genres", []))
    return len(covered) / len(all_genres)

def measure_latency(fn: Callable, *args, **kwargs) -> Tuple[float, any]:
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed, result

def full_report(recommendations: List[Tuple[dict, float]],
                all_genres: List[str],
                latency_s: float) -> Dict[str, float]:

    anime_list = [a for a, _ in recommendations]
    return {
        "count":                   len(anime_list),
        "intra_list_diversity":    round(intra_list_diversity(anime_list), 4),
        "avg_raw_score":           round(average_score(anime_list), 3),
        "avg_normalized_score":    round(average_normalized_score(anime_list), 4),
        "genre_coverage":          round(genre_coverage(anime_list, all_genres), 4),
        "latency_ms":              round(latency_s * 1000, 2),
    }


def print_report(report: Dict[str, float]):
    print("\n  ── Evaluation Metrics ──────────────────────────────────")
    for k, v in report.items():
        print(f"    {k:<28} {v}")
    print()
