import sys
import os
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import jaccard, clamp

AdjList = Dict[int, Dict[int, float]]

def _genre_similarity(a: dict, b: dict) -> float:
    return jaccard(set(a.get("all_tags") or a.get("genres", [])),
                   set(b.get("all_tags") or b.get("genres", [])))


def _score_similarity(a: dict, b: dict) -> float:
    """Return 1 if scores are close, 0 if far apart."""
    diff = abs((a.get("normalized_score") or 0.0) - (b.get("normalized_score") or 0.0))
    return clamp(1.0 - diff)

def build_genre_edges(processed: List[dict]) -> AdjList:
    graph: AdjList = {a["id"]: {} for a in processed}
    n = len(processed)
    threshold = config.GENRE_SIM_THRESHOLD

    for i in range(n):
        for j in range(i + 1, n):
            genre_sim = _genre_similarity(processed[i], processed[j])
            if genre_sim >= threshold:
                score_sim = _score_similarity(processed[i], processed[j])
                weight = clamp(genre_sim * 0.7 + score_sim * 0.3)
                id_i, id_j = processed[i]["id"], processed[j]["id"]
                graph[id_i][id_j] = max(graph[id_i].get(id_j, 0.0), weight)
                graph[id_j][id_i] = max(graph[id_j].get(id_i, 0.0), weight)

    return graph


def add_recommendation_edges(graph: AdjList,
                              processed: List[dict],
                              rec_map: Dict[int, List[int]]) -> AdjList:

    existing_ids = {a["id"] for a in processed}
    for src_id, targets in rec_map.items():
        if src_id not in graph:
            graph[src_id] = {}
        for tgt_id in targets:
            if tgt_id in existing_ids:
                graph[src_id][tgt_id] = max(graph[src_id].get(tgt_id, 0.0), 1.0)
                if tgt_id not in graph:
                    graph[tgt_id] = {}
                graph[tgt_id][src_id] = max(graph[tgt_id].get(src_id, 0.0), 1.0)
    return graph


def build_graph(processed: List[dict],
                rec_map: Dict[int, List[int]]) -> AdjList:
    print("  Building genre-similarity edges…")
    graph = build_genre_edges(processed)

    edge_count = sum(len(v) for v in graph.values()) // 2
    print(f"    Genre edges: {edge_count}")

    if rec_map:
        print("  Merging Jikan recommendation edges…")
        graph = add_recommendation_edges(graph, processed, rec_map)
        edge_count = sum(len(v) for v in graph.values()) // 2
        print(f"    Total edges after merge: {edge_count}")

    return graph

def graph_to_json(graph: AdjList) -> Dict[str, List[int]]:
    """Convert {int: set} adjacency list to JSON-serialisable {str: list}."""
    return {str(k): sorted(v) for k, v in graph.items()}


def graph_from_json(data: Dict[str, List[int]]) -> AdjList:
    """Restore adjacency list from JSON-serialised form."""
    return {int(k): set(v) for k, v in data.items()}