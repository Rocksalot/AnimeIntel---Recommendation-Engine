# recommendation/signals.py - Five independent recommendation signals

"""
Signal A : Content-based  (genre + score similarity to user preferences)
Signal B : Graph-based    (BFS over recommendation graph)
Signal C : Popularity     (raw popularity/members weight)
Signal D : Recency        (boost newer anime)
Signal E : Diversity      (penalty for repeated genres already shown)
"""

import sys
import os
from collections import deque
from typing import Dict, List, Optional, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import clamp, cosine_similarity, jaccard

AnimeMap = Dict[int, dict]   # id -> processed record
AdjList  = Dict[int, Set[int]]


# ── Signal A: Content-based ───────────────────────────────────────────────

def signal_content(anime_map: AnimeMap, # have to distinguish theme, demographic, and genre
                   user_genre_prefs: Dict[str, float],
                   user_score_pref: float,
                   seed_ids: Optional[List[int]] = None) -> Dict[int, float]:

    scores: Dict[int, float] = {}

    pref_genres: Set[str] = {g for g, w in user_genre_prefs.items() if w > 0.1}

    for aid, anime in anime_map.items():
        tag_set = set(anime.get("all_tags") or anime.get("genres", []))

        # Genre similarity
        if pref_genres:
            overlap = sum(user_genre_prefs.get(g, 0.0) for g in tag_set & pref_genres)
            genre_sim = clamp(overlap / max(len(pref_genres), 1))
        else:
            genre_sim = 0.5   # cold start: neutral

        norm_score = anime.get("normalized_score", 0.5)
        score_sim = clamp(1.0 - abs(norm_score - user_score_pref))

        scores[aid] = 0.75 * genre_sim + 0.25 * score_sim

    return scores

def signal_graph(anime_map: AnimeMap,
                 graph: AdjList,
                 seed_ids: List[int],
                 depth: int = config.GRAPH_BFS_DEPTH) -> Dict[int, float]:
    """
    BFS from seed anime (user's history).
    Closer neighbours get higher scores; score decays by 1/(depth+1).
    """
    scores: Dict[int, float] = {}
    if not seed_ids:
        return scores

    visited: Dict[int, int] = {}   # node -> min depth reached
    queue = deque()

    for sid in seed_ids:
        if sid in graph:
            queue.append((sid, 0))
            visited[sid] = 0

    while queue:
        node, d = queue.popleft()
        if d > depth:
            continue
        weight = 1.0 / (d + 1)
        if node not in seed_ids:   # don't score seeds
            scores[node] = max(scores.get(node, 0.0), weight)

        for neighbour in (graph.get(node) or set()):
            if neighbour not in visited or visited[neighbour] > d + 1:
                visited[neighbour] = d + 1
                queue.append((neighbour, d + 1))

    # Normalise to [0, 1]
    if scores:
        max_v = max(scores.values())
        if max_v > 0:
            scores = {k: v / max_v for k, v in scores.items()}

    return scores

#these are computed in feature extraction time
def signal_popularity(anime_map: AnimeMap) -> Dict[int, float]:
    return {aid: anime.get("popularity_weight", 0.0)
            for aid, anime in anime_map.items()}

def signal_recency(anime_map: AnimeMap) -> Dict[int, float]:
    return {aid: anime.get("recency_score", 0.0)
            for aid, anime in anime_map.items()}

def signal_hype(anime_map: AnimeMap) -> Dict[int, float]:
    return {aid: anime.get("hype_score", 0.0)
            for aid, anime in anime_map.items()}


def signal_diversity_penalty(anime_map: AnimeMap,
                              recently_shown: List[int]) -> Dict[int, float]:
    if not recently_shown:
        return {aid: 0.0 for aid in anime_map}

    # Build genre/tag profile of recently shown
    shown_genres: Dict[str, int] = {}
    for sid in recently_shown:
        anime = anime_map.get(sid, {})
        for g in (anime.get("all_tags") or anime.get("genres", [])):
            shown_genres[g] = shown_genres.get(g, 0) + 1

    total_shown = len(recently_shown)
    genre_freq: Dict[str, float] = {g: c / total_shown for g, c in shown_genres.items()}

    penalties: Dict[int, float] = {}
    for aid, anime in anime_map.items():
        tag_set = set(anime.get("all_tags") or anime.get("genres", []))
        if not tag_set:
            penalties[aid] = 0.0
            continue
        overlap_score = sum(genre_freq.get(g, 0.0) for g in tag_set) / len(tag_set)
        penalties[aid] = clamp(overlap_score)

    return penalties