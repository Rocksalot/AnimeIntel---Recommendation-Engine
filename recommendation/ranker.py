import sys
import os
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from recommendation.signals import (
    signal_content,
    signal_graph,
    signal_popularity,
    signal_recency,
    signal_hype,
    signal_diversity_penalty,
)
from recommendation.user_model import UserModel

AnimeMap = Dict[int, dict]


def _get_all_signals(
    anime_map: AnimeMap,
    graph: dict,
    user: UserModel,
) -> Tuple[dict, dict, dict, dict, dict, dict]:
    content_scores   = signal_content(anime_map, user.genre_prefs, user.score_pref)
    graph_scores     = signal_graph(anime_map, graph, user.seed_ids())
    pop_scores       = signal_popularity(anime_map)
    recency_scores   = signal_recency(anime_map)
    hype_scores      = signal_hype(anime_map)
    diversity_penalt = signal_diversity_penalty(anime_map, user.recently_shown)
    return content_scores, graph_scores, pop_scores, recency_scores, hype_scores, diversity_penalt


def _combine(
    aid: int,
    content: dict,
    graph: dict,
    pop: dict,
    recency: dict,
    hype: dict,
    diversity: dict,
    weights: dict,
) -> float:
    w = weights
    final = (
        w["content"]      * content.get(aid, 0.0)
        + w["graph"]      * graph.get(aid, 0.0)
        + w["popularity"] * pop.get(aid, 0.0)
        + w["recency"]    * recency.get(aid, 0.0)
        + w.get("hype", 0.07) * hype.get(aid, 0.0)
        - w["diversity"]  * diversity.get(aid, 0.0)   # penalty
    )
    return max(0.0, final)


def _extract_franchise(title: str) -> str:
    """
    Extract a rough franchise key from a title to avoid showing
    multiple seasons of the same show (e.g. 'Attack on Titan Season 2').
    Strips common sequel suffixes and returns a normalised base title.
    """
    import re
    t = title.lower().strip()
    t = re.sub(r'\s+(season|part|cour|movie|film|ova|special|the\s+movie)\s*\d*$', '', t)
    t = re.sub(r'\s+\d+(st|nd|rd|th)\s+(season|cour)$', '', t)
    t = re.sub(r':\s*.+$', '', t)   
    t = re.sub(r'\s+[ivx]+$', '', t)  
    t = re.sub(r'\s+\d+$', '', t)    
    return t.strip()


def rank(
    anime_map: AnimeMap,
    graph: dict,
    user: UserModel,
    exclude_ids: Optional[Set[int]] = None,
    top_n: int = config.TOP_N_RECOMMENDATIONS,
    weights: Optional[dict] = None,
) -> List[Tuple[dict, float]]:

    if weights is None:
        weights = config.SIGNAL_WEIGHTS

    excluded = (exclude_ids or set()) | user.seen_ids() | set(user.recently_shown)

    content, graph_s, pop, recency, hype, diversity = _get_all_signals(
        anime_map, graph, user
    )

    scored: List[Tuple[int, float]] = []
    for aid in anime_map:
        if aid in excluded:
            continue
        score = _combine(aid, content, graph_s, pop, recency, hype, diversity, weights)
        scored.append((aid, score))

    scored.sort(key=lambda x: -x[1])

    results = []
    seen_franchises: set = set()
    for aid, score in scored:
        anime = anime_map[aid]
        franchise = _extract_franchise(anime.get("title", ""))
        if franchise in seen_franchises:
            continue
        seen_franchises.add(franchise)
        results.append((anime, score))
        if len(results) >= top_n:
            break

    return results


def cold_start_rank(
    anime_map: AnimeMap,
    top_n: int = config.TOP_N_RECOMMENDATIONS,
) -> List[Tuple[dict, float]]:
    
    pop = signal_popularity(anime_map)
    recency = signal_recency(anime_map)
    hype = signal_hype(anime_map)

    scored = []
    for aid, anime in anime_map.items():
        score_bonus = anime.get("normalized_score", 0.5) * 0.3
        base = (
            0.4 * pop.get(aid, 0.0)
            + 0.15 * recency.get(aid, 0.0)
            + 0.15 * hype.get(aid, 0.0)
            + score_bonus
        )
        scored.append((aid, base))

    scored.sort(key=lambda x: -x[1])

    results = []
    covered_genres: set = set()
    seen_franchises: set = set()

    candidates = [(anime_map[aid], sc) for aid, sc in scored]
    for anime, sc in candidates:
        franchise = _extract_franchise(anime.get("title", ""))
        if franchise in seen_franchises:
            continue
        anime_genres = set(anime.get("genres", []))
        new_genres = anime_genres - covered_genres
        if new_genres or len(results) >= top_n // 2:
            seen_franchises.add(franchise)
            covered_genres.update(anime_genres)
            results.append((anime, sc))
        if len(results) >= top_n:
            break

    if len(results) < top_n:
        for anime, sc in candidates:
            franchise = _extract_franchise(anime.get("title", ""))
            if franchise not in seen_franchises:
                seen_franchises.add(franchise)
                results.append((anime, sc))
            if len(results) >= top_n:
                break

    return results[:top_n]