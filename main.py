import json
import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

# ── Project imports ───────────────────────────────────────────────────────
import config
from utils.cache import JsonCache
from utils.helpers import deduplicate_by, timer
from ingestion.jikan_client import ingest_dataset, fetch_anime_recommendations
from processing.feature_extractor import extract_all
from processing.graph_builder import (
    build_graph,
    graph_to_json,
    graph_from_json,
)
from recommendation.user_model import UserModel
from recommendation.ranker import rank, cold_start_rank
from evaluation.metrics import full_report, print_report, measure_latency

os.makedirs(config.DATA_DIR, exist_ok=True)

_PROC_CACHE = JsonCache(config.PROCESSED_DATA_PATH)

@timer
def stage_ingest() -> List[dict]:
    print("\n[Stage 1] Data Ingestion")
    return ingest_dataset(pages=config.TOP_ANIME_PAGES)


@timer
def stage_feature_extraction(raw: List[dict]) -> Tuple[List[dict], List[str]]:
    print("\n[Stage 2] Feature Extraction")
    return extract_all(raw)

@timer
def stage_fetch_recommendations(processed: List[dict]) -> Dict[int, List[int]]:
    """Fetch Jikan recommendation links for all anime (with caching)."""
    print("\n[Stage 3] Fetching Recommendation Links")
    rec_map: Dict[int, List[int]] = {}
    if not config.FETCH_RECOMMENDATIONS:
        return rec_map

    total = len(processed)
    for i, anime in enumerate(processed):
        aid = anime["id"]
        print(f"  [{i+1}/{total}] {anime['title'][:100]}", end="\r")
        recs = fetch_anime_recommendations(aid)
        targets = []
        for entry in recs:
            target_id = (entry.get("entry") or {}).get("mal_id")
            if target_id:
                targets.append(target_id)
        if targets:
            rec_map[aid] = targets
    print()
    print(f"  Recommendation links collected for {len(rec_map)} anime")
    return rec_map


@timer
def stage_build_graph(processed: List[dict],
                       rec_map: Dict[int, List[int]]) -> dict:
    print("\n[Stage 4] Graph Construction")
    return build_graph(processed, rec_map)

def load_or_build() -> Tuple[Dict[int, dict], List[str], dict]:
    """
    Try to restore processed data + graph from cache.
    If stale / missing, run the full pipeline and persist.
    """
    # Check if cache is warm
    ## cached_proc = _PROC_CACHE.get("processed", ttl=3600 * 24)  # 24-hour TTL Commented out to avoid recomputation, and refresing cache
    cached_proc = _PROC_CACHE.get("processed")
    cached_registry = _PROC_CACHE.get("genre_registry")
    cached_graph = _PROC_CACHE.get("graph")

    if cached_proc and cached_registry and cached_graph:
        print("\n---Loaded processed dataset from cache.")
        processed = cached_proc
        genre_registry = cached_registry
        graph = graph_from_json(cached_graph)
        anime_map = {a["id"]: a for a in processed}
        print(f"  Anime in dataset : {len(processed)}")
        print(f"  Genres tracked   : {len(genre_registry)}")
        return anime_map, genre_registry, graph

    # Full pipeline
    print("\n---Building pipeline from scratch…")

    raw = stage_ingest()
    processed, genre_registry = stage_feature_extraction(raw)

    rec_map = stage_fetch_recommendations(processed)
    graph = stage_build_graph(processed, rec_map)

    # Persist to cache
    _PROC_CACHE.set("processed", processed)
    _PROC_CACHE.set("genre_registry", genre_registry)
    _PROC_CACHE.set("graph", graph_to_json(graph))
    _PROC_CACHE.save()

    anime_map = {a["id"]: a for a in processed}
    print(f"\n---Pipeline complete. {len(processed)} anime indexed.")
    return anime_map, genre_registry, graph

def _divider(char="─", width=70):
    print(char * width)


def display_recommendations(recs: List[Tuple[dict, float]], page_num: int = 1):
    print(f"\n{'═'*70}")
    print(f"  🎌  TOP {len(recs)} RECOMMENDATIONS  (round {page_num})")
    print(f"{'═'*70}")
    for idx, (anime, score) in enumerate(recs, start=1):
        genres = ", ".join(anime.get("genres", [])[:3]) or "—"
        year   = anime.get("year") or "?"
        ep     = anime.get("episodes") or "?"
        rating = anime.get("score") or 0.0
        print(
            f"  {idx:>2}. {anime['title'][:48]:<50}"
            f"  ★{rating:<4}  [{genres}]"
        )
        print(f"      Year: {year}   Episodes: {ep}   Pipeline score: {score:.4f}")
        _divider("·", 70)


def prompt_selection(recs: List[Tuple[dict, float]]) -> Optional[dict]:
    """
    Ask the user to pick an anime from the list, enter 'q' to quit,
    or 's' to skip / refresh without selecting.
    """
    print("\n  Enter a number to select an anime, 's' to skip, 'q' to quit:")
    while True:
        raw = input("  > ").strip().lower()
        if raw == "q":
            return None
        if raw == "s":
            return {}   # empty dict = skip
        try:
            choice = int(raw)
            if 1 <= choice <= len(recs):
                return recs[choice - 1][0]
            print(f"  Please enter a number between 1 and {len(recs)}.")
        except ValueError:
            print("  Invalid input.")

def interactive_loop(anime_map: Dict[int, dict],
                     genre_registry: List[str],
                     graph: dict):
    user = UserModel()
    round_num = 0
    show_eval = True   # toggle evaluation output

    print("\n" + "═"*70) ## these are pretty useless. for testing 
    print("----ANIME RECOMMENDATION INTELLIGENCE PIPELINE")
    print("  Multi-signal · Graph-aware · Adaptive")
    print("═"*70)
    print(f"  Dataset: {len(anime_map)} anime  |  Genres tracked: {len(genre_registry)}")
    print("  Type the number of an anime to select it.")
    print("  's' = skip/refresh  |  'q' = quit  |  'e' = toggle eval metrics")

    while True:
        round_num += 1

        if user.is_cold_start():
            latency, recs = measure_latency(
                cold_start_rank, anime_map, config.TOP_N_RECOMMENDATIONS
            )
        else:
            latency, recs = measure_latency(
                rank, anime_map, graph, user,
                top_n=config.TOP_N_RECOMMENDATIONS
            )

        if not recs:
            print("\n  No more recommendations available. Thanks for using the system!")
            break

        display_recommendations(recs, round_num)

        if show_eval:
            report = full_report(recs, genre_registry, latency)
            print_report(report)

        if not user.is_cold_start():
            print("  Your preference profile:")
            print(user.summary())
            print()

        # Track shown anime for diversity penalty
        shown_ids = [a["id"] for a, _ in recs]
        user.record_shown(shown_ids)

        selection = prompt_selection(recs)

        if selection is None:
            print("\n---Goodbye! Preferences cleared on exit.")
            break

        if selection == {}:
            print("---Refreshing recommendations…")
            continue

        user.record_selection(selection)
        print(f"\n---Selected: {selection['title']}")

    print("\n  Pipeline session ended.\n")

def main():
    try:
        anime_map, genre_registry, graph = load_or_build()
        interactive_loop(anime_map, genre_registry, graph)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
