# processing/feature_extractor.py - Transform raw Jikan data into feature-rich records

import math
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import clamp, normalize, safe_get

# All genre names seen across the dataset (built dynamically)
_ALL_GENRES: List[str] = []


# ── Genre registry ────────────────────────────────────────────────────────

def build_genre_registry(raw_list: List[dict]) -> List[str]:
    """Collect all unique genre/theme/demographic names from raw anime list."""
    genres: set = set()
    for anime in raw_list:
        for g in (anime.get("genres") or []):
            name = g.get("name")
            if name:
                genres.add(name)
        # Also incorporate themes and demographics as features
        for t in (anime.get("themes") or []):
            name = t.get("name")
            if name:
                genres.add(f"theme:{name}")
        for d in (anime.get("demographics") or []):
            name = d.get("name")
            if name:
                genres.add(f"demo:{name}")
    return sorted(genres)


def get_genre_vector(genres: List[str], registry: List[str]) -> List[float]:
    """Multi-hot encode genres against the registry."""
    genre_set = set(genres)
    return [1.0 if g in genre_set else 0.0 for g in registry]


# ── Individual anime extraction ───────────────────────────────────────────

def extract_anime(raw: dict, score_bounds: Tuple[float, float],
                  pop_bounds: Tuple[int, int],
                  genre_registry: List[str]) -> Optional[dict]:
    """
    Convert a raw Jikan anime dict into a structured feature record.
    Returns None if the record is too incomplete to be useful.
    """
    mal_id = raw.get("mal_id")
    if not mal_id:
        return None

    title = (raw.get("title_english") or raw.get("title") or "Unknown")
    score = raw.get("score") or 0.0
    popularity = raw.get("popularity") or 9999
    members = raw.get("members") or 0
    episodes = raw.get("episodes") or 0
    status = raw.get("status") or ""
    genres = [g["name"] for g in (raw.get("genres") or []) if g.get("name")]
    # Incorporate themes and demographics as genre-like features
    themes = [f"theme:{t['name']}" for t in (raw.get("themes") or []) if t.get("name")]
    demographics = [f"demo:{d['name']}" for d in (raw.get("demographics") or []) if d.get("name")]
    all_tags = genres + themes + demographics
    year = safe_get(raw, "aired", "prop", "from", "year") or 0
    image_url = safe_get(raw, "images", "jpg", "image_url") or ""
    synopsis = raw.get("synopsis") or ""
    anime_type = raw.get("type") or "TV"

    # ── Derived features ──────────────────────────────────────────────────

    score_lo, score_hi = score_bounds
    pop_lo, pop_hi = pop_bounds

    normalized_score = clamp(
        (score - score_lo) / (score_hi - score_lo) if score_hi > score_lo else 0.5
    )

    # Popularity rank: lower number = more popular; invert and normalize
    pop_inv = pop_hi - popularity  # higher is better
    popularity_weight = clamp(
        (pop_inv - (pop_hi - pop_hi)) / (pop_hi - pop_lo) if pop_hi > pop_lo else 0.5
    )
    # Simpler: map low rank number → high weight
    # popularity_weight = clamp(1.0 - (popularity - 1) / max(pop_hi, 1))

    # Hype score: ratio of members to rank; rewards anime with lots of watchers relative to rank
    if members and popularity:
        hype_score = clamp(math.log1p(members) / math.log1p(max(popularity, 1)) / 10)
    else:
        hype_score = 0.0

    # Recency score: exponential decay, but classics get a floor boost
    CURRENT_YEAR = 2026
    if year and year > 0:
        age = max(0, CURRENT_YEAR - year)
        recency_score = clamp(math.exp(-age / 20))   # exponential decay, half-life ~14 yrs
        # Classic boost: old but high-scoring anime get a recency floor
        if age > 20 and score > 8.1:
            recency_score = max(recency_score, 0.3)
    else:
        recency_score = 0.0

    genre_vector = get_genre_vector(all_tags, genre_registry)

    return {
        # Raw
        "id": mal_id,
        "title": title,
        "genres": genres,
        "themes": themes,
        "demographics": demographics,
        "all_tags": all_tags,
        "score": score,
        "popularity": popularity,
        "members": members,
        "episodes": episodes,
        "year": year,
        "status": status,
        "type": anime_type,
        "image_url": image_url,
        "synopsis": synopsis[:800] if synopsis else "",
        # Derived
        "normalized_score": normalized_score,
        "popularity_weight": popularity_weight,
        "recency_score": recency_score,
        "hype_score": hype_score,
        "genre_vector": genre_vector,
    }


# ── Batch extraction ──────────────────────────────────────────────────────

def extract_all(raw_list: List[dict]) -> Tuple[List[dict], List[str]]:
    """
    Extract features for all raw anime.
    Returns (processed_list, genre_registry).
    """
    # Step 1: build genre registry
    genre_registry = build_genre_registry(raw_list)

    # Step 2: compute global bounds for normalisation
    scores = [a.get("score") or 0.0 for a in raw_list]
    pops = [a.get("popularity") or 9999 for a in raw_list]
    score_bounds = (min(scores), max(scores))
    pop_bounds = (min(pops), max(pops))

    # Step 3: extract each anime
    processed = []
    for raw in raw_list:
        record = extract_anime(raw, score_bounds, pop_bounds, genre_registry)
        if record:
            processed.append(record)

    print(f"  Feature extraction complete: {len(processed)} records, "
          f"{len(genre_registry)} genres")
    return processed, genre_registry