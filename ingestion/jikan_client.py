import time
import sys
import os
import requests
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.cache import JsonCache

_RAW_CACHE = JsonCache(config.RAW_CACHE_PATH)
_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

def _get(endpoint: str, params: Optional[dict] = None, cache_ttl: int = 86400) -> Optional[dict]:

    key = endpoint + str(sorted((params or {}).items()))
    cached = _RAW_CACHE.get(key, ttl=cache_ttl)
    if cached is not None:
        return cached

    url = f"{config.JIKAN_BASE_URL}/{endpoint}"
    try:
        resp = _SESSION.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        _RAW_CACHE.set(key, data)
        _RAW_CACHE.save()
        time.sleep(config.RATE_LIMIT_SLEEP)   # polite rate limiting
        return data
    except requests.RequestException as exc:
        print(f"  [JikanClient] Request failed: {exc} --- Params: {params}")
        return None

def fetch_top_anime(page: int = 1) -> List[dict]:
    data = _get("top/anime", params={"page": page, "limit": 25})
    return (data or {}).get("data", [])


def fetch_seasonal_anime(year: Optional[int] = None, season: Optional[str] = None) -> List[dict]:
    if year and season:
        endpoint = f"seasons/{year}/{season}"
    else:
        endpoint = "seasons/now"
    data = _get(endpoint, params={"limit": 25})
    return (data or {}).get("data", [])


def fetch_anime_by_id(anime_id: int) -> Optional[dict]:
    data = _get(f"anime/{anime_id}/full")
    return (data or {}).get("data")


def fetch_anime_recommendations(anime_id: int) -> List[dict]:
    data = _get(f"anime/{anime_id}/recommendations")
    entries = (data or {}).get("data", [])
    return entries[: config.MAX_RECOMMENDATIONS_PER_ANIME]


def fetch_anime_characters(anime_id: int) -> List[dict]: # unused feature ATM. the idea is that beloved characters are a good idicator of beloved shows. i.e. levi, naruto, ram etc 
    data = _get(f"anime/{anime_id}/characters") # maybe create feature for members to most favourited character ratio and then normalize over the whole data set to find shows with high favourited characters
    return (data or {}).get("data", [])

def ingest_dataset(pages: int = config.TOP_ANIME_PAGES) -> List[dict]:
    raw_anime: List[dict] = []

    print(f"  Fetching top anime ({pages} page(s))…")
    for page in range(1, pages + 1):
        batch = fetch_top_anime(page)
        print(f"    Page {page}: {len(batch)} anime")
        raw_anime.extend(batch)

    if config.SEASONAL_FETCH:
        print("  Fetching current season anime…")
        seasonal = fetch_seasonal_anime()
        print(f"    Season: {len(seasonal)} anime")
        raw_anime.extend(seasonal)

    # mal_id is unique identifier for every field in the API
    seen: set = set()
    unique = []
    for a in raw_anime:
        mid = a.get("mal_id")
        if mid and mid not in seen:
            seen.add(mid)
            unique.append(a)

    print(f"  Total unique anime after ingestion: {len(unique)}")
    return unique
