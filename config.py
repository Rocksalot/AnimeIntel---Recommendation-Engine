# config.py - Central configuration for syhstem
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CACHE_PATH = os.path.join(DATA_DIR, "raw_cache.json") # from jikan
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.json") # pre-processed cache

# Jikan API
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
RATE_LIMIT_SLEEP = 0.5        # seconds between API calls
REQUEST_TIMEOUT = 15           # in case server issue? 

TOP_ANIME_PAGES = 80           # pages of top anime to fetch (25 per page)
SEASONAL_FETCH = True          # also fetch current season
FETCH_RECOMMENDATIONS = True   # fetch per-anime recommendations from Jikan
MAX_RECOMMENDATIONS_PER_ANIME = 5  # limit per anime to control API calls. there is a runtime error if no recommendations for that anime, handle this

GENRE_SIM_THRESHOLD = 0.35      # Jaccard similarity floor for genre edges
SCORE_SIM_THRESHOLD = 0.8      # normalised score proximity floor

SIGNAL_WEIGHTS = {
    "content":    0.30,
    "graph":      0.25,
    "popularity": 0.18,
    "recency":    0.10,
    "hype":       0.07,   # hype score signal -was unused feature
    "diversity":  0.10,   # penalty weight
}
TOP_N_RECOMMENDATIONS = 10     # how many results to show the user
GRAPH_BFS_DEPTH = 2            # BFS depth for graph signal - complexity increases exponentially with no real benefit. Keep low

GENRE_DECAY = 0.70             # decay factor for older genre preferences 

EVAL_SAMPLE_SIZE = 20          # anime sampled for diversity metric