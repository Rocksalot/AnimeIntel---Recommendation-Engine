import sys
import os
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.helpers import clamp


class UserModel:
    def __init__(self):
        self.history: List[int] = []
        self.genre_prefs: Dict[str, float] = {}
        self.score_pref: float = 0.55      # default: slightly above midpoint (less high-score bias)
        self.recently_shown: List[int] = []
        self._DECAY = config.GENRE_DECAY

    def record_selection(self, anime: dict):

        aid = anime.get("id")
        if aid:
            self.history.append(aid)

        genres = anime.get("genres", [])
        norm_score = anime.get("normalized_score", 0.5)

        # Decay existing preferences (recency bias)
        for g in self.genre_prefs:
            self.genre_prefs[g] *= self._DECAY

        # Boost selected anime's genres
        for g in genres:
            self.genre_prefs[g] = clamp(self.genre_prefs.get(g, 0.0) + 0.3)

        # score preference toward selected score... this one feels odd
        self.score_pref = 0.7 * self.score_pref + 0.3 * norm_score

        # Prune unused genres
        self.genre_prefs = {g: w for g, w in self.genre_prefs.items() if w > 0.02}

    def record_shown(self, anime_ids: List[int]):
        """Track which anime were shown to support diversity penalty."""
        self.recently_shown.extend(anime_ids)
        # Keep only the last 30
        self.recently_shown = self.recently_shown[-50:] #tune this more ? GET SOME FEEDBACK

    def is_cold_start(self) -> bool:
        return len(self.history) == 0

    def seen_ids(self) -> set:
        return set(self.history)

    def seed_ids(self) -> List[int]:
        """Return the most recent N selections as seeds for graph BFS."""
        return self.history[-5:]

    def summary(self) -> str:
        top_genres = sorted(self.genre_prefs.items(), key=lambda x: -x[1])[:5]
        genre_str = ", ".join(f"{g}({w:.2f})" for g, w in top_genres) or "none yet"
        return (
            f"  Selections : {len(self.history)}\n"
            f"  Score pref : {self.score_pref:.2f}\n"
            f"  Top genres : {genre_str}"
        )