#!/usr/bin/env python3
# api.py - Flask API bridge for the Anime Recommendation Intelligence Pipeline
# Run: python api.py
# Then open http://localhost:5000 in your browser

import os
import sys
import uuid
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory, session

# ── Project imports ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.cache import JsonCache
from processing.graph_builder import graph_from_json
from recommendation.user_model import UserModel
from recommendation.ranker import rank, cold_start_rank
from evaluation.metrics import full_report, measure_latency

# ── Flask setup ───────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = os.environ.get("FLASK_SECRET", "anime-intel-secret-change-me")

# ── Load pipeline state once at startup ──────────────────────────────────
_PROC_CACHE = JsonCache(config.PROCESSED_DATA_PATH)

print("\n---Loading pipeline from cache…")
_processed  = _PROC_CACHE.get("processed")
_genre_reg  = _PROC_CACHE.get("genre_registry")
_graph_json = _PROC_CACHE.get("graph")

if not (_processed and _genre_reg and _graph_json):
    print("---No cached data found. Run main.py first to build the pipeline.")
    sys.exit(1)

ANIME_MAP: Dict[int, dict] = {a["id"]: a for a in _processed}
GENRE_REGISTRY = _genre_reg
GRAPH = graph_from_json(_graph_json)
print(f"----{len(ANIME_MAP)} anime loaded  |  {len(GENRE_REGISTRY)} genres  |  Graph ready")

# ── Server-side session store  ────────────────────────────────────────────
# Maps session_id -> UserModel instance
_USER_SESSIONS: Dict[str, UserModel] = {}


def _get_user() -> UserModel:
    """Get or create a UserModel for the current browser session."""
    sid = session.get("sid")
    if not sid or sid not in _USER_SESSIONS:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        _USER_SESSIONS[sid] = UserModel()
    return _USER_SESSIONS[sid]


def _anime_to_card(anime: dict, score: float, rank_pos: int) -> dict:
    """
    Convert a processed anime record + pipeline score into the shape
    the frontend template expects.
    """
    members = anime.get("members", 0)
    return {
        "id":       anime["id"],
        "title":    anime.get("title", "Unknown"),
        "year":     anime.get("year") or "?",
        "episodes": anime.get("episodes") or "?",
        "score":    anime.get("score", 0.0),
        "members":  members,
        "type":     anime.get("type", "TV"),
        "genres":   anime.get("genres", []),
        "img":      anime.get("image_url", ""),
        "pipeline": round(score, 4),
        "rank":     rank_pos,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the HTML frontend."""
    return send_from_directory(".", "anime_intel_dark.html")


@app.route("/api/config")
def api_config():
    """
    Return signal weights and metadata so the frontend can reflect real config.
    """
    return jsonify({
        "signal_weights": config.SIGNAL_WEIGHTS,
        "total_anime":    len(ANIME_MAP),
        "total_genres":   len(GENRE_REGISTRY),
    })


@app.route("/api/recommendations")
def api_recommendations():
    """
    GET /api/recommendations?filter=all|TV|Movie|OVA

    Returns ranked recommendations for the current session user.
    """
    user = _get_user()
    filter_type = request.args.get("filter", "all").strip()

    # Build an exclusion set based on the type filter
    if filter_type != "all":
        exclude_by_type = {
            aid for aid, a in ANIME_MAP.items()
            if a.get("type", "TV") != filter_type
        }
    else:
        exclude_by_type = set()

    t0_import = __import__("time").perf_counter()

    if user.is_cold_start():
        latency, recs = measure_latency(cold_start_rank, ANIME_MAP, config.TOP_N_RECOMMENDATIONS)
        # Apply type filter manually for cold start
        if filter_type != "all":
            recs = [(a, s) for a, s in recs if a.get("type") == filter_type]
            recs = recs[:config.TOP_N_RECOMMENDATIONS]
    else:
        latency, recs = measure_latency(
            rank,
            ANIME_MAP,
            GRAPH,
            user,
            exclude_ids=exclude_by_type,
            top_n=config.TOP_N_RECOMMENDATIONS,
        )

    # Build evaluation metrics
    report = full_report(recs, GENRE_REGISTRY, latency)

    # Track shown IDs
    shown_ids = [a["id"] for a, _ in recs]
    user.record_shown(shown_ids)

    cards = [_anime_to_card(a, s, i + 1) for i, (a, s) in enumerate(recs)]

    return jsonify({
        "recommendations": cards,
        "metrics": report,
        "is_cold_start": user.is_cold_start(),
        "user_profile": {
            "selections":  len(user.history),
            "score_pref":  round(user.score_pref, 3),
            "genre_prefs": {
                g: round(w, 3)
                for g, w in sorted(user.genre_prefs.items(), key=lambda x: -x[1])
                if w > 0.05
            },
            "history_titles": [
                ANIME_MAP[aid]["title"]
                for aid in user.history[-4:][::-1]
                if aid in ANIME_MAP
            ],
        },
    })


@app.route("/api/select", methods=["POST"])
def api_select():
    """
    POST /api/select  { "anime_id": <int> }

    Record a user selection and update the preference model.
    """
    data = request.get_json(force=True) or {}
    anime_id = data.get("anime_id")

    if anime_id is None:
        return jsonify({"error": "anime_id required"}), 400

    anime = ANIME_MAP.get(int(anime_id))
    if not anime:
        return jsonify({"error": f"Unknown anime_id {anime_id}"}), 404

    user = _get_user()
    user.record_selection(anime)

    return jsonify({
        "ok": True,
        "selected": anime.get("title"),
        "user_profile": {
            "selections":  len(user.history),
            "score_pref":  round(user.score_pref, 3),
            "genre_prefs": {
                g: round(w, 3)
                for g, w in sorted(user.genre_prefs.items(), key=lambda x: -x[1])
                if w > 0.05
            },
        },
    })


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """
    POST /api/refresh

    Clear the recently-shown list so diversity penalty resets,
    giving the user a fresh set of recommendations.
    """
    user = _get_user()
    user.recently_shown = []
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """
    POST /api/reset

    Fully reset the user model for the current session (new session).
    """
    sid = session.get("sid")
    if sid and sid in _USER_SESSIONS:
        _USER_SESSIONS[sid] = UserModel()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎌  Anime Intel API starting…")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)
