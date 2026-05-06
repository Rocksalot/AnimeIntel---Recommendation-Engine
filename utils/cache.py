import json
import os
import time
from typing import Any, Optional


class JsonCache:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._store: dict = {}
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._store = {}
        else:
            self._store = {}

    def save(self):
        """Flush current state to disk."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        Retrieve a cached value.
        If ttl is provided (seconds), expired entries return None.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        if ttl is not None: # This is not being used ATM to avoid refreshing cache, which is normally once 24h. Stopped because pre-processing is quiet intensive, and make testing, development, and presentation hard
            age = time.time() - entry.get("ts", 0)
            if age > ttl:
                return None
        return entry.get("value")

    def set(self, key: str, value: Any):
        self._store[key] = {"value": value, "ts": time.time()} #ts will be used to filter outdated cache

    def has(self, key: str) -> bool:
        return key in self._store

    def delete(self, key: str):
        self._store.pop(key, None)

    def clear(self):
        self._store = {}
        self.save()

    def keys(self) -> list:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"<JsonCache path='{self.filepath}' entries={len(self)}>"
