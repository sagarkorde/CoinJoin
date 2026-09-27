"""
A polite multi-endpoint Bitcoin transaction fetcher.

An earlier verification pass used ten concurrent workers against a single
public explorer and was rate-limited with HTTP 429, after which it made
almost no progress while continuing to issue requests. That is both futile
and discourteous to a free service, so fetching is handled here instead, with
three properties the earlier code lacked.

  * **Several endpoints.** Requests are spread across public Esplora-compatible
    explorers, so no single service absorbs the whole load.
  * **Per-endpoint cooldown.** A 429 or 5xx response puts that endpoint to
    sleep for a growing interval while the others continue. An endpoint that
    keeps refusing is used less and less rather than hammered.
  * **A global rate ceiling.** Requests are paced against a shared token
    bucket, so the aggregate rate stays near a stated target regardless of
    worker count.

Responses are cached on disk by transaction id, so a rerun costs no requests
and an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINTS = [
    "https://mempool.space/api/tx/{}",
    "https://blockstream.info/api/tx/{}",
]

USER_AGENT = "coinjoin-research/1.0 (academic replication; contact via repository)"
TARGET_RPS = 2.5          # aggregate ceiling across all workers
COOLDOWN_BASE = 20.0      # seconds an endpoint rests after a refusal
COOLDOWN_MAX = 300.0
MAX_ATTEMPTS = 6


class _RateLimiter:
    """Token bucket shared by every worker."""

    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if wait:
            time.sleep(wait)


class ExplorerPool:
    """Fetches transactions across several explorers, with caching."""

    def __init__(self, cache_dir: Path, endpoints: list[str] | None = None,
                 target_rps: float = TARGET_RPS):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.endpoints = list(endpoints or ENDPOINTS)
        self._limiter = _RateLimiter(target_rps)
        self._lock = threading.Lock()
        self._blocked_until = {e: 0.0 for e in self.endpoints}
        self._penalty = {e: 0.0 for e in self.endpoints}
        self._cursor = 0
        self.stats = {"cache_hits": 0, "fetched": 0, "not_found": 0,
                      "failed": 0, "throttled": 0}

    # -- endpoint selection -------------------------------------------------
    def _pick(self) -> str | None:
        """Least-recently-used endpoint that is not cooling down."""
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self.endpoints)):
                ep = self.endpoints[self._cursor % len(self.endpoints)]
                self._cursor += 1
                if self._blocked_until[ep] <= now:
                    return ep
            soonest = min(self._blocked_until.values())
        time.sleep(max(0.5, soonest - time.monotonic()))
        return None

    def _penalise(self, ep: str) -> None:
        with self._lock:
            self._penalty[ep] = min(COOLDOWN_MAX,
                                    max(COOLDOWN_BASE, self._penalty[ep] * 2))
            self._blocked_until[ep] = time.monotonic() + self._penalty[ep]
            self.stats["throttled"] += 1

    def _reward(self, ep: str) -> None:
        with self._lock:
            self._penalty[ep] = 0.0

    # -- fetching -----------------------------------------------------------
    def get(self, txid: str) -> dict | None:
        path = self.cache / f"{txid}.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self.stats["cache_hits"] += 1
                return d
            except json.JSONDecodeError:
                path.unlink()

        for _ in range(MAX_ATTEMPTS):
            ep = self._pick()
            if ep is None:
                continue
            self._limiter.acquire()
            try:
                req = urllib.request.Request(ep.format(txid),
                                             headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=30) as r:
                    d = json.load(r)
                path.write_text(json.dumps(d), encoding="utf-8")
                self._reward(ep)
                with self._lock:
                    self.stats["fetched"] += 1
                return d
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    with self._lock:
                        self.stats["not_found"] += 1
                    return None
                if e.code == 429 or 500 <= e.code < 600:
                    self._penalise(ep)
                else:
                    time.sleep(1.0)
            except Exception:
                self._penalise(ep)

        with self._lock:
            self.stats["failed"] += 1
        return None

    def summary(self) -> dict:
        with self._lock:
            return dict(self.stats)
