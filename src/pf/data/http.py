"""Wspólne narzędzia sieciowe: ograniczenie częstotliwości zapytań i ponawianie z backoffem."""

from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

import requests

log = logging.getLogger(__name__)

T = TypeVar("T")

USER_AGENT = "Mozilla/5.0 (pf research pipeline)"


class RateLimiter:
    """Wymusza minimalny odstęp ``min_interval`` [s] między kolejnymi wywołaniami ``wait``."""

    def __init__(self, min_interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                self._sleep(delay)
                now += delay
        self._last = now


def retry(fn: Callable[[], T], retries: int = 4, backoff: float = 2.0, what: str = "",
          exceptions: tuple[type[BaseException], ...] = (Exception,),
          sleep: Callable[[float], None] = time.sleep) -> T:
    """Wywołuje ``fn``; po błędzie czeka backoff·2^i s (i = 0, 1, …) i ponawia do ``retries`` razy."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except exceptions as exc:
            if attempt == retries:
                raise
            delay = backoff * 2 ** attempt
            log.warning("%s: %s – ponawiam za %.0f s", what or fn, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def get_text(url: str, params: dict | None = None, retries: int = 4, backoff: float = 2.0,
             timeout: float = 30.0) -> str:
    def call() -> str:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.text

    return retry(call, retries, backoff, what=url, exceptions=(requests.RequestException,))


def get_json(url: str, params: dict | None = None, retries: int = 4, backoff: float = 2.0,
             timeout: float = 60.0) -> dict:
    def call() -> dict:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json()

    return retry(call, retries, backoff, what=url, exceptions=(requests.RequestException, ValueError))
