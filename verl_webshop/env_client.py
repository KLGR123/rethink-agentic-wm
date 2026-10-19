"""Async HTTP client for the WebShop env server.

The server runs in a separate process so WebShop's
Python 3.10 + pyserini deps don't need to exist in the verl training env.
"""

import asyncio
import aiohttp
import logging
from typing import Any

from verl_webshop.config import ENV_SERVER_URL

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=60)
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


class EnvClient:
    """Per-AgentLoopWorker async client. One instance, many concurrent calls."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_TIMEOUT)
        return self._session

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(_MAX_RETRIES):
            try:
                s = await self._sess()
                async with s.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise RuntimeError(f"EnvClient POST {path} failed after {_MAX_RETRIES} attempts: {exc}") from exc
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
        raise RuntimeError("unreachable")

    async def reset(self, goal_idx: int) -> dict[str, Any]:
        """Reserve an env slot, set goal, return first observation.
        Response: {"session": str, "obs": str, "instruction": str,
                   "available_actions": {"has_search_bar": bool, "clickables": [str]}}
        """
        return await self._post("/reset", {"goal_idx": int(goal_idx)})

    async def step(self, session: str, action: str) -> dict[str, Any]:
        """Execute one action and return the result.
        Response: {"obs": str, "reward": float, "done": bool,
                   "available_actions": {"has_search_bar": bool, "clickables": [str]}}
        """
        return await self._post("/step", {"session": session, "action": action})

    async def release(self, session: str) -> None:
        """Return the env slot to the idle pool."""
        try:
            await self._post("/release", {"session": session})
        except Exception:
            logger.warning("Failed to release session %s — pool may shrink by 1.", session)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
