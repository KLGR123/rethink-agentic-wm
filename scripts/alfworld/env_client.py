"""Async HTTP client for the ALFWorld env server.

The server runs in a separate process so ALFWorld's Python 3.10 + textworld
deps don't need to exist in the verl training env.
"""

import asyncio
import os

import aiohttp
import logging
from typing import Any

logger = logging.getLogger(__name__)

# A failed /reset or /step is expensive to retry: the server may already have
# executed it (the handler outlives a client disconnect), so a retry can leak an
# env slot or apply the same action twice. Only connection errors — where the
# request provably never reached the server — are retried.
# ALFWorld /reset reloads and compiles a TextWorld game (~1.5 s), so it needs a
# more generous ceiling than WebShop's product lookup.
_RESET_TIMEOUT = float(os.getenv("ALFWORLD_RESET_TIMEOUT", "300"))
_STEP_TIMEOUT = float(os.getenv("ALFWORLD_STEP_TIMEOUT", "60"))
_RELEASE_TIMEOUT = float(os.getenv("ALFWORLD_RELEASE_TIMEOUT", "15"))
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0


class EnvClient:
    """Per-AgentLoopWorker async client. One instance, many concurrent calls."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Per-request timeout is passed to s.post(); the session-level one
            # only bounds connection setup and is generous enough not to race.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
            )
        return self._session

    async def _post(self, path: str, payload: dict, timeout: float) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(_MAX_RETRIES):
            try:
                s = await self._sess()
                async with s.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status >= 500:
                        # 503 from /reset means the env pool stayed exhausted
                        # for RESET_WAIT_SECONDS: fail fast with the reason
                        # instead of retrying into the same full pool.
                        body = (await resp.text())[:300]
                        raise RuntimeError(f"EnvClient POST {path} -> HTTP {resp.status}: {body}")
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientConnectorError as exc:
                # Never reached the server: safe to retry any endpoint.
                if attempt == _MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"EnvClient POST {path} failed after {_MAX_RETRIES} attempts: {exc}"
                    ) from exc
                await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            except asyncio.TimeoutError as exc:
                # The server handler may still complete this request. Retrying
                # would queue another one and (for /reset) leak another slot.
                raise RuntimeError(
                    f"EnvClient POST {path} timed out after {timeout:.1f}s "
                    f"(env server busy or pool exhausted? check /health)"
                ) from exc
            except aiohttp.ClientError as exc:
                raise RuntimeError(f"EnvClient POST {path} failed: {exc}") from exc
        raise RuntimeError("unreachable")

    async def reset(self, goal_idx: int, split: str = "train") -> dict[str, Any]:
        """Reserve an env slot, load one game, return the first observation.

        Response: {"session": str, "obs": str, "instruction": str,
                   "admissible_commands": [str], "game_file": str}
        """
        return await self._post(
            "/reset", {"goal_idx": int(goal_idx), "split": split}, _RESET_TIMEOUT
        )

    async def step(self, session: str, action: str) -> dict[str, Any]:
        """Execute one command and return the result.

        Response: {"obs": str, "reward": float, "done": bool,
                   "won": bool, "score": float, "admissible_commands": [str]}
        """
        return await self._post("/step", {"session": session, "action": action}, _STEP_TIMEOUT)

    async def release(self, session: str) -> None:
        """Return the env slot to the idle pool."""
        try:
            await self._post("/release", {"session": session}, _RELEASE_TIMEOUT)
        except Exception:
            logger.warning("Failed to release session %s — pool may shrink by 1.", session)

    async def health(self) -> dict:
        s = await self._sess()
        async with s.get(
            f"{self.base_url}/health", timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
