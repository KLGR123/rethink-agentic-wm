#!/usr/bin/env python3

"""WebShop environment HTTP server.
Run this with conda env (Python 3.10 + pyserini + spacy).
The verl training process talks to it via HTTP, keeping the two Python environments
fully separate.
"""

import argparse
import asyncio
import logging
import os
import random
import sys
import time
import uuid
from typing import Any
import numpy as np
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 600  # release leaked sessions after 10 min


def setup_java_home() -> None:
    """Point JAVA_HOME/JVM_PATH at the JVM bundled in the conda env.

    Fallback for running this file directly (the wrapper script sets these too).
    `setdefault` only, so an explicit env from the caller always wins.

    Java 11 ships inside the env at <env>/lib/jvm. The <env>/bin/java stub is a symlink
    into it, so resolve the link to find the real JVM root — jnius looks for
    $JAVA_HOME/lib/server/libjvm.so and will not find it under the env root.
    """

    java_bin = os.path.join(os.path.dirname(sys.executable), "java")
    if os.path.isfile(java_bin):
        java_home = os.path.dirname(os.path.dirname(os.path.realpath(java_bin)))
        os.environ.setdefault("JAVA_HOME", java_home)
        jvm_path = os.path.join(java_home, "lib", "server", "libjvm.so")
        if os.path.isfile(jvm_path):
            os.environ.setdefault("JVM_PATH", jvm_path)


class EnvPool:
    """Manages N WebAgentTextEnv instances shared across concurrent requests.
    All env calls are dispatched to a thread pool because they are synchronous
    and CPU-intensive (Lucene BM25 + BeautifulSoup).
    """

    def __init__(
        self,
        num_envs: int,
        webshop_path: str,
        observation_mode: str = "text",
        file_path: str = "/root/WebShop/data/items_shuffle.json",
    ):
        self._num_envs = num_envs
        self._webshop_path = webshop_path
        self._observation_mode = observation_mode
        self._file_path = file_path
        self._envs: list[Any] = []
        self._idle: asyncio.Queue = asyncio.Queue()
        self._active: dict[str, dict] = {}  # session_id -> {env_idx, deadline}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ttl_task: asyncio.Task | None = None

    def _create_envs(self) -> None:
        """Blocking init: construct all env instances (called once at startup)."""

        if self._webshop_path not in sys.path:
            sys.path.insert(0, self._webshop_path)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: PLC0415
        from web_agent_site.envs.web_agent_text_env import SimServer  # noqa: PLC0415
        from web_agent_site.engine.engine import (  # noqa: PLC0415
            load_products,
            init_search_engine,
        )
        from web_agent_site.engine.goal import get_goals  # noqa: PLC0415

        # Parse items_shuffle.json (5.5 GB) and build the Lucene engine exactly once.
        # WebAgentTextEnv otherwise repeats both per instance for identical, read-only data.
        logger.info("Loading products + search engine once for %d envs …", self._num_envs)
        all_products, product_item_dict, product_prices, _ = load_products(
            filepath=self._file_path, num_products=None, human_goals=True
        )
        goals = get_goals(all_products, product_prices, human_goals=True)
        random.seed(233)
        random.shuffle(goals)
        search_engine = init_search_engine(num_products=None)
        logger.info("Loaded %d products / %d goals.", len(all_products), len(goals))

        # One SimServer per env so each has an independent user_sessions table (per-episode
        # state); the heavy tables below are shared by reference, not copied.
        class _SharedServer(SimServer):
            def __init__(self):
                self.base_url = "http://127.0.0.1:3000"
                self.all_products = all_products
                self.product_item_dict = product_item_dict
                self.product_prices = product_prices
                self.search_engine = search_engine
                self.goals = goals
                self.show_attrs = False
                self.weights = [goal["weight"] for goal in goals]
                self.cum_weights = [0] + np.cumsum(self.weights).tolist()
                self.user_sessions = dict()
                self.search_time = 0
                self.render_time = 0
                self.sample_time = 0
                self.assigned_instruction_text = None

        logger.info("Constructing %d WebAgentTextEnv instances …", self._num_envs)
        for i in range(self._num_envs):
            env = WebAgentTextEnv(
                observation_mode=self._observation_mode,
                num_products=None,
                human_goals=True,
                file_path=self._file_path,
                server=_SharedServer(),
            )
            self._envs.append(env)
            self._idle.put_nowait(i)
            if (i + 1) % 8 == 0:
                logger.info("  %d/%d envs ready", i + 1, self._num_envs)
        logger.info("All %d envs ready.", self._num_envs)

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        await loop.run_in_executor(None, self._create_envs)
        self._ttl_task = asyncio.create_task(self._ttl_loop())

    async def _ttl_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            stale = [sid for sid, info in self._active.items() if now > info["deadline"]]
            for sid in stale:
                logger.warning("Session %s timed out; releasing.", sid)
                await self._release(sid)

    # Public API (coroutine-safe; env calls offloaded to executor)
    
    async def reset(self, goal_idx: int) -> dict:
        env_idx = await self._idle.get()
        env = self._envs[env_idx]
        session_id = f"ws-{uuid.uuid4().hex[:8]}"

        def _do_reset():
            obs, _ = env.reset(session=goal_idx)
            available = env.get_available_actions()
            instruction = env.instruction_text
            return obs, available, instruction

        obs, available, instruction = await self._loop.run_in_executor(None, _do_reset)

        self._active[session_id] = {
            "env_idx": env_idx,
            "deadline": time.monotonic() + SESSION_TTL_SECONDS,
        }
        return {
            "session": session_id,
            "obs": obs,
            "instruction": instruction,
            "available_actions": available,
        }

    async def step(self, session_id: str, action: str) -> dict:
        info = self._active.get(session_id)
        if info is None:
            raise KeyError(f"Unknown session: {session_id}")
        info["deadline"] = time.monotonic() + SESSION_TTL_SECONDS

        env = self._envs[info["env_idx"]]

        def _do_step():
            obs, reward, done, _ = env.step(action)
            available = env.get_available_actions()
            return obs, reward, done, available

        obs, reward, done, available = await self._loop.run_in_executor(None, _do_step)
        return {"obs": obs, "reward": float(reward), "done": bool(done), "available_actions": available}

    async def release(self, session_id: str) -> None:
        await self._release(session_id)

    async def _release(self, session_id: str) -> None:
        info = self._active.pop(session_id, None)
        if info is None:
            return
        self._idle.put_nowait(info["env_idx"])

    def health(self) -> dict:
        return {
            "num_envs": self._num_envs,
            "idle": self._idle.qsize(),
            "active": len(self._active),
        }


# HTTP handlers

async def handle_reset(request: web.Request) -> web.Response:
    pool: EnvPool = request.app["pool"]
    data = await request.json()
    result = await pool.reset(int(data["goal_idx"]))
    return web.json_response(result)


async def handle_step(request: web.Request) -> web.Response:
    pool: EnvPool = request.app["pool"]
    data = await request.json()
    try:
        result = await pool.step(data["session"], data["action"])
    except KeyError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
    return web.json_response(result)


async def handle_release(request: web.Request) -> web.Response:
    pool: EnvPool = request.app["pool"]
    data = await request.json()
    await pool.release(data["session"])
    return web.json_response({"ok": True})


async def handle_health(request: web.Request) -> web.Response:
    pool: EnvPool = request.app["pool"]
    return web.json_response(pool.health())


# App lifecycle

async def on_startup(app: web.Application) -> None:
    loop = asyncio.get_event_loop()
    await app["pool"].start(loop)


def build_app(num_envs: int, webshop_path: str, obs_mode: str, file_path: str) -> web.Application:
    pool = EnvPool(
        num_envs=num_envs, 
        webshop_path=webshop_path, 
        observation_mode=obs_mode, 
        file_path=file_path
    )
    app = web.Application()
    app["pool"] = pool
    app.on_startup.append(on_startup)
    app.router.add_post("/reset", handle_reset)
    app.router.add_post("/step", handle_step)
    app.router.add_post("/release", handle_release)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="WebShop environment HTTP server")
    parser.add_argument("--webshop-path", default="")
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument("--obs-mode", default="text", choices=["text", "text_rich", "html"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--file-path", default="", help="Path to items JSON")
    args = parser.parse_args()

    setup_java_home()

    app = build_app(args.num_envs, args.webshop_path, args.obs_mode, args.file_path)
    logger.info("Starting env server on %s:%d with %d envs", args.host, args.port, args.num_envs)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
