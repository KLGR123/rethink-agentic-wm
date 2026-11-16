#!/usr/bin/env python3

"""ALFWorld (TextWorld) environment HTTP server.

Run this with the ALFWorld conda env (Python 3.10 + textworld + alfworld).
The verl training process talks to it via HTTP, keeping the two Python
environments fully separate.

Unlike WebShop, ALFWorld has no stable goal_idx API: textworld's batch gym
iterates game files in a shuffled order. This server instead builds a sorted
list of game files per split and loads game i into env slot i on reset, so
goal_idx means the same game at data-prep time and at rollout time.

A reset costs ~1.5 s (TextWorld compiles the .tw-pddl game on load), a step
~80 ms, which is why resets are offloaded to a thread pool and the client keeps
episodes in flight below the pool size.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = int(os.getenv("ALFWORLD_SESSION_TTL", "300"))  # release leaked sessions
RESET_WAIT_SECONDS = float(os.getenv("ALFWORLD_RESET_WAIT", "60"))  # max queue time for a slot

# Add the repository root for both direct-file and module execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.alfworld.game_files import collect_game_files, default_data_path, read_instruction, SPLITS  # noqa: E402
from scripts.alfworld.threadlocal_parsers import install as _install_threadlocal_parsers  # noqa: E402
_install_threadlocal_parsers()  # must run before any env is constructed


class EnvPoolExhausted(RuntimeError):
    """No env slot became free within RESET_WAIT_SECONDS."""


class EnvPool:
    """Manages N TextWorld env instances shared across concurrent requests.

    A slot reloads (and therefore recompiles) the requested game on every
    reset. The env calls are synchronous and CPU-bound, so they are dispatched
    to a thread pool with one worker per slot.
    """

    def __init__(
        self,
        num_envs: int,
        alfworld_data: str,
        max_episode_steps: int = 50,
        splits: list[str] | None = None,
    ):
        self._num_envs = num_envs
        self._alfworld_data = alfworld_data
        self._max_episode_steps = max_episode_steps
        self._splits = splits or ["train", "valid_seen", "valid_unseen"]
        self._game_files: dict[str, list[str]] = {}
        self._envs: list[Any] = []
        self._idle: asyncio.Queue = asyncio.Queue()
        self._active: dict[str, dict] = {}  # session_id -> {env_idx, deadline}
        # env_idx -> "busy" state used to make cancellation safe. A cancelled
        # HTTP handler does not stop the executor thread, so the slot may only
        # be returned once the env call has actually finished.
        self._busy: set[int] = set()
        self._recycle: set[int] = set()  # return slot as soon as its call ends
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._ttl_task: asyncio.Task | None = None

    def _create_envs(self) -> None:
        """Blocking init: collect games and construct all env instances."""
        from textworld import EnvInfos
        from textworld.envs.wrappers import Filter, GenericEnvironment, Limit
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        for split in self._splits:
            self._game_files[split] = collect_game_files(self._alfworld_data, split)
            logger.info("Split %s: %d games.", split, len(self._game_files[split]))

        request_infos = EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])

        def _make_env():
            # AlfredDemangler reads the entity table of the game it loads, so
            # each slot needs its own wrapper stack.
            env = GenericEnvironment(request_infos)
            env = Limit(env, max_episode_steps=self._max_episode_steps)
            for wrapper in [AlfredDemangler(), AlfredInfos, Filter]:
                env = wrapper(env)
            return env

        logger.info("Constructing %d ALFWorld env instances …", self._num_envs)
        for i in range(self._num_envs):
            self._envs.append(_make_env())
            self._idle.put_nowait(i)
            if (i + 1) % 8 == 0:
                logger.info("  %d/%d envs ready", i + 1, self._num_envs)
        logger.info("All %d envs ready.", self._num_envs)

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

        def _init_with_path():
            # alfworld ships its own copy of the env module; prefer the one on
            # the checkout passed via --alfworld-path so code and data agree.
            self._create_envs()

        # One worker thread per env: the default executor is capped well below
        # num_envs on many-core machines and would serialize env calls.
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_envs, thread_name_prefix="alfworld-env"
        )
        await loop.run_in_executor(self._executor, _init_with_path)
        self._ttl_task = asyncio.create_task(self._ttl_loop())

    async def _call(self, env_idx: int, fn):
        """Run a blocking env call in the thread pool with cancellation safety.

        aiohttp may cancel this handler while ``fn`` is still running in the
        worker thread. The slot is therefore only returned from the future's
        done-callback, never from a ``finally`` that runs before ``fn`` ends.
        """
        self._busy.add(env_idx)
        fut = self._loop.run_in_executor(self._executor, fn)
        try:
            return await asyncio.shield(fut)
        finally:
            if fut.done():
                self._finished(env_idx)
            else:
                fut.add_done_callback(lambda _: self._finished(env_idx))

    def _finished(self, env_idx: int) -> None:
        self._busy.discard(env_idx)
        if env_idx in self._recycle:
            self._recycle.discard(env_idx)
            self._idle.put_nowait(env_idx)

    async def _ttl_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            now = time.monotonic()
            stale = [
                sid
                for sid, info in self._active.items()
                # An env whose call is still running must not be handed out.
                if now > info["deadline"] and info["env_idx"] not in self._busy
            ]
            for sid in stale:
                logger.warning("Session %s timed out; releasing.", sid)
                await self._release(sid)

    # Public API (coroutine-safe; env calls offloaded to executor)

    async def reset(self, goal_idx: int, split: str = "train") -> dict:
        if split not in self._game_files:
            raise KeyError(f"Unknown split: {split}")
        games = self._game_files[split]
        if not 0 <= goal_idx < len(games):
            raise KeyError(
                f"goal_idx {goal_idx} out of range for split {split} "
                f"({len(games)} games)"
            )

        try:
            env_idx = await asyncio.wait_for(self._idle.get(), timeout=RESET_WAIT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise EnvPoolExhausted(
                f"no idle env after {RESET_WAIT_SECONDS:.1f}s "
                f"(idle={self._idle.qsize()}, active={len(self._active)}, "
                f"busy={len(self._busy)}, recycling={len(self._recycle)})"
            ) from exc

        env = self._envs[env_idx]
        game_file = games[goal_idx]
        session_id = f"alf-{uuid.uuid4().hex[:8]}"

        def _do_reset():
            env.load(game_file)
            obs, infos = env.reset()
            return obs, list(infos["admissible_commands"]), read_instruction(game_file)

        try:
            obs, admissible, instruction = await self._call(env_idx, _do_reset)
        except BaseException:
            # Cancelled or failed mid-reset: hand the slot back (deferred until
            # the thread finishes if the cancellation beat the reset).
            if env_idx in self._busy:
                self._recycle.add(env_idx)
            else:
                self._idle.put_nowait(env_idx)
            raise

        self._active[session_id] = {
            "env_idx": env_idx,
            "deadline": time.monotonic() + SESSION_TTL_SECONDS,
        }
        return {
            "session": session_id,
            "obs": obs,
            "instruction": instruction,
            "admissible_commands": admissible,
            "game_file": game_file,
        }

    async def step(self, session_id: str, action: str) -> dict:
        info = self._active.get(session_id)
        if info is None:
            raise KeyError(f"Unknown session: {session_id}")
        info["deadline"] = time.monotonic() + SESSION_TTL_SECONDS

        env_idx = info["env_idx"]
        env = self._envs[env_idx]

        def _do_step():
            obs, score, done, infos = env.step(action)
            return obs, score, done, infos

        obs, score, done, infos = await self._call(env_idx, _do_step)
        won = bool(infos.get("won", False))
        # Terminal reward matches the ALFWorld success metric but keeps
        # TextWorld's partial score for trajectories that finish short of the
        # goal, so GRPO still sees a gradient on near-misses.
        reward = 1.0 if won else float(score or 0.0)
        return {
            "obs": obs,
            "reward": reward,
            "done": bool(done),
            "won": won,
            "score": float(score or 0.0),
            "admissible_commands": list(infos.get("admissible_commands", [])),
        }

    async def release(self, session_id: str) -> None:
        await self._release(session_id)

    async def _release(self, session_id: str) -> None:
        info = self._active.pop(session_id, None)
        if info is None:
            return
        env_idx = info["env_idx"]
        if env_idx in self._busy:
            # A cancelled step may still be running; recycle on completion.
            self._recycle.add(env_idx)
        else:
            self._idle.put_nowait(env_idx)

    def health(self) -> dict:
        return {
            "num_envs": self._num_envs,
            "idle": self._idle.qsize(),
            "active": len(self._active),
            "busy": len(self._busy),
            "recycling": len(self._recycle),
            "games": {split: len(games) for split, games in self._game_files.items()},
        }


# HTTP handlers

async def handle_reset(request: web.Request) -> web.Response:
    pool: EnvPool = request.app["pool"]
    data = await request.json()
    try:
        result = await pool.reset(int(data["goal_idx"]), str(data.get("split", "train")))
    except EnvPoolExhausted as exc:
        # Fail fast with a distinguishable status instead of holding the
        # client's slot queue position until its own timeout fires.
        raise web.HTTPServiceUnavailable(text=str(exc)) from exc
    except KeyError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc
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


def build_app(
    num_envs: int, alfworld_data: str, max_episode_steps: int, splits: list[str]
) -> web.Application:
    pool = EnvPool(
        num_envs=num_envs,
        alfworld_data=alfworld_data,
        max_episode_steps=max_episode_steps,
        splits=splits,
    )
    app = web.Application()
    app["pool"] = pool
    app.on_startup.append(on_startup)
    app.router.add_post("/reset", handle_reset)
    app.router.add_post("/step", handle_step)
    app.router.add_post("/release", handle_release)
    app.router.add_get("/health", handle_health)
    return app


def install_alfworld_path(alfworld_path: str) -> None:
    """Put the checkout in front of site-packages so env code and data agree."""
    if alfworld_path and alfworld_path not in sys.path:
        sys.path.insert(0, alfworld_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALFWorld environment HTTP server")
    parser.add_argument("--alfworld-path", default="")
    parser.add_argument("--alfworld-data", default=default_data_path())
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max commands per episode (ALFWorld baseline: 50)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    install_alfworld_path(args.alfworld_path)
    os.environ["ALFWORLD_DATA"] = args.alfworld_data

    app = build_app(args.num_envs, args.alfworld_data, args.max_steps, args.splits)
    logger.info(
        "Starting env server on %s:%d with %d envs (data: %s)",
        args.host, args.port, args.num_envs, args.alfworld_data,
    )
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        # Cancel handlers when the client disconnects, so a timed-out /reset
        # stops waiting for a slot instead of leaking it for the TTL window.
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
