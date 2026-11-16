#!/usr/bin/env python3

"""ScienceWorld environment HTTP server.

Run this with the sciworld conda env (Python 3.8 + py4j + Java from the env).
The verl training process talks to it via HTTP, keeping the two Python
environments fully separate.

Unlike WebShop (one stable goal per index) and ALFWorld (one game file per
index), ScienceWorld addresses a scenario as (task_name, variation_idx). The
flat goal_idx used in the parquet is a position in the split's goal list built
by tasks.py, so data prep and the server always agree on what index N means.

Each ScienceWorldEnv is a py4j gateway to its own JVM (scienceworld.jar), and
one PythonInterface holds exactly one mutable task/universe state, so a slot
serves one episode at a time and a reset re-loads the scenario. Measured on the
dev machine: ~1.3 s to spawn the JVM, ~220 MB RSS per JVM, ~0.18 s to load a
task, ~60 ms per step. JVM calls release the GIL, so slots step in parallel.
"""

from __future__ import annotations  # the sciworld env is Python 3.8: list[str] etc. are strings only

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

SESSION_TTL_SECONDS = int(os.getenv("SCIWORLD_SESSION_TTL", "300"))  # release leaked sessions
RESET_WAIT_SECONDS = float(os.getenv("SCIWORLD_RESET_WAIT", "60"))  # max queue time for a slot


def setup_java_home() -> None:
    """Point JAVA_HOME at the JVM bundled in the sciworld conda env.

    Fallback for running this file directly (the wrapper script sets it too).
    `setdefault` only, so an explicit env from the caller always wins.

    ScienceWorldEnv launches py4j with plain `java` from PATH, so the JVM lives
    at <env>/lib/jvm and <env>/bin/java is a symlink into it; resolve the link
    to find the real root.
    """

    java_bin = os.path.join(os.path.dirname(sys.executable), "java")
    if os.path.isfile(java_bin):
        java_home = os.path.dirname(os.path.dirname(os.path.realpath(java_bin)))
        os.environ.setdefault("JAVA_HOME", java_home)
        if os.path.dirname(java_bin) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = os.path.dirname(java_bin) + os.pathsep + os.environ.get("PATH", "")

# Add the repository root for both direct-file and module execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.sciworld.tasks import SPLITS, TASKS, collect_goals  # noqa: E402


class EnvPoolExhausted(RuntimeError):
    """No env slot became free within RESET_WAIT_SECONDS."""


class EnvPool:
    """Manages N ScienceWorld env instances shared across concurrent requests.

    A slot reloads the requested (task, variation) on every reset. The env calls
    are synchronous py4j RPCs, so they are dispatched to a thread pool with one
    worker per slot; each slot is therefore only ever touched by one thread at
    a time, which is what the underlying PythonInterface requires.
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_steps: int = 100,
        simplification: str = "",
        splits: list[str] | None = None,
    ):
        self._num_envs = num_envs
        self._max_episode_steps = max_episode_steps
        self._simplification = simplification
        self._splits = splits or list(SPLITS)
        self._goals: dict[str, list[dict]] = {}
        for split in self._splits:
            self._goals[split] = collect_goals(split)
            logger.info("Split %s: %d scenarios.", split, len(self._goals[split]))
        self._envs: list[Any] = []
        # Created in start(), not __init__: aiohttp may construct the app before
        # its event loop exists, and a Queue bound to the wrong loop only fails
        # when a get() actually has to wait (i.e. under concurrency).
        self._idle: asyncio.Queue | None = None
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
        """Blocking init: spawn all JVM gateways and validate the task table."""
        from scienceworld import ScienceWorldEnv

        logger.info("Constructing %d ScienceWorld env instances (one JVM each) …", self._num_envs)
        # The table in tasks.py mirrors TaskMaker1.getMaxVariations(). If the jar
        # or tasks.json changes underneath it, goal_idx would silently shift, so
        # compare against the live simulator instead of trusting the constant.
        for i in range(self._num_envs):
            env = ScienceWorldEnv("boil", envStepLimit=self._max_episode_steps)
            # Without this, py4j serialises RPCs through the callback server's
            # single thread and concurrent slots queue behind each other.
            env._gateway.thread_safe = True
            if i == 0:
                from scripts.sciworld.tasks import TASK_NAMES

                live = {name: env.get_max_variations(name) for name in TASK_NAMES}
                mismatch = {
                    name: (expected, live[name])
                    for name, expected in TASK_NAMES.items()
                    if live[name] != expected
                }
                if mismatch:
                    raise RuntimeError(
                        "tasks.py variation counts disagree with the simulator "
                        f"(expected, live): {mismatch}. Re-generate the table and "
                        "re-prepare the parquet data."
                    )
            self._envs.append(env)
            if (i + 1) % 8 == 0:
                logger.info("  %d/%d envs ready", i + 1, self._num_envs)
        logger.info("All %d envs ready.", self._num_envs)

    async def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._idle = asyncio.Queue()
        # One worker thread per env: a slot must never be driven by two threads
        # at once (the executor may reuse a worker for a different slot only
        # after the previous call finished).
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_envs, thread_name_prefix="sciworld-env"
        )
        await loop.run_in_executor(self._executor, self._create_envs)
        # Fill the queue from the loop thread; asyncio.Queue is not thread-safe.
        for env_idx in range(self._num_envs):
            self._idle.put_nowait(env_idx)
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
        if split not in self._goals:
            raise KeyError(f"Unknown split: {split}")
        goals = self._goals[split]
        if not 0 <= goal_idx < len(goals):
            raise KeyError(
                f"goal_idx {goal_idx} out of range for split {split} "
                f"({len(goals)} scenarios)"
            )
        goal = goals[goal_idx]

        try:
            env_idx = await asyncio.wait_for(self._idle.get(), timeout=RESET_WAIT_SECONDS)
        except asyncio.TimeoutError as exc:
            raise EnvPoolExhausted(
                f"no idle env after {RESET_WAIT_SECONDS:.1f}s "
                f"(idle={self._idle.qsize()}, active={len(self._active)}, "
                f"busy={len(self._busy)}, recycling={len(self._recycle)})"
            ) from exc

        env = self._envs[env_idx]
        session_id = f"sci-{uuid.uuid4().hex[:8]}"
        simplification = self._simplification

        def _do_reset():
            env.load(goal["task_name"], goal["variation_idx"], simplification)
            obs, infos = env.reset()
            return obs, infos

        try:
            obs, infos = await self._call(env_idx, _do_reset)
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
            "instruction": _clean_task_desc(infos.get("taskDesc", "")),
            "inventory": infos.get("inv", ""),
            "admissible_commands": list(infos.get("valid", [])),
            "score": float(infos.get("score", 0.0)),
            "goal_progress": _goal_progress(env),
            "task_name": goal["task_name"],
            "variation_idx": goal["variation_idx"],
        }

    async def step(self, session_id: str, action: str) -> dict:
        info = self._active.get(session_id)
        if info is None:
            raise KeyError(f"Unknown session: {session_id}")
        info["deadline"] = time.monotonic() + SESSION_TTL_SECONDS

        env_idx = info["env_idx"]
        env = self._envs[env_idx]

        def _do_step():
            return env.step(action)

        obs, reward, done, infos = await self._call(env_idx, _do_step)
        # PythonInterface scores 0-100; the task reward is score/100. `reward`
        # here is score - previous score, kept only for logging.
        score = float(infos.get("score", 0.0))
        return {
            "obs": obs,
            "reward": score / 100.0,
            "done": bool(done),
            "won": bool(done) and score >= 100.0,
            "score": score,
            "step_reward": float(reward),
            "moves": int(infos.get("moves", 0)),
            "inventory": infos.get("inv", ""),
            "admissible_commands": list(infos.get("valid", [])),
            "goal_progress": _goal_progress(env),
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
            "idle": self._idle.qsize() if self._idle else 0,
            "active": len(self._active),
            "busy": len(self._busy),
            "recycling": len(self._recycle),
            "tasks": len(TASKS),
            "scenarios": {split: len(goals) for split, goals in self._goals.items()},
        }


def _clean_task_desc(desc: str) -> str:
    """Drop the 'Task Description:' header ScienceWorld prepends."""
    desc = desc.strip()
    marker = "Task Description:"
    if desc.startswith(marker):
        desc = desc[len(marker):].strip()
    return desc


def _goal_progress(env: Any) -> str:
    try:
        return env.get_goal_progress()
    except Exception:  # older jars / uninitialized env: progress is optional
        return ""


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
    num_envs: int, max_episode_steps: int, simplification: str, splits: list[str]
) -> web.Application:
    pool = EnvPool(
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        simplification=simplification,
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


def install_scienceworld_path(scienceworld_path: str) -> None:
    """Put the checkout in front of site-packages so code and jar agree."""
    if scienceworld_path and scienceworld_path not in sys.path:
        sys.path.insert(0, scienceworld_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ScienceWorld environment HTTP server")
    parser.add_argument("--scienceworld-path", default="")
    parser.add_argument("--num-envs", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max commands per episode (ScienceWorld default: 100)")
    parser.add_argument("--simplification", default="",
                        help="Comma-separated simplifications, e.g. 'easy' or "
                             "'teleportAction,openDoors' (empty = full task)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS),
                        choices=list(SPLITS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    install_scienceworld_path(args.scienceworld_path)

    app = build_app(args.num_envs, args.max_steps, args.simplification, args.splits)
    logger.info(
        "Starting env server on %s:%d with %d envs (simplifications: %r)",
        args.host, args.port, args.num_envs, args.simplification,
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
