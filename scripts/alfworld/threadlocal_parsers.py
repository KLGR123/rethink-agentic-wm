"""Make TextWorld's env reset path thread-safe.

Two independent races make concurrent ``PddlEnv.load()`` + ``reset()`` calls
fail when run from a thread pool:

Race 1 — tatsu parser singletons
---------------------------------
TextWorld keeps one tatsu parser per DSL as a module-level singleton::

    # textworld/envs/pddl/logic/__init__.py
    _PARSER = PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True)

tatsu parsers are *stateful*: ``parse()`` resets the instance's
``_statestack``, ``_rule_stack``, and ``_tokenizer``, then pushes and pops
those stacks as it walks the grammar. Two threads in ``parse()`` on the same
instance stomp on each other's stacks, producing::

    IndexError: list index out of range     # _statestack[-1] after a foreign reset
    IndexError: pop from empty list         # _rule_stack.pop() after a foreign reset
    FailedToken / TypeError: 'NoneType' object is not iterable

Fix: thread-local parser proxies — each worker thread gets its own lazily
constructed parser instance and they never share state.

Race 2 — fast_downward.pddl2sas process globals
-------------------------------------------------
``PddlState.__init__`` calls ``fast_downward.pddl2sas()`` (twice per reset).
``pddl2sas`` mutates three process-global resources that are not protected by
any lock:

* ``sys.argv`` — set to ``["translate.py", "domain", "task"]`` at the start
* ``sys.stdout`` — redirected to a ``StringIO`` for the duration of the call
  (via ``CapturingStdout``)
* the ``fast_downward.translate.options`` module — several flags are written
  depending on the ``optimize`` argument

Two concurrent ``pddl2sas`` calls therefore corrupt each other's
``sys.stdout``, ``sys.argv``, and option flags, producing KeyErrors and
UnboundLocalErrors deep inside the translator.

Fix: a result cache keyed on ``(domain_text, pddl_problem_text, optimize)``.
``pddl2sas`` is purely functional — same inputs always produce the same
``(task, sas)`` — so a cache miss runs the call under a global lock (safe),
and a cache hit skips the call entirely (fully parallel).  After the first
load of each unique game file the lock is never contended.

Call :func:`install` once, before any envs are constructed.
"""

import threading
import copy

_installed = False


# ---------------------------------------------------------------------------
# Race 1 fix: thread-local tatsu parser proxies
# ---------------------------------------------------------------------------

class _ThreadLocalParser:
    """Per-thread tatsu parser, built lazily by ``factory``.

    Parsers are cheap to construct relative to a game load (the grammar tables
    are generated code, not parsed at runtime), and there is one per worker
    thread, so the memory cost is bounded by the env pool size.
    """

    def __init__(self, factory, name):
        self._factory = factory
        self._name = name
        self._local = threading.local()

    def _parser(self):
        parser = getattr(self._local, "parser", None)
        if parser is None:
            parser = self._factory()
            self._local.parser = parser
        return parser

    def parse(self, *args, **kwargs):
        return self._parser().parse(*args, **kwargs)

    def __getattr__(self, item):
        # Only reached for attributes this proxy does not define; `_factory`,
        # `_name` and `_local` live in __dict__ so this cannot recurse.
        return getattr(self._parser(), item)

    def __repr__(self):
        return f"<_ThreadLocalParser {self._name}>"


def _install_threadlocal_parsers() -> None:
    import textworld.envs.pddl.logic
    import textworld.envs.pddl.textgen
    import textworld.logic
    import textworld.textgen
    from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics
    from textworld.envs.pddl.logic.parser import PddlLogicParser
    from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics
    from textworld.envs.pddl.textgen.parser import CSGParser
    from textworld.logic.model import GameLogicModelBuilderSemantics
    from textworld.logic.parser import GameLogicParser
    from textworld.textgen.model import TextGrammarModelBuilderSemantics
    from textworld.textgen.parser import TextGrammarParser

    def _proxy(parser_cls, semantics_cls, name):
        # Fresh semantics per thread too: ModelBuilderSemantics memoizes the
        # constructors it resolves, so it must not be shared either.
        return _ThreadLocalParser(
            lambda: parser_cls(semantics=semantics_cls(), parseinfo=True), name
        )

    # Module-level `_PARSER` is read as a global inside each module's
    # `_parse_and_convert`; rebinding the attribute is sufficient.
    textworld.envs.pddl.logic._PARSER = _proxy(
        PddlLogicParser, PddlLogicModelBuilderSemantics, "pddl.logic"
    )
    textworld.envs.pddl.textgen._PARSER = _proxy(
        CSGParser, CSGModelBuilderSemantics, "pddl.textgen"
    )
    textworld.logic._PARSER = _proxy(
        GameLogicParser, GameLogicModelBuilderSemantics, "logic"
    )
    # TextGrammar uses `cls._PARSER` (class attribute), not a module global.
    textworld.textgen.TextGrammar._PARSER = _proxy(
        TextGrammarParser, TextGrammarModelBuilderSemantics, "textgen"
    )


# ---------------------------------------------------------------------------
# Race 2 fix: cached, serialized pddl2sas
# ---------------------------------------------------------------------------

_pddl2sas_lock = threading.Lock()
_pddl2sas_cache: dict = {}  # (domain, problem, optimize) -> (task, sas)


def _install_cached_pddl2sas() -> None:
    import fast_downward
    _original_pddl2sas = fast_downward.pddl2sas

    def _cached_pddl2sas(domain: str, problem: str, verbose: bool = False,
                         optimize: bool = False):
        key = (domain, problem, optimize)
        cached = _pddl2sas_cache.get(key)
        if cached is not None:
            # Deep-copy so callers that mutate the returned objects don't
            # corrupt the cache entry (PddlState does mutate task.actions).
            task, sas = cached
            return copy.deepcopy(task), sas  # sas is a plain string

        with _pddl2sas_lock:
            # Double-checked: another thread may have populated it while we
            # waited for the lock.
            cached = _pddl2sas_cache.get(key)
            if cached is not None:
                task, sas = cached
                return copy.deepcopy(task), sas

            task, sas = _original_pddl2sas(domain, problem, verbose=verbose,
                                            optimize=optimize)
            _pddl2sas_cache[key] = (copy.deepcopy(task), sas)
            return task, sas

    # Patch the symbol the rest of textworld.envs.pddl.logic imports.
    fast_downward.pddl2sas = _cached_pddl2sas
    import textworld.envs.pddl.logic as _pddl_logic
    _pddl_logic.fast_downward.pddl2sas = _cached_pddl2sas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def install() -> None:
    """Apply both thread-safety fixes.  Idempotent."""
    global _installed
    if _installed:
        return
    _install_threadlocal_parsers()
    _install_cached_pddl2sas()
    _installed = True
