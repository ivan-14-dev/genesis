"""Command line: ``python -m genesis {init,start,demo,benchmark,vision,grounded,inspect,improve,approve,rollback,
dashboard,chat}``
and administration: ``python -m genesis {admin,audit,corpus,lm}`` (see :mod:`genesis.cli_admin`)."""

from __future__ import annotations

import argparse
import json
import os
import sys


def _seeds(spec: str) -> list:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def _chat_terminal(db: str, worlds, objects=None) -> None:
    from .agent import Genesis
    from .chat import ChatEngine
    agent = Genesis(db)
    engine = ChatEngine(agent, worlds, objects)
    cid = engine.new_conversation()["id"]
    print("GENESIS (no external language model). Type a number sequence, 'help'/'aide', or Ctrl+D to quit.")
    try:
        while True:
            try:
                text = input("\nyou> ").strip()
            except EOFError:
                break
            if text:
                print("\ngenesis> " + engine.send(cid, text)["reply"]["text"])
    except KeyboardInterrupt:
        pass
    finally:
        agent.close()


def _version() -> str:
    from importlib import metadata
    try:
        return metadata.version("genesis-cognitive")
    except metadata.PackageNotFoundError:
        return "source tree"


def _serve_chat(db: str, host: str, port: int, allow_host, pub=None, open_browser: bool = False) -> int:
    from .chat_server import serve as serve_chat
    from .environments import ObjectWorld, SequenceWorld, sandboxed
    worlds = lambda s: sandboxed(SequenceWorld(s), 1000)  # noqa: E731 - harness supplies sandboxed worlds
    objects = lambda s: sandboxed(ObjectWorld(s), 2000)  # noqa: E731
    on_ready = None
    if open_browser:
        import webbrowser
        on_ready = webbrowser.open
    try:
        serve_chat(db, host, port, worlds, objects, allow_host, pub, on_ready)
    except OSError as exc:
        print(f"cannot serve on {host}:{port}: {exc} (already running? try --port)", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="genesis")
    ap.add_argument("--version", action="version", version=f"genesis {_version()}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ini = sub.add_parser("init", help="first-time setup: admin key, memory and generated web credentials")
    ini.add_argument("--home", help="GENESIS folder (default: $GENESIS_HOME or ~/.genesis)")
    ini.add_argument("--user", default="admin", help="web administrator user name")
    ini.add_argument("--port", type=int, default=8766, help="port written in the credentials file")
    ini.add_argument("--reset-password", action="store_true", help="sign a new policy with a new password")
    ini.add_argument("--json", action="store_true", help="machine-readable output")
    st = sub.add_parser("start", help="serve the web interface of the memory created by 'genesis init'")
    st.add_argument("--home", help="GENESIS folder (default: $GENESIS_HOME or ~/.genesis)")
    st.add_argument("--host", default="127.0.0.1")
    st.add_argument("--port", type=int, default=8766)
    st.add_argument("--open", action="store_true", help="open the web interface in the browser")
    st.add_argument("--allow-host", action="append", default=[], help="extra accepted Host header (repeatable)")
    d = sub.add_parser("demo", help="developmental run of one agent through stages 0-7")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--db", default="runs/demo.db")
    b = sub.add_parser("benchmark", help="multi-seed benchmarks with baselines and controls")
    b.add_argument("--seeds", default="100-109")
    b.add_argument("--out", default="runs/benchmark.json")
    b.add_argument("--quick", action="store_true")
    v = sub.add_parser("vision", help="phase 5: rule discovery from pixels vs the symbolic lab (sample counts)")
    v.add_argument("--seeds", default="100-109")
    v.add_argument("--looks", type=int, default=10, help="motor-babbling looks used to learn to see")
    v.add_argument("--kinds", default="neural,kmeans", help="visual codes to compare (first one is judged)")
    v.add_argument("--pixel-noise", type=float, default=0.08)
    v.add_argument("--out", default="runs/vision.json")
    gr = sub.add_parser("grounded", help="object-concept learning, transfer, baselines and adaptation")
    gr.add_argument("--seed", type=int, default=0)
    gr.add_argument("--steps", type=int, default=80, help="training interactions per phase (1-256)")
    gr.add_argument("--db", default="runs/grounded.db", help="new memory path; existing files are never replaced")
    i = sub.add_parser("inspect", help="inspect a memory database")
    i.add_argument("--db", default="runs/demo.db")
    i.add_argument("--explain", type=int, help="knowledge id to explain")
    i.add_argument("--status", help="filter knowledge by status")
    im = sub.add_parser("improve", help="one isolated self-improvement cycle (never deploys)")
    im.add_argument("--db", default="runs/demo.db")
    im.add_argument("--trial-seeds", default="9001-9003")
    im.add_argument("--holdout-seeds", default="9101-9103")
    im.add_argument("--params", default=None, help="comma-separated tunable parameters")
    ap_ = sub.add_parser("approve", help="human approval: deploy a pending model version")
    ap_.add_argument("--db", default="runs/demo.db")
    ap_.add_argument("--version", type=int, required=True)
    ap_.add_argument("--by", required=True, help="name of the approving human")
    ap_.add_argument("--key", help="admin private key (required once an admin key is pinned)")
    rb = sub.add_parser("rollback", help="restore the configuration active before the last deployment")
    rb.add_argument("--db", default="runs/demo.db")
    ds = sub.add_parser("dashboard", help="read-only monitoring web interface")
    ds.add_argument("--db", default="runs/demo.db")
    ds.add_argument("--host", default="127.0.0.1")
    ds.add_argument("--port", type=int, default=8765)
    ds.add_argument("--allow-host", action="append", default=[], help="extra accepted Host header (repeatable)")
    ch = sub.add_parser("chat", help="ChatGPT-style entry point (web UI, or --terminal); no external LLM")
    ch.add_argument("--db", default="runs/demo.db", help="memory to use (a newborn agent if it does not exist)")
    ch.add_argument("--host", default="127.0.0.1")
    ch.add_argument("--port", type=int, default=8766)
    ch.add_argument("--terminal", action="store_true", help="chat in the terminal instead of the browser")
    ch.add_argument("--allow-host", action="append", default=[], help="extra accepted Host header (repeatable)")
    ch.add_argument("--admin-pub", help="expected admin public key (file or hex); refuse to start otherwise")
    from .cli_admin import add_parsers, run as run_admin
    add_parsers(sub)
    args = ap.parse_args(argv)
    if args.cmd in ("admin", "audit", "corpus", "lm"):
        return run_admin(args)
    if getattr(args, "db", None) and args.cmd in ("improve", "approve", "rollback", "dashboard") \
            and not os.path.exists(args.db):
        print(f"no memory at {args.db}", file=sys.stderr)
        return 1

    if args.cmd == "init":
        from . import bootstrap, governance
        try:
            res = (bootstrap.reset_password if args.reset_password else bootstrap.initialise)(
                args.home, args.user, args.port)
        except (OSError, PermissionError, ValueError, governance.PolicyIntegrityError) as exc:
            print(f"init: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(res, indent=2) if args.json else bootstrap.describe(res))
    elif args.cmd == "start":
        from . import bootstrap, crypto
        home = bootstrap.home_dir(args.home)
        if not bootstrap.is_initialised(home):
            print(f"GENESIS is not initialised in {home}: run 'genesis init' first", file=sys.stderr)
            return 1
        p = bootstrap.paths(home)
        print(f"Sign in with the credentials in {p['credentials']}", flush=True)
        return _serve_chat(p["db"], args.host, args.port, args.allow_host, crypto.load_public(p["pub"]), args.open)
    elif args.cmd == "demo":
        from .demo import run_demo
        run_demo(args.seed, args.db)
    elif args.cmd == "grounded":
        from .experiments import run_grounded_learning
        try:
            res = run_grounded_learning(args.seed, args.steps, args.db)
        except (FileExistsError, ValueError) as exc:
            print(f"grounded experiment refused: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(res, indent=2))
    elif args.cmd == "benchmark":
        from .experiments import run_benchmark_suite
        res = run_benchmark_suite(_seeds(args.seeds), quick=args.quick)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "vision":
        from .experiments import run_pixel_comparison
        res = run_pixel_comparison(_seeds(args.seeds), looks=args.looks, kinds=tuple(args.kinds.split(",")),
                                   pixel_noise=args.pixel_noise)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(json.dumps({k: v for k, v in res.items() if k != "pixel"} | {
            "pixel": {k: {m: x for m, x in v.items() if m != "per_seed"} for k, v in res["pixel"].items()}},
            indent=2, default=str))
    elif args.cmd == "inspect":
        from .memory import LongTermMemory
        if not os.path.exists(args.db):
            print(f"no memory at {args.db}", file=sys.stderr)
            return 1
        ltm = LongTermMemory(args.db)
        if args.explain is not None:
            print(json.dumps(ltm.explain(args.explain), indent=2, default=str))
        else:
            print(f"session {ltm.get_meta('session')}, interactions {ltm.get_meta('interactions')}")
            for k in ltm.knowledge(status=args.status):
                print(f"[{k.id:>4}] {k.status:<13} {k.confidence:5.2f}  {k.key}: {k.statement}   ({k.provenance})")
        ltm.close()
    elif args.cmd == "improve":
        from . import self_improvement as SI
        from .environments import SequenceWorld, sandboxed
        params = tuple(args.params.split(",")) if args.params else SI.DEFAULT_SEARCH
        res = SI.improve(args.db, lambda s: sandboxed(SequenceWorld(s), 10 ** 6), _seeds(args.trial_seeds),
                         _seeds(args.holdout_seeds), params=params)
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "approve":
        from . import self_improvement as SI
        from .crypto import load_private
        seed = load_private(args.key) if args.key else None
        try:
            print(json.dumps({"deployed_config": SI.approve(args.db, args.version, args.by, seed)}))
        except PermissionError as exc:
            print(f"approval refused: {exc}", file=sys.stderr)
            return 1
    elif args.cmd == "rollback":
        from . import self_improvement as SI
        print(json.dumps({"active_config": SI.rollback(args.db)}))
    elif args.cmd == "dashboard":
        from .dashboard import serve
        serve(args.db, args.host, args.port, args.allow_host)
    elif args.cmd == "chat":
        os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
        if args.terminal:
            from .environments import ObjectWorld, SequenceWorld, sandboxed
            _chat_terminal(args.db, lambda s: sandboxed(SequenceWorld(s), 1000),
                           lambda s: sandboxed(ObjectWorld(s), 2000))
        else:
            from .crypto import load_public
            return _serve_chat(args.db, args.host, args.port, args.allow_host,
                               load_public(args.admin_pub) if args.admin_pub else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
