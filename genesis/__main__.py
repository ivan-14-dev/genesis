"""Command line: ``python -m genesis {demo,benchmark,inspect,improve,approve,rollback,dashboard}``."""

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="genesis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="developmental run of one agent through stages 0-7")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--db", default="runs/demo.db")
    b = sub.add_parser("benchmark", help="multi-seed benchmarks with baselines and controls")
    b.add_argument("--seeds", default="100-109")
    b.add_argument("--out", default="runs/benchmark.json")
    b.add_argument("--quick", action="store_true")
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
    rb = sub.add_parser("rollback", help="restore the configuration active before the last deployment")
    rb.add_argument("--db", default="runs/demo.db")
    ds = sub.add_parser("dashboard", help="read-only monitoring web interface")
    ds.add_argument("--db", default="runs/demo.db")
    ds.add_argument("--host", default="127.0.0.1")
    ds.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if getattr(args, "db", None) and args.cmd in ("improve", "approve", "rollback", "dashboard") \
            and not os.path.exists(args.db):
        print(f"no memory at {args.db}", file=sys.stderr)
        return 1

    if args.cmd == "demo":
        from .demo import run_demo
        run_demo(args.seed, args.db)
    elif args.cmd == "benchmark":
        from .experiments import run_benchmark_suite
        res = run_benchmark_suite(_seeds(args.seeds), quick=args.quick)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(json.dumps(res, indent=2, default=str))
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
        print(json.dumps({"deployed_config": SI.approve(args.db, args.version, args.by)}))
    elif args.cmd == "rollback":
        from . import self_improvement as SI
        print(json.dumps({"active_config": SI.rollback(args.db)}))
    elif args.cmd == "dashboard":
        from .dashboard import serve
        serve(args.db, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
