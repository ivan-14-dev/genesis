"""Command line: ``python -m genesis {demo,benchmark,inspect}``."""

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
    d = sub.add_parser("demo", help="developmental run of one agent through stages 0-5")
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
    args = ap.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
