"""CLI entry for ``python -m vmem_bench.annotation.pipeline.servers.fleet``."""

from __future__ import annotations

import json
import sys

from vmem_bench.annotation.pipeline.servers.fleet.registry import list_fleet
from vmem_bench.annotation.pipeline.servers.fleet.supervise import main as supervise_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(
            "usage:\n"
            "  python -m ...fleet list [--probe]\n"
            "  python -m ...fleet.supervise --port N --gpu G --model M -- <cmd...>",
            file=sys.stderr,
        )
        return 0 if args and args[0] in {"-h", "--help"} else 2
    if args[0] == "list":
        probe = "--probe" in args[1:]
        print(json.dumps(list_fleet(probe=probe), ensure_ascii=False, indent=2))
        return 0
    if args[0] == "supervise":
        return supervise_main(args[1:])
    print(f"unknown subcommand: {args[0]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
