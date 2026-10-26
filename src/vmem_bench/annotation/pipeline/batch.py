"""Compatibility re-export; prefer orchestration.batch."""
from vmem_bench.annotation.pipeline.orchestration.batch import *  # noqa: F403
from vmem_bench.annotation.pipeline.orchestration.batch import main

if __name__ == "__main__":
    main()
