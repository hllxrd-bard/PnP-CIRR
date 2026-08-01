from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cir.config import load_config  # noqa: E402
from cir.engine import CIREngine  # noqa: E402
from cir.schemas import CIRRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated CIR requests.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    request = CIRRequest.model_validate_json(Path(args.input).read_text(encoding="utf-8"))
    engine = CIREngine(config, warmup=True)
    for _ in range(args.warmup_runs):
        engine.search(request)

    values = []
    breakdowns = []
    for _ in range(args.runs):
        output = engine.search(request)
        values.append(output.timings_ms.total)
        breakdowns.append(output.timings_ms.model_dump())

    report = {
        "runs": args.runs,
        "total_ms": {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[max(0, min(len(values) - 1, int(0.95 * len(values)) - 1))],
        },
        "last_breakdown_ms": breakdowns[-1],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
