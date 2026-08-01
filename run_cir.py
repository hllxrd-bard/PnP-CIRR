from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from cir.config import load_config
from cir.engine import CIREngine
from cir.schemas import CIRRequest, CIROutput, TimingInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composed image retrieval from JSON to JSON.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    parser.add_argument("--input", required=True, help="Path to input JSON.")
    parser.add_argument("--output", required=True, help="Path to output JSON.")
    parser.add_argument("--warmup", action="store_true", help="Warm up SigLIP2 before the request.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = load_config(args.config)
        logging.basicConfig(
            level=getattr(logging, str(config.get("runtime.log_level", "INFO")).upper()),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
        with Path(args.input).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        request = CIRRequest.model_validate(payload)
        engine = CIREngine(config, warmup=args.warmup)
        output = engine.search(request)
        exit_code = 0
    except (ValidationError, Exception) as exc:
        logging.exception("CIR request failed")
        output = CIROutput(
            status="error",
            request={},
            timings_ms=TimingInfo(),
            warnings=[],
            results=[],
            error=str(exc),
        )
        exit_code = 1

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output.model_dump(mode="json"), handle, ensure_ascii=False, indent=2)
    print(f"Wrote: {output_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
