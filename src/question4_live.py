"""Run the Question 4 mixed-source policy against the official simulator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from question3_live import OfficialSimulatorClient
from question4_strategy import MixedDirectionalSearch


class Question4SimulatorClient(OfficialSimulatorClient):
    """Official client with Question 4 request identifiers."""

    def _request_id(self, prefix: str) -> str:
        self._counter += 1
        return f"q4-{prefix}-{self._counter:05d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", required=True, help="Current simulator team ID")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path)
    parser.add_argument(
        "--survey-profile",
        choices=("fast", "turbo", "rapid", "balanced", "certified"),
        default="turbo",
        help="Speed/reliability tradeoff; certified retains the 25-point proof.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log or Path("results/tables") / f"question4_client_{timestamp}.jsonl"
    client = Question4SimulatorClient(
        args.robot_id,
        base_url=args.base_url,
        log_path=log_path,
    )
    try:
        remaining = client.enter()
        if remaining < 60:
            raise RuntimeError(f"Only {remaining} real seconds remain; aborting safely.")
        result = MixedDirectionalSearch(survey_profile=args.survey_profile).run(client)
        client.exit()
        output = asdict(result)
        output["client_log"] = str(log_path)
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    finally:
        client.close()


if __name__ == "__main__":
    main()
