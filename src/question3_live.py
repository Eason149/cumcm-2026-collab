"""Run the Question 3 policy against the official HTTP simulator."""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from question1_geometry import Point
from question3_strategy import BatchOmniSearch, ClearReply, MeasureReply


class OfficialSimulatorClient:
    def __init__(
        self,
        robot_id: str,
        *,
        base_url: str = "http://127.0.0.1:2026",
        log_path: Path,
        timeout_s: float = 8.0,
        retry_count: int = 3,
    ) -> None:
        if not robot_id:
            raise ValueError("robot_id must not be empty.")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retry_count = retry_count
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        self._counter = 0
        self._position = Point(0.0, 0.0)
        self.current_channel = 1
        self._virtual_time_s = 0.0
        self.entered = False

    @property
    def position(self) -> Point:
        return self._position

    @property
    def virtual_time_s(self) -> float:
        return self._virtual_time_s

    def _request_id(self, prefix: str) -> str:
        self._counter += 1
        return f"q3-{prefix}-{self._counter:05d}"

    def _base(self, request_id: str) -> dict[str, object]:
        return {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def _write_log(
        self,
        path: str,
        payload: dict[str, object],
        response: dict[str, object] | None,
        error: str | None,
    ) -> None:
        record = {
            "real_time": datetime.now().astimezone().isoformat(),
            "path": path,
            "request": payload,
            "response": response,
            "error": error,
        }
        self._log.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log.flush()

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retry_count):
            request = Request(
                self.base_url + path,
                data=encoded,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as http_response:
                    response = json.loads(http_response.read().decode("utf-8"))
                self._write_log(path, payload, response, None)
                if response.get("accepted") is not True:
                    raise RuntimeError(f"Simulator rejected {path}: {response}")
                self._virtual_time_s = float(response["virtual_time_s"])
                return response
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                self._write_log(path, payload, None, f"HTTP {error.code}: {body}")
                raise RuntimeError(f"HTTP {error.code} for {path}: {body}") from error
            except (URLError, TimeoutError, socket.timeout) as error:
                last_error = error
                if attempt + 1 < self.retry_count:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                self._write_log(path, payload, None, repr(error))
        raise RuntimeError(f"No response for {path}: {last_error}")

    def enter(self) -> int:
        response = self._post("/enter", self._base(self._request_id("enter")))
        self.entered = True
        return int(response["remaining_real_duration_s"])

    def measure(self, position: Point, channel: int) -> MeasureReply:
        request_id = self._request_id("measure")
        payload = self._base(request_id)
        payload.update(
            {
                "position": {"x": position.x, "y": position.y},
                "channel": channel,
            }
        )
        response = self._post("/measure", payload)
        self._position = position
        self.current_channel = channel
        result = str(response["measure_result"])
        bearing = float(response["svd_deg"]) if result == "direction" else None
        return MeasureReply(result, self._virtual_time_s, bearing)

    def clear(self, position: Point, channel: int) -> ClearReply:
        request_id = self._request_id("clear")
        payload = self._base(request_id)
        payload.update(
            {
                "position": {"x": position.x, "y": position.y},
                "channel": channel,
            }
        )
        response = self._post("/clear", payload)
        self._position = position
        return ClearReply(response["clear_result"] == "success", self._virtual_time_s)

    def exit(self) -> dict[str, object]:
        response = self._post("/exit", self._base(self._request_id("exit")))
        self.entered = False
        return response

    def close(self) -> None:
        self._log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", required=True, help="Current simulator team ID")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = args.log or Path("results/tables") / f"question3_client_{timestamp}.jsonl"
    client = OfficialSimulatorClient(
        args.robot_id,
        base_url=args.base_url,
        log_path=log_path,
    )
    try:
        remaining = client.enter()
        if remaining < 60:
            raise RuntimeError(f"Only {remaining} real seconds remain; aborting safely.")
        result = BatchOmniSearch().run(client)
        client.exit()
        output = asdict(result)
        output["client_log"] = str(log_path)
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    finally:
        client.close()


if __name__ == "__main__":
    main()
