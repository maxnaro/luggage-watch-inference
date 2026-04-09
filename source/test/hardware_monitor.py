from __future__ import annotations

import csv
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _iso_time(unix_ts: float) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace("%", "")
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _is_active(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    number = _to_float(value)
    if number is not None:
        return number > 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "active"}
    return False


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return str(value)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(next_prefix, nested, out)
        return

    if isinstance(value, (list, tuple)):
        for idx, nested in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten(next_prefix, nested, out)
        return

    out[prefix] = value


def _detect_throttling(
    flat_stats: dict[str, Any], thermal_threshold_c: float
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    for key, value in flat_stats.items():
        lowered = key.lower()
        if "throt" in lowered and _is_active(value):
            reasons.append(f"{key}={value}")

        if "temp" in lowered:
            maybe_temp = _to_float(value)
            if maybe_temp is not None and maybe_temp >= thermal_threshold_c:
                reasons.append(f"{key}>={thermal_threshold_c:.1f}C")

    return (len(reasons) > 0, reasons)


@dataclass
class _RunningStat:
    count: int = 0
    total: float = 0.0
    maximum: float = float("-inf")

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value > self.maximum:
            self.maximum = value

    def summary(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"samples": 0, "avg": 0.0, "max": 0.0}
        return {
            "samples": self.count,
            "avg": self.total / self.count,
            "max": self.maximum,
        }


class JetsonHardwareMonitor:
    """Collects Jetson hardware telemetry via jtop in a background thread."""

    def __init__(
        self,
        csv_path: str,
        summary_path: str,
        sample_interval_s: float = 1.0,
        thermal_threshold_c: float = 85.0,
    ) -> None:
        self.csv_path = csv_path
        self.summary_path = summary_path
        self.sample_interval_s = max(0.1, float(sample_interval_s))
        self.thermal_threshold_c = float(thermal_threshold_c)

        self._stop_event = threading.Event()
        self._context_lock = threading.Lock()
        self._context_label = ""

        self._thread: threading.Thread | None = None
        self._jtop_cls = None

        self._error: str | None = None
        self._started = False
        self._summary_written = False

        self._started_at_unix: float | None = None
        self._ended_at_unix: float | None = None

        self._sample_count = 0
        self._throttle_count = 0
        self._throttle_reason_counts: dict[str, int] = {}
        self._metrics: dict[str, _RunningStat] = {
            "gpu_util_percent": _RunningStat(),
            "emc_util_percent": _RunningStat(),
            "ram_util_percent": _RunningStat(),
            "cpu_core_mean_percent": _RunningStat(),
            "max_temp_c": _RunningStat(),
        }
        self._board_info: dict[str, Any] = {}

    @property
    def error(self) -> str | None:
        return self._error

    def set_context(self, label: str) -> None:
        with self._context_lock:
            self._context_label = label

    def clear_context(self) -> None:
        with self._context_lock:
            self._context_label = ""

    def start(self) -> tuple[bool, str]:
        try:
            from jtop import jtop  # type: ignore
        except Exception as exc:
            self._error = (
                "jtop is unavailable; install jetson-stats to enable "
                f"hardware logging ({exc})"
            )
            self._write_summary_once()
            return (False, self._error)

        self._jtop_cls = jtop
        self._started = True
        self._thread = threading.Thread(
            target=self._run,
            name="jetson-hardware-monitor",
            daemon=True,
        )
        self._thread.start()
        return (
            True,
            f"enabled (csv={self.csv_path}, summary={self.summary_path})",
        )

    def stop(self) -> str:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._ended_at_unix = self._ended_at_unix or time.time()
        self._write_summary_once()
        return self.summary_path

    def _run(self) -> None:
        self._started_at_unix = time.time()
        started_mono = time.monotonic()

        try:
            with self._jtop_cls(interval=self.sample_interval_s) as jetson:
                self._board_info = _to_json_safe(getattr(jetson, "board", {}))

                with open(self.csv_path, "w", newline="") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=[
                            "timestamp_iso",
                            "timestamp_unix",
                            "elapsed_s",
                            "context",
                            "throttling_detected",
                            "throttling_reasons",
                            "stats_json",
                        ],
                    )
                    writer.writeheader()

                    while jetson.ok() and not self._stop_event.is_set():
                        now_unix = time.time()
                        elapsed_s = time.monotonic() - started_mono
                        stats = dict(getattr(jetson, "stats", {}) or {})

                        flat_stats: dict[str, Any] = {}
                        _flatten("", stats, flat_stats)

                        throttled, reasons = _detect_throttling(
                            flat_stats,
                            self.thermal_threshold_c,
                        )

                        with self._context_lock:
                            context_label = self._context_label

                        writer.writerow(
                            {
                                "timestamp_iso": _iso_time(now_unix),
                                "timestamp_unix": f"{now_unix:.3f}",
                                "elapsed_s": f"{elapsed_s:.3f}",
                                "context": context_label,
                                "throttling_detected": int(throttled),
                                "throttling_reasons": ";".join(reasons),
                                "stats_json": json.dumps(_to_json_safe(stats), sort_keys=True),
                            }
                        )
                        csv_file.flush()

                        self._update_aggregate_stats(flat_stats, throttled, reasons)

                        if self._stop_event.wait(self.sample_interval_s):
                            break
        except Exception as exc:
            self._error = f"hardware monitor failed: {exc}"
        finally:
            self._ended_at_unix = time.time()
            self._write_summary_once()

    def _update_aggregate_stats(
        self,
        flat_stats: dict[str, Any],
        throttled: bool,
        reasons: list[str],
    ) -> None:
        self._sample_count += 1
        if throttled:
            self._throttle_count += 1
            for reason in reasons:
                self._throttle_reason_counts[reason] = (
                    self._throttle_reason_counts.get(reason, 0) + 1
                )

        gpu = _to_float(flat_stats.get("GPU"))
        if gpu is not None:
            self._metrics["gpu_util_percent"].add(gpu)

        emc = _to_float(flat_stats.get("EMC"))
        if emc is not None:
            self._metrics["emc_util_percent"].add(emc)

        ram = _to_float(flat_stats.get("RAM"))
        if ram is not None:
            self._metrics["ram_util_percent"].add(ram)

        cpu_values: list[float] = []
        max_temp_c: float | None = None

        for key, value in flat_stats.items():
            lowered = key.lower()
            if re.fullmatch(r"cpu[0-9]+", lowered):
                maybe_cpu = _to_float(value)
                if maybe_cpu is not None:
                    cpu_values.append(maybe_cpu)

            if "temp" in lowered:
                maybe_temp = _to_float(value)
                if maybe_temp is not None:
                    if max_temp_c is None or maybe_temp > max_temp_c:
                        max_temp_c = maybe_temp

        if cpu_values:
            self._metrics["cpu_core_mean_percent"].add(
                sum(cpu_values) / len(cpu_values)
            )

        if max_temp_c is not None:
            self._metrics["max_temp_c"].add(max_temp_c)

    def _summary_payload(self) -> dict[str, Any]:
        started_at = self._started_at_unix
        ended_at = self._ended_at_unix or time.time()
        duration_s = 0.0
        if started_at is not None:
            duration_s = max(0.0, ended_at - started_at)

        return {
            "monitor_enabled": self._started,
            "error": self._error,
            "sample_interval_s": self.sample_interval_s,
            "thermal_threshold_c": self.thermal_threshold_c,
            "samples": self._sample_count,
            "throttle_samples": self._throttle_count,
            "throttle_ratio": (
                (self._throttle_count / self._sample_count)
                if self._sample_count > 0
                else 0.0
            ),
            "throttle_reasons": self._throttle_reason_counts,
            "started_at": _iso_time(started_at) if started_at is not None else None,
            "ended_at": _iso_time(ended_at),
            "duration_s": duration_s,
            "metrics": {
                name: stat.summary() for name, stat in self._metrics.items()
            },
            "board": self._board_info,
            "csv_path": self.csv_path,
        }

    def _write_summary_once(self) -> None:
        if self._summary_written:
            return

        payload = self._summary_payload()
        with open(self.summary_path, "w") as summary_file:
            json.dump(payload, summary_file, indent=2, sort_keys=True)
        self._summary_written = True
