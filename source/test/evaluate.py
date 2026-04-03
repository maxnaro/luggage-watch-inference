"""
End-to-end evaluation of the abandonment detection pipeline.

Runs each video listed in a ground truth JSON file through the DeepStream
pipeline, collects abandonment events via the EventLogger, and computes
detection accuracy, temporal error, and spatial (IoU) metrics.

Usage (from project root):
    python -m source.test.evaluate --ground-truth source/test/ground_truth.json --video-dir /path/to/videos

Requires a running DeepStream environment (e.g. Jetson Orin Nano).
"""

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore

from .. import constants as c
from ..pipeline.builder import build_pipeline
from ..logic.probes import tracker_src_pad_buffer_probe, set_event_logger
from ..logic.process import reset_contexts
from .event_logger import EventLogger
from .metrics import evaluate_video, summarise, EvalSummary


def _get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """Return (width, height) of a video file using GstPbutils.Discoverer."""
    try:
        gi.require_version("GstPbutils", "1.0")
        from gi.repository import GstPbutils  # type: ignore

        uri = f"file://{os.path.abspath(video_path)}"
        discoverer = GstPbutils.Discoverer.new(5 * Gst.SECOND)
        info = discoverer.discover_uri(uri)
        for stream in info.get_video_streams():
            return (stream.get_width(), stream.get_height())
    except Exception:
        pass
    return None


def _run_pipeline_for_video(
    video_path: str,
    logger: EventLogger,
    gt_entry: dict,
    headless: bool = False,
) -> None:
    """Run the DeepStream pipeline on a single video and collect events."""
    logger.reset()
    reset_contexts()

    # Wire the logger into the existing probe
    set_event_logger(logger)

    # Override constants for this video
    original_uri = c.SOURCE_URI
    original_radius = c.OWNER_RADIUS_PX
    original_timeout = c.ABANDONMENT_TIMEOUT_SECONDS
    c.SOURCE_URI = f"file://{os.path.abspath(video_path)}"
    c.OWNER_RADIUS_PX = gt_entry.get("radius_px", original_radius)
    c.ABANDONMENT_TIMEOUT_SECONDS = gt_entry.get("threshold_s", original_timeout)

    pipeline_error: str | None = None

    try:
        pipeline, elements = build_pipeline(headless=headless)

        tracker_pad = elements["object_tracker"].get_static_pad("src")
        tracker_pad.add_probe(
            Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0
        )

        pipeline.set_state(Gst.State.PLAYING)
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_bus_message(bus, msg):
            nonlocal pipeline_error
            match msg.type:
                case Gst.MessageType.EOS:
                    loop.quit()
                case Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    if "Internal data stream error" in err.message:
                        loop.quit()
                    else:
                        pipeline_error = err.message
                        loop.quit()

        bus.connect("message", on_bus_message)

        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        finally:
            pipeline.set_state(Gst.State.NULL)
    finally:
        c.SOURCE_URI = original_uri
        c.OWNER_RADIUS_PX = original_radius
        c.ABANDONMENT_TIMEOUT_SECONDS = original_timeout
        set_event_logger(None)

    if pipeline_error is not None:
        raise RuntimeError(f"Pipeline error: {pipeline_error}")


@contextlib.contextmanager
def _redirect_to_file(path: str):
    """Redirect stdout and stderr to *path* for the duration of the block."""
    with open(path, "a") as fh:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = fh
        sys.stderr = fh
        # Redirect the C-level file descriptors (1=stdout, 2=stderr) so that
        # GStreamer native debug output ends up in the log file too.
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
        try:
            yield fh
        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class _TeeWriter:
    """Write to multiple file-like objects at once."""

    def __init__(self, *targets):
        self._targets = targets

    def write(self, s):
        for t in self._targets:
            t.write(s)

    def flush(self):
        for t in self._targets:
            t.flush()


def _print_summary(summary: EvalSummary) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for r in summary.results:
        status = ""
        if r.expected_abandonment and r.iou is not None:
            status = f"TP  frame_err={r.frame_error}  IoU={r.iou:.3f}"
        elif r.expected_abandonment and not r.detected_abandonment:
            status = "FN  (missed)"
        elif not r.expected_abandonment and r.detected_abandonment:
            if r.detected_frame is not None:
                status = f"FP  (false alarm at frame {r.detected_frame})"
            else:
                status = "FP  (false alarm)"
        elif not r.expected_abandonment:
            status = "TN"
        else:
            status = f"FP+FN  frame_err={r.frame_error} (outside tolerance)"
        print(f"  {r.video:<24s} {status}")

    print("-" * 60)
    print(f"  TP={summary.tp}  FP={summary.fp}  FN={summary.fn}  TN={summary.tn}")
    print(f"  Precision:        {summary.precision:.3f}")
    print(f"  Recall:           {summary.recall:.3f}")
    print(f"  F1:               {summary.f1:.3f}")

    if summary.mean_frame_error is not None:
        print(f"  Mean frame error: {summary.mean_frame_error:.1f}")
    if summary.mean_iou is not None:
        print(f"  Mean IoU:         {summary.mean_iou:.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate abandonment detection pipeline"
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--video-dir",
        required=True,
        help="Directory containing the test video files",
    )
    parser.add_argument(
        "--temporal-tolerance",
        type=int,
        default=90,
        help="Allowed frame deviation for a true positive (default: 90)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Use fakesink instead of nveglglessink (no display required)",
    )
    args = parser.parse_args()

    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Set up log directory and timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "log")
    os.makedirs(log_dir, exist_ok=True)
    run_log_path = os.path.join(log_dir, f"run_{timestamp}.log")
    eval_log_path = os.path.join(log_dir, f"evaluate_{timestamp}.log")

    Gst.init(None)
    logger = EventLogger()
    results = []

    eval_log = open(eval_log_path, "w")
    # Tee evaluate output to both the terminal and the evaluate log file
    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout, eval_log)

    try:
        print(f"  Logs: {os.path.abspath(run_log_path)}")
        print(f"  Eval: {os.path.abspath(eval_log_path)}")

        for video_name, gt_events in sorted(ground_truth.items()):
            video_path = os.path.join(args.video_dir, video_name)
            if not os.path.isfile(video_path):
                print(f"  SKIP {video_name} (file not found)", file=sys.stderr)
                continue

            # HACK: use parameters from the first event for pipeline configuration
            first_event = gt_events[0] if gt_events else {}
            print(f"  Running {video_name} ({len(gt_events)} expected event(s), radius={first_event.get('radius_px')}px, timeout={first_event.get('threshold_s')}s) ...")
            try:
                with _redirect_to_file(run_log_path):
                    _run_pipeline_for_video(video_path, logger, first_event, headless=args.headless)
            except RuntimeError as e:
                print(f"  ABORT: {e}", file=sys.stderr)
                sys.exit(1)

            print(f"    Detected {len(logger.events)} abandonment event(s)")
            video_size = _get_video_resolution(video_path)
            eval_results = evaluate_video(
                video_name, gt_events, logger.events, args.temporal_tolerance,
                muxer_size=(c.MUXER_WIDTH, c.MUXER_HEIGHT),
                video_size=video_size,
            )
            results.extend(eval_results)

        summary = summarise(results)
        _print_summary(summary)
    finally:
        sys.stdout = original_stdout
        eval_log.close()


if __name__ == "__main__":
    main()
