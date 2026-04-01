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
import json
import os
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore

from .. import constants as c
from ..pipeline.builder import build_pipeline
from ..logic.probes import tracker_src_pad_buffer_probe, set_event_logger
from ..logic.process import _contexts
from .event_logger import EventLogger
from .metrics import evaluate_video, summarise, EvalSummary


def _run_pipeline_for_video(
    video_path: str, logger: EventLogger, headless: bool = False
) -> None:
    """Run the DeepStream pipeline on a single video and collect events."""
    logger.reset()
    _contexts.clear()

    # Wire the logger into the existing probe
    set_event_logger(logger)

    # Override the source URI for this video
    original_uri = c.SOURCE_URI
    c.SOURCE_URI = f"file://{video_path}"

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
        set_event_logger(None)

    if pipeline_error is not None:
        raise RuntimeError(f"Pipeline error: {pipeline_error}")


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
            status = "FP  (false alarm)"
        elif not r.expected_abandonment:
            status = "TN"
        else:
            status = f"FP+FN  frame_err={r.frame_error} (outside tolerance)"
        print(f"  {r.video:<16s} {status}")

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

    Gst.init(None)
    logger = EventLogger()
    results = []

    for video_name, gt_entry in sorted(ground_truth.items()):
        video_path = os.path.join(args.video_dir, video_name)
        if not os.path.isfile(video_path):
            print(f"  SKIP {video_name} (file not found)", file=sys.stderr)
            continue

        print(f"  Running {video_name} ...")
        try:
            _run_pipeline_for_video(video_path, logger, headless=args.headless)
        except RuntimeError as e:
            print(f"  ABORT: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"    Detected {len(logger.events)} abandonment event(s)")
        result = evaluate_video(
            video_name, gt_entry, logger.events, args.temporal_tolerance
        )
        results.append(result)

    summary = summarise(results)
    _print_summary(summary)


if __name__ == "__main__":
    main()
