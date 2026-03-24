"""
End-to-end evaluation of the abandonment detection pipeline.

Runs each video listed in a ground truth JSON file through the DeepStream
pipeline, collects abandonment events via the EventLogger, and computes
detection accuracy, temporal error, and spatial (IoU) metrics.

Usage (from inference/source/):
    python -m test.evaluate --ground-truth test/ground_truth.json --video-dir /path/to/videos

Requires a running DeepStream environment (e.g. Jetson Orin Nano).
"""

import argparse
import json
import os
import sys

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # type: ignore

import constants as c
from pipeline.builder import build_pipeline
from logic.process import process_frame, _contexts
from test.event_logger import EventLogger
from test.metrics import evaluate_video, summarise, EvalSummary


# ── Shared logger instance (used by the probe) ──────────────────────────
_logger = EventLogger()


def _make_probe(logger: EventLogger):
    """Returns a buffer probe function that feeds frame data into the logger."""

    def _probe(pad, info, u_data):
        import pyds

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        frame = batch_meta.frame_meta_list

        while frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame.data)
            except StopIteration:
                break

            obj = frame_meta.obj_meta_list
            persons = []
            luggage_items = []

            while obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj.data)
                except StopIteration:
                    break

                if obj_meta.object_id != c.UNTRACKED_OBJECT_ID:
                    if obj_meta.class_id == c.PERSON_CLASS_ID:
                        persons.append(obj_meta)
                    elif obj_meta.class_id == c.LUGGAGE_CLASS_ID:
                        luggage_items.append(obj_meta)

                try:
                    obj = obj.next
                except StopIteration:
                    break

            # Capture bounding boxes before they are consumed
            luggage_bboxes: dict[int, tuple[float, float, float, float]] = {}
            for item in luggage_items:
                r = item.rect_params
                luggage_bboxes[item.object_id] = (r.left, r.top, r.width, r.height)

            luggage_info = process_frame(persons, luggage_items)
            logger.log_frame(frame_meta.frame_num, luggage_info, luggage_bboxes)

            try:
                frame = frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return _probe


def _run_pipeline_for_video(video_path: str, logger: EventLogger) -> None:
    """Run the DeepStream pipeline on a single video and collect events."""
    logger.reset()
    _contexts.clear()

    # Override the source URI for this video
    original_uri = c.SOURCE_URI
    c.SOURCE_URI = f"file://{video_path}"

    try:
        pipeline, elements = build_pipeline()

        tracker_pad = elements["object_tracker"].get_static_pad("src")
        tracker_pad.add_probe(Gst.PadProbeType.BUFFER, _make_probe(logger), 0)

        pipeline.set_state(Gst.State.PLAYING)
        loop = GLib.MainLoop()
        bus = pipeline.get_bus()
        bus.add_signal_watch()

        def on_bus_message(bus, msg):
            match msg.type:
                case Gst.MessageType.EOS:
                    loop.quit()
                case Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    print(f"  ERROR: {err.message}", file=sys.stderr)
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
        _run_pipeline_for_video(video_path, logger)

        print(f"    Detected {len(logger.events)} abandonment event(s)")
        result = evaluate_video(
            video_name, gt_entry, logger.events, args.temporal_tolerance
        )
        results.append(result)

    summary = summarise(results)
    _print_summary(summary)


if __name__ == "__main__":
    main()
