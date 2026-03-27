import argparse
from pathlib import Path

import gi

import constants as c

gi.require_version(c.GSTREAMER_PACKAGE, c.GSTREAMER_VERSION)
from gi.repository import Gst, GLib  # type: ignore

from logic.helpers.mot_writer import MotWriter
from logic.process import reset_contexts
from logic.probes import tracker_src_pad_buffer_probe
from pipeline.builder import build_pipeline

DEFAULT_VIDEO_EXTENSIONS = ".mp4,.avi,.mov,.mkv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepStream tracking and export MOT rows for CVAT correction."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input video file or a directory containing videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(c.ROOT_DIR) / "outputs" / "mot"),
        help="Directory where MOT files are written (one .txt per video).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Use display sink instead of headless mode.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable overlay rendering updates to reduce processing overhead.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when --input is a directory.",
    )
    parser.add_argument(
        "--extensions",
        default=DEFAULT_VIDEO_EXTENSIONS,
        help=(
            "Comma-separated video extensions used for directory scans "
            f"(default: {DEFAULT_VIDEO_EXTENSIONS})."
        ),
    )
    return parser.parse_args()


def normalize_extensions(raw_extensions: str) -> set[str]:
    normalized: set[str] = set()
    for extension in raw_extensions.split(","):
        cleaned = extension.strip().lower()
        if not cleaned:
            continue
        if not cleaned.startswith("."):
            cleaned = "." + cleaned
        normalized.add(cleaned)

    if not normalized:
        raise ValueError("At least one valid extension is required.")

    return normalized


def discover_videos(input_path: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in extensions:
            raise ValueError(
                f"Input file extension '{input_path.suffix}' not in {sorted(extensions)}"
            )
        return [input_path]

    globber = input_path.rglob if recursive else input_path.glob
    videos = [
        path
        for extension in sorted(extensions)
        for path in globber(f"*{extension}")
        if path.is_file()
    ]
    videos = sorted(set(videos))

    if not videos:
        raise FileNotFoundError(
            f"No videos found in {input_path} matching {sorted(extensions)}"
        )

    return videos


def run_single_video(
    video_path: Path,
    output_file: Path,
    headless: bool,
    draw_overlay: bool,
) -> bool:
    reset_contexts()
    writer = MotWriter(output_file)
    probe_data = {"writer": writer, "draw_overlay": draw_overlay}
    pipeline, elements = build_pipeline(source_uri=video_path.resolve().as_uri(), headless=headless)

    tracker_src_pad = elements["object_tracker"].get_static_pad("src")
    tracker_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        tracker_src_pad_buffer_probe,
        probe_data,
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    run_ok = True

    def on_bus_message(bus, msg):
        nonlocal run_ok

        msg_type = msg.type
        match msg_type:
            case Gst.MessageType.EOS:
                print(c.EOS_MESSAGE)
                loop.quit()
            case Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                run_ok = False
                print(f"{c.ERROR_PREFIX} from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"{c.DEBUG_INFO_PREFIX}: {debug}")
                loop.quit()
            case Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                print(f"{c.WARNING_PREFIX} from {msg.src.get_name()}: {err.message}")
                if debug:
                    print(f"{c.DEBUG_INFO_PREFIX}: {debug}")
            case Gst.MessageType.STATE_CHANGED:
                if msg.src == pipeline:
                    old, new, _pending = msg.parse_state_changed()
                    print(
                        f"{c.PIPELINE_STATE_PREFIX}: {old.value_nick} -> {new.value_nick}"
                    )

    bus.connect(c.BUS_MESSAGE_SIGNAL, on_bus_message)
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        run_ok = False
        print("Interrupt caught, stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        bus.remove_signal_watch()
        writer.close()

    return run_ok


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = normalize_extensions(args.extensions)
    videos = discover_videos(input_path, args.recursive, extensions)

    Gst.init(None)

    failed_videos: list[Path] = []
    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        mot_output = output_dir / f"{video_path.stem}.txt"
        print(f"[{index}/{total}] Processing {video_path}")

        success = run_single_video(
            video_path=video_path,
            output_file=mot_output,
            headless=not args.display,
            draw_overlay=not args.no_overlay,
        )

        if success:
            print(f"MOT rows written to {mot_output}")
        else:
            failed_videos.append(video_path)

    if failed_videos:
        print("\nFailed videos:")
        for failed in failed_videos:
            print(f"- {failed}")
        raise SystemExit(1)

    print("All videos processed successfully.")


if __name__ == "__main__":
    main()
