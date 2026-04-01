import gi

from . import constants as c

gi.require_version(c.GSTREAMER_PACKAGE, c.GSTREAMER_VERSION)
from gi.repository import Gst, GLib  # type: ignore

from .pipeline.builder import build_pipeline
from .logic.probes import tracker_src_pad_buffer_probe


def main():
    Gst.init(None)

    # Build the GStreamer pipeline
    pipeline, elements = build_pipeline()

    # Attach the probes
    tracker_src_pad = elements["object_tracker"].get_static_pad("src")
    tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_src_pad_buffer_probe, 0)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus_message(bus, msg):
        msg_type = msg.type
        match msg_type:
            case Gst.MessageType.EOS:
                print(c.EOS_MESSAGE)
                loop.quit()
            case Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                if "Internal data stream error" in err.message:
                    print(c.EOS_MESSAGE)
                else:
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
                    old, new, pending = msg.parse_state_changed()
                    print(
                        f"{c.PIPELINE_STATE_PREFIX}: {old.value_nick} -> {new.value_nick}"
                    )

    bus.connect(c.BUS_MESSAGE_SIGNAL, on_bus_message)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("Interrupt caught, stopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
