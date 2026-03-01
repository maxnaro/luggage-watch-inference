import os
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

import pyds

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")


def main():
    Gst.init(None)

    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make(
        "uridecodebin", "source"
    )  # Use GstV4l2Src for live camera input
    muxer = Gst.ElementFactory.make("nvstreammux", "muxer")
    primary_infer = Gst.ElementFactory.make("nvinfer", "primary-infer")
    object_tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    # TODO: Metadata probe (abandonment logic)
    osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    # Set stream and muxer properties
    source.set_property("uri", "file:///~/source/AVSS_E2.avi")  # Test video
    muxer.set_property("width", 1280)
    muxer.set_property("height", 1280)
    muxer.set_property("batch-size", 1)

    # Set inference and tracker configurations
    primary_infer.set_property("config-file-path", os.path.join(CONFIG_DIR, "config_infer_primary.txt"))
    object_tracker.set_property(
        "ll-lib-file",
        "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
    )
    object_tracker.set_property("ll-config-file", os.path.join(CONFIG_DIR, "config_tracker_NvDCF_perf.yml"))

    for element in [source, muxer, primary_infer, object_tracker, osd, sink]:
        pipeline.add(element)

    muxer.link(primary_infer)
    primary_infer.link(object_tracker)
    object_tracker.link(osd)
    osd.link(sink)

    def on_pad_added(src, new_pad):
        sink_pad = muxer.get_request_pad("sink_0")
        new_pad.link(sink_pad)

    source.connect("pad-added", on_pad_added)

    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect(
        "message",
        lambda bus, msg: (
            loop.quit()
            if msg.type == Gst.MessageType.EOS or msg.type == Gst.MessageType.ERROR
            else None
        ),
    )
    loop.run()

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
