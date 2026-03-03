import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import constants as c


def build_pipeline():
    """Creates and links the GStreamer elements for the DeepStream pipeline."""
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make(c.URI_DECODE_BIN, c.SOURCE_ELEMENT_NAME)
    muxer = Gst.ElementFactory.make(c.NVSTREAMMUX, c.MUXER_ELEMENT_NAME)
    primary_infer = Gst.ElementFactory.make(c.NVINFER, c.PRIMARY_INFER_ELEMENT_NAME)
    object_tracker = Gst.ElementFactory.make(c.NVTRACKER, c.TRACKER_ELEMENT_NAME)
    osd = Gst.ElementFactory.make(c.NVDSOSD, c.OSD_ELEMENT_NAME)
    sink = Gst.ElementFactory.make(c.NVEGLGLESSINK, c.SINK_ELEMENT_NAME)

    if not all([source, muxer, primary_infer, object_tracker, osd, sink]):
        raise RuntimeError(
            "Failed to create GStreamer elements. Check DeepStream installation."
        )

    # Set properties
    source.set_property(c.PROPERTY_URI, c.SOURCE_URI)
    muxer.set_property(c.PROPERTY_WIDTH, c.MUXER_WIDTH)
    muxer.set_property(c.PROPERTY_HEIGHT, c.MUXER_HEIGHT)
    muxer.set_property(c.PROPERTY_BATCH_SIZE, c.MUXER_BATCH_SIZE)

    primary_infer.set_property(
        c.PROPERTY_CONFIG_FILE_PATH,
        os.path.join(c.CONFIG_DIR, c.PRIMARY_INFER_CONFIG_FILE),
    )
    object_tracker.set_property(c.PROPERTY_LL_LIB_FILE, c.TRACKER_LIB_FILE)
    object_tracker.set_property(
        c.PROPERTY_LL_CONFIG_FILE, os.path.join(c.CONFIG_DIR, c.TRACKER_CONFIG_FILE)
    )

    # Add elements to pipeline
    for element in [source, muxer, primary_infer, object_tracker, osd, sink]:
        pipeline.add(element)

    # Link elements (source to muxer is linked dynamically via pad-added signal)
    muxer.link(primary_infer)
    primary_infer.link(object_tracker)
    object_tracker.link(osd)
    osd.link(sink)

    def on_pad_added(src, new_pad):
        sink_pad = muxer.request_pad_simple(c.MUXER_SINK_PAD_NAME)
        new_pad.link(sink_pad)

    source.connect(c.PAD_ADDED_SIGNAL, on_pad_added)

    # Return a dictionary of elements so we can attach probes in app.py
    elements = {
        "primary_infer": primary_infer,
        "object_tracker": object_tracker,
    }

    return pipeline, elements
