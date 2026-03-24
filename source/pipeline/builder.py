import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

import constants as c


def build_pipeline(headless: bool = False):
    """Creates and links the GStreamer elements for the DeepStream pipeline.

    Args:
        headless: If True, use fakesink instead of nveglglessink (for evaluation).
    """
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make(c.URI_DECODE_BIN, c.SOURCE_ELEMENT_NAME)
    muxer = Gst.ElementFactory.make(c.NVSTREAMMUX, c.MUXER_ELEMENT_NAME)
    primary_infer = Gst.ElementFactory.make(c.NVINFER, c.PRIMARY_INFER_ELEMENT_NAME)
    object_tracker = Gst.ElementFactory.make(c.NVTRACKER, c.TRACKER_ELEMENT_NAME)
    osd = Gst.ElementFactory.make(c.NVDSOSD, c.OSD_ELEMENT_NAME)
    sink_type = c.FAKESINK if headless else c.NVEGLGLESSINK
    sink = Gst.ElementFactory.make(sink_type, c.SINK_ELEMENT_NAME)

    queue0 = Gst.ElementFactory.make(c.QUEUE, c.QUEUE + "0")
    queue1 = Gst.ElementFactory.make(c.QUEUE, c.QUEUE + "1")
    queue2 = Gst.ElementFactory.make(c.QUEUE, c.QUEUE + "2")
    queue3 = Gst.ElementFactory.make(c.QUEUE, c.QUEUE + "3")

    if not all(
        [
            source,
            muxer,
            queue0,
            primary_infer,
            queue1,
            object_tracker,
            queue2,
            osd,
            queue3,
            sink,
        ]
    ):
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
    object_tracker.set_property(c.PROPERTY_TRACKER_WIDTH, c.TRACKER_WIDTH)
    object_tracker.set_property(c.PROPERTY_TRACKER_HEIGHT, c.TRACKER_HEIGHT)

    osd.set_property(c.PROPERTY_PROCESS_MODE, c.OSD_PROCESS_MODE)

    # Add elements to pipeline
    for element in [
        source,
        muxer,
        queue0,
        primary_infer,
        queue1,
        object_tracker,
        queue2,
        osd,
        queue3,
        sink,
    ]:
        pipeline.add(element)

    # Link elements (source to muxer is linked dynamically via pad-added signal)
    muxer.link(queue0)
    queue0.link(primary_infer)
    primary_infer.link(queue1)
    queue1.link(object_tracker)
    object_tracker.link(queue2)
    queue2.link(osd)
    osd.link(queue3)
    queue3.link(sink)

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
