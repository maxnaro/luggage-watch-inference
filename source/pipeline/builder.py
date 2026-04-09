import os
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore

from .. import constants as c


def _has_display_server() -> bool:
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def _select_sink_type(headless: bool) -> str:
    if headless:
        return c.FAKESINK
    if not _has_display_server():
        return c.FAKESINK
    return c.NVEGLGLESSINK


def build_pipeline(headless: bool = False, record_path: str | None = None):
    """Creates and links the GStreamer elements for the DeepStream pipeline.

    Args:
        headless: If True, use fakesink instead of nveglglessink (for evaluation).
        record_path: If set, encode the OSD output to an mp4 file at this path.
    """
    pipeline = Gst.Pipeline()

    source = Gst.ElementFactory.make(c.URI_DECODE_BIN, c.SOURCE_ELEMENT_NAME)
    muxer = Gst.ElementFactory.make(c.NVSTREAMMUX, c.MUXER_ELEMENT_NAME)
    primary_infer = Gst.ElementFactory.make(c.NVINFER, c.PRIMARY_INFER_ELEMENT_NAME)
    object_tracker = Gst.ElementFactory.make(c.NVTRACKER, c.TRACKER_ELEMENT_NAME)
    osd = Gst.ElementFactory.make(c.NVDSOSD, c.OSD_ELEMENT_NAME)
    sink_type = _select_sink_type(headless)
    sink = Gst.ElementFactory.make(sink_type, c.SINK_ELEMENT_NAME)
    if sink is None and sink_type != c.FAKESINK:
        sink_type = c.FAKESINK
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
    if record_path is not None:
        object_tracker.set_property("user-meta-pool-size", 64)

    osd.set_property(c.PROPERTY_PROCESS_MODE, c.OSD_PROCESS_MODE)

    if sink_type == c.FAKESINK:
        # Headless runs should not block on render clock synchronization.
        sink.set_property("sync", False)

    # Build recording elements (if requested)
    record_elements = []
    tee = None
    queue_display = None
    queue_record = None
    record_sink = None
    if record_path is not None:
        queue_record = Gst.ElementFactory.make(c.QUEUE, "queue_record")
        record_sink = Gst.ElementFactory.make("nvvideoencfilesinkbin", "record_sink")

        if not all([queue_record, record_sink]):
            raise RuntimeError(
                "Failed to create recording elements. Ensure DeepStream video encoder plugins are installed."
            )

        # Configure the DeepStream recording bin (queue -> convert -> encode -> mux -> file).
        record_sink.set_property("output-file", record_path)
        record_sink.set_property("container", 1)  # mp4
        record_sink.set_property("codec", 1)  # h264
        record_sink.set_property("bitrate", 4_000_000)
        record_sink.set_property("sync", False)

        record_elements = [queue_record, record_sink]

        # In non-headless mode, keep the normal display sink and record in parallel.
        if sink_type != c.FAKESINK:
            tee = Gst.ElementFactory.make("tee", "tee")
            queue_display = Gst.ElementFactory.make(c.QUEUE, "queue_display")
            if not all([tee, queue_display]):
                raise RuntimeError(
                    "Failed to create tee elements for display + recording."
                )
            record_elements = [tee, queue_display] + record_elements

    # Add elements to pipeline
    pipeline_elements = [
        source,
        muxer,
        queue0,
        primary_infer,
        queue1,
        object_tracker,
        queue2,
        osd,
        queue3,
    ]
    if record_path is None or sink_type != c.FAKESINK:
        pipeline_elements.append(sink)

    for element in pipeline_elements + record_elements:
        pipeline.add(element)

    # Link elements (source to muxer is linked dynamically via pad-added signal)
    muxer.link(queue0)
    queue0.link(primary_infer)
    primary_infer.link(queue1)
    queue1.link(object_tracker)
    object_tracker.link(queue2)
    queue2.link(osd)

    if record_path is not None:
        osd.link(queue3)

        if tee is None:
            # Headless + recording: route directly to the recording sink.
            queue3.link(queue_record)
            queue_record.link(record_sink)
        else:
            # Display + recording: split output via tee.
            queue3.link(tee)

            tee_display_pad = tee.request_pad_simple("src_%u")
            tee_record_pad = tee.request_pad_simple("src_%u")
            if tee_display_pad is None or tee_record_pad is None:
                raise RuntimeError(
                    "Failed to request tee source pads for recording pipeline."
                )

            if (
                tee_display_pad.link(queue_display.get_static_pad("sink"))
                != Gst.PadLinkReturn.OK
            ):
                raise RuntimeError("Failed to link tee display branch.")
            if (
                tee_record_pad.link(queue_record.get_static_pad("sink"))
                != Gst.PadLinkReturn.OK
            ):
                raise RuntimeError("Failed to link tee recording branch.")

            queue_display.link(sink)
            queue_record.link(record_sink)
    else:
        osd.link(queue3)
        queue3.link(sink)

    def on_pad_added(src, new_pad):
        sink_pad = muxer.get_static_pad(c.MUXER_SINK_PAD_NAME)
        if sink_pad is None:
            sink_pad = muxer.request_pad_simple(c.MUXER_SINK_PAD_NAME)
        new_pad.link(sink_pad)

    source.connect(c.PAD_ADDED_SIGNAL, on_pad_added)

    # Return a dictionary of elements so we can attach probes in app.py
    elements = {
        "primary_infer": primary_infer,
        "object_tracker": object_tracker,
        "sink_type": sink_type,
    }

    return pipeline, elements
