from __future__ import annotations

import pyds
import gi

from ..ui.overlay import update_osd_metadata
from .process import process_frame
from .. import constants as c

gi.require_version(c.GSTREAMER_PACKAGE, c.GSTREAMER_VERSION)
from gi.repository import Gst  # type:ignore

# Optional event logger for evaluation — set externally before running the pipeline.
_event_logger = None


def set_event_logger(logger) -> None:
    """Attach an EventLogger instance for evaluation runs."""
    global _event_logger
    _event_logger = logger


def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    Reads the Tracking IDs and runs the Abandonment Logic.
    """
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

        luggage_info = process_frame(persons, luggage_items)

        if _event_logger is not None:
            luggage_bboxes = {}
            for item in luggage_items:
                r = item.rect_params
                luggage_bboxes[item.object_id] = (r.left, r.top, r.width, r.height)
            _event_logger.log_frame(frame_meta.frame_num, luggage_info, luggage_bboxes)

        update_osd_metadata(batch_meta, frame_meta, persons, luggage_info)

        try:
            frame = frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
