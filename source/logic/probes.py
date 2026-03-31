import pyds
import gi

from ui.overlay import update_osd_metadata
from logic.process import process_frame
from logic.helpers.mot_writer import MotWriter
import constants as c

gi.require_version(c.GSTREAMER_PACKAGE, c.GSTREAMER_VERSION)
from gi.repository import Gst  # type:ignore


def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    Reads the Tracking IDs and runs the Abandonment Logic.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    probe_data = u_data if isinstance(u_data, dict) else {}
    writer = probe_data.get("writer")
    draw_overlay = probe_data.get("draw_overlay", True)

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

                if isinstance(writer, MotWriter):
                    _write_mot_row(writer, frame_meta, obj_meta)

            try:
                obj = obj.next
            except StopIteration:
                break

        luggage_info = process_frame(persons, luggage_items)

        if draw_overlay:
            update_osd_metadata(batch_meta, frame_meta, persons, luggage_info)

        try:
            frame = frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def _write_mot_row(writer: MotWriter, frame_meta, obj_meta) -> None:
    if obj_meta.class_id not in {c.PERSON_CLASS_ID, c.LUGGAGE_CLASS_ID}:
        return

    confidence = float(obj_meta.confidence)
    if confidence < 0:
        confidence = 1.0

    rect = obj_meta.rect_params
    left = max(0.0, float(rect.left))
    top = max(0.0, float(rect.top))
    width = max(0.0, float(rect.width))
    height = max(0.0, float(rect.height))

    # nvinfer runs on nvstreammux resolution; remap to source frame size so
    # CVAT overlays align with the original video dimensions.
    src_w = int(getattr(frame_meta, "source_frame_width", 0) or 0)
    src_h = int(getattr(frame_meta, "source_frame_height", 0) or 0)
    if src_w > 0 and src_h > 0 and c.MUXER_WIDTH > 0 and c.MUXER_HEIGHT > 0:
        scale_x = float(src_w) / float(c.MUXER_WIDTH)
        scale_y = float(src_h) / float(c.MUXER_HEIGHT)
        left *= scale_x
        top *= scale_y
        width *= scale_x
        height *= scale_y

        # Clamp to source frame bounds.
        left = min(max(0.0, left), float(src_w))
        top = min(max(0.0, top), float(src_h))
        width = min(max(0.0, width), max(0.0, float(src_w) - left))
        height = min(max(0.0, height), max(0.0, float(src_h) - top))

    writer.write(
        frame_number=int(frame_meta.frame_num),
        track_id=int(obj_meta.object_id),
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=confidence,
        class_id=int(obj_meta.class_id),
        visibility=1.0,
    )
