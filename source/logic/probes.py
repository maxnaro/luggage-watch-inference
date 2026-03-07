import pyds
import gi

from logic.process import process_frame

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type:ignore
import constants as c


def tracker_src_pad_buffer_probe(pad, info, u_data):
    """
    Reads the Tracking IDs and runs the Abandonment Logic.
    """
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list

    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        persons = []
        luggage_items = []

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.object_id != c.UNTRACKED_OBJECT_ID:
                # obj_meta.obj_label will automatically be "person" or "luggage"
                obj_meta.text_params.display_text = (
                    f"{obj_meta.obj_label} ID:{obj_meta.object_id}"
                )

            # Collect for state machine
            if obj_meta.class_id == 0:
                persons.append(obj_meta)
            elif obj_meta.class_id == 1:
                luggage_items.append(obj_meta)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print(process_frame(persons, luggage_items))

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK
