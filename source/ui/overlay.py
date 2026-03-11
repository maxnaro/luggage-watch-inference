import pyds

from constants import COLORS, UNTRACKED_OBJECT_ID, LUGGAGE_CLASS_ID


def update_osd_metadata(frame_meta, luggage_states: dict[int, str]):
    """
    Updates the on-screen display (OSD) metadata for a given frame to
    reflect the state of luggage and their owners.
    """
    obj = frame_meta.obj_meta_list
    while obj is not None:
        try:
            obj_meta = pyds.NvDsObjectMeta.cast(obj.data)
        except StopIteration:
            break

        if obj_meta.object_id != UNTRACKED_OBJECT_ID:
            text = f"{obj_meta.obj_label} ID:{obj_meta.object_id}"
            box_color = COLORS.get(obj_meta.obj_label, COLORS["default"])

            if obj_meta.class_id == LUGGAGE_CLASS_ID:
                state = luggage_states.get(obj_meta.object_id, "Attended")
                text += f" ({state})"
                box_color = COLORS.get(state, box_color)

            obj_meta.text_params.display_text = text
            obj_meta.rect_params.border_color.set(*box_color)
            obj_meta.rect_params.border_width = 3
            obj_meta.text_params.text_bg_clr.set(
                box_color[0], box_color[1], box_color[2], 0.4
            )
            obj_meta.text_params.set_bg_clr = 1

        try:
            obj = obj.next
        except StopIteration:
            break
