import pyds

from ..constants import (
    COLORS,
    UNTRACKED_OBJECT_ID,
    LUGGAGE_CLASS_ID,
    PERSON_CLASS_ID,
    DISPLAY_META_LIMIT,
    OWNER_RADIUS_PX
)
from ..logic.helpers.bbox import BBox


def update_osd_metadata(batch_meta, frame_meta, persons, luggage_info: dict[int, str]):
    """
    Updates the on-screen display (OSD) metadata for a given frame to
    reflect the state of luggage and their owners.
    """
    person_centres: dict[int, tuple[float, float]] = {
        person.object_id: BBox(person.rect_params).centre for person in persons
    }

    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_lines = 0
    display_meta.num_circles = 0

    obj = frame_meta.obj_meta_list
    while obj is not None:
        try:
            obj_meta: pyds.NvDsObjectMeta = pyds.NvDsObjectMeta.cast(obj.data)
        except StopIteration:
            break

        if obj_meta.object_id != UNTRACKED_OBJECT_ID:
            text = f"{obj_meta.obj_label} ID:{obj_meta.object_id} {obj_meta.confidence:.2f}"
            box_color = COLORS.get(obj_meta.obj_label, COLORS["default"])

            if obj_meta.class_id == LUGGAGE_CLASS_ID:
                info = luggage_info.get(
                    obj_meta.object_id, {"state": "Attended", "owner_id": None}
                )
                state = info.get("state")
                owner_id = info.get("owner_id")
                text += f" ({state})"
                box_color = COLORS.get(state, box_color)

                if state == "Attended" and owner_id in person_centres:
                    luggage_centre = BBox(obj_meta.rect_params).centre
                    person_centre = person_centres[owner_id]
                    display_meta = _draw_ownership_overlay(
                        display_meta, batch_meta, frame_meta, luggage_centre, person_centre
                    )

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
        
    if display_meta.num_lines > 0 or display_meta.num_circles > 0:
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def _draw_ownership_overlay(
    display_meta: pyds.NvDsDisplayMeta,
    batch_meta: pyds.NvDsBatchMeta,
    frame_meta: pyds.NvDsFrameMeta,
    luggage_centre: tuple[float, float],
    person_centre: tuple[float, float],
) -> pyds.NvDsDisplayMeta:
    if (
        display_meta.num_lines == DISPLAY_META_LIMIT
        or display_meta.num_circles == DISPLAY_META_LIMIT
    ):
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_lines = 0
        display_meta.num_circles = 0

    line = display_meta.line_params[display_meta.num_lines]
    line.x1, line.y1 = int(luggage_centre[0]), int(luggage_centre[1])
    line.x2, line.y2 = int(person_centre[0]), int(person_centre[1])
    line.line_width = 3
    line.line_color.set(0.0, 1.0, 0.0, 1.0) 
    display_meta.num_lines += 1

    circle = display_meta.circle_params[display_meta.num_circles]
    circle.xc, circle.yc = int(luggage_centre[0]), int(luggage_centre[1])
    circle.radius = int(OWNER_RADIUS_PX)
    circle.has_bg_color = 1
    circle.bg_color.set(0.0, 1.0, 0.0, 0.2) 
    display_meta.num_circles += 1
    
    return display_meta