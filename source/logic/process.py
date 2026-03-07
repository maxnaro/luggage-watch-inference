from logic.helpers.bbox import BBox
from logic.state_machine.luggage_context import LuggageContext

_contexts: dict[int, LuggageContext] = {}  # luggage_id -> context


def process_frame(persons: list, luggage_items: list) -> dict[int, str]:
    """
    Evaluates the spatial relationship between persons and luggage.
    Updates the state of each luggage item (Attended -> Unattended -> Abandoned).
    Returns {luggage_id: state_name}.
    """
    person_bboxes: dict[int, BBox] = {}
    for person in persons:
        rect = person.rect_params
        person_bboxes[person.object_id] = BBox(
            rect.left, rect.top, rect.width, rect.height
        )

    seen_ids: set[int] = set()
    results: dict[int, str] = {}  # luggage_id -> state_name

    for luggage in luggage_items:
        luggage_id = luggage.object_id
        seen_ids.add(luggage_id)
        rect = luggage.rect_params
        luggage_bbox = BBox(rect.left, rect.top, rect.width, rect.height)

        if luggage_id not in _contexts:
            _contexts[luggage_id] = LuggageContext(luggage_id)

        results[luggage_id] = _contexts[luggage_id].update(luggage_bbox, person_bboxes)

    for missing_id in set(_contexts) - seen_ids:
        del _contexts[missing_id]

    return results
