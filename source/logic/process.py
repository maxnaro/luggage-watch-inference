import time

from .helpers.bbox import BBox
from .state_machine.luggage_context import LuggageContext
from ..constants import SPATIAL_TOLERANCE_PX, CONTEXT_TTL_SECONDS

_contexts: dict[int, LuggageContext] = {}  # luggage_id -> context


def reset_contexts() -> None:
    """Clear all tracked luggage contexts."""
    _contexts.clear()


def process_frame(
    persons: list, luggage_items: list
) -> dict[int, dict[str, str | int | None]]:
    """
    Evaluates the spatial relationship between persons and luggage.
    Updates the state of each luggage item (Attended -> Unattended -> Abandoned).
    Returns {luggage_id: {"state": state_name, "owner_id": owner_id}}.
    """
    person_bboxes: dict[int, BBox] = {}
    for person in persons:
        rect = person.rect_params
        person_bboxes[person.object_id] = BBox(rect)

    frame_ids: set[int] = {item.object_id for item in luggage_items}
    results: dict[int, dict[str, str | int | None]] = {}  # luggage_id -> {"state": state_name, "owner_id": owner_id}

    for luggage in luggage_items:
        luggage_id = luggage.object_id
        rect = luggage.rect_params
        luggage_bbox = BBox(rect)

        _handle_tracker_flip(luggage_id, luggage_bbox, frame_ids)

        results[luggage_id] = _contexts[luggage_id].update(luggage_bbox, person_bboxes)

    _handle_tracker_loss(frame_ids)

    return results


def _handle_tracker_flip(
    luggage_id: int, luggage_bbox: BBox, frame_ids: set[int]
) -> None:
    """
    Handles tracker ID flips by transferring context from the old ID to the new ID.
    """
    if luggage_id not in _contexts:
        matched_id = None
        best_distance = float("inf")
        for old_id, context in _contexts.items():
            if old_id == luggage_id or old_id in frame_ids:
                continue

            if context.last_bbox is None:
                continue

            distance = luggage_bbox.distance_to(context.last_bbox)
            if (
                distance < SPATIAL_TOLERANCE_PX
                and distance < best_distance
            ):
                matched_id = old_id
                best_distance = distance

        if matched_id is not None:
            _contexts[luggage_id] = _contexts.pop(matched_id)
            _contexts[luggage_id].id = luggage_id
        else:
            _contexts[luggage_id] = LuggageContext(luggage_id)

    _contexts[luggage_id].last_seen = time.monotonic()


def _handle_tracker_loss(seen_ids: set[int]) -> None:
    """
    Handles tracker loss by removing contexts for lost IDs when time has exceeded
    the grace period.
    """
    current_time = time.monotonic()
    for missing_id in set(_contexts) - seen_ids:
        if current_time - _contexts[missing_id].last_seen > CONTEXT_TTL_SECONDS:
            del _contexts[missing_id]
        # else: continue with existing context in case ID reappears
