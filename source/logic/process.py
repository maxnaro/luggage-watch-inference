import time
from typing import Callable

from .helpers.bbox import BBox
from .state_machine.luggage_context import LuggageContext
from .. import constants as c

_contexts: dict[int, LuggageContext] = {}  # luggage_id -> context
_debug_hook: Callable[[str, dict[str, object]], None] | None = None


def set_debug_hook(
    hook: Callable[[str, dict[str, object]], None] | None,
) -> None:
    """Set a debug callback for context lifecycle events."""
    global _debug_hook
    _debug_hook = hook


def _emit_debug(event: str, **payload: object) -> None:
    if _debug_hook is None:
        return
    try:
        _debug_hook(event, payload)
    except Exception:
        # Debug hooks must never interfere with pipeline execution.
        pass


def _context_spatial_tolerance(context: LuggageContext) -> float:
    if context.state.name == "Abandoned":
        return (
            c.SPATIAL_TOLERANCE_PX
            * c.ABANDONED_CONTEXT_SPATIAL_TOLERANCE_MULTIPLIER
        )
    return float(c.SPATIAL_TOLERANCE_PX)


def reset_contexts() -> None:
    """Clear all tracked luggage contexts."""
    _emit_debug("contexts_reset", active_contexts=len(_contexts))
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
            tolerance = _context_spatial_tolerance(context)
            if (
                distance < tolerance
                and distance < best_distance
            ):
                matched_id = old_id
                best_distance = distance

        if matched_id is not None:
            _contexts[luggage_id] = _contexts.pop(matched_id)
            _contexts[luggage_id].id = luggage_id
            _emit_debug(
                "context_relinked",
                old_id=matched_id,
                new_id=luggage_id,
                distance_px=round(best_distance, 2),
                tolerance_px=round(
                    _context_spatial_tolerance(_contexts[luggage_id]), 2
                ),
                state=_contexts[luggage_id].state.name,
            )
        else:
            _contexts[luggage_id] = LuggageContext(luggage_id)
            _emit_debug("context_created", luggage_id=luggage_id)

    _contexts[luggage_id].last_seen = time.monotonic()


def _handle_tracker_loss(seen_ids: set[int]) -> None:
    """
    Handles tracker loss by removing contexts for lost IDs when time has exceeded
    the grace period.
    """
    current_time = time.monotonic()
    for missing_id in set(_contexts) - seen_ids:
        context = _contexts[missing_id]
        ttl_seconds = c.CONTEXT_TTL_SECONDS
        if context.state.name == "Abandoned":
            ttl_seconds = max(ttl_seconds, c.ABANDONED_CONTEXT_TTL_SECONDS)

        if current_time - context.last_seen > ttl_seconds:
            _emit_debug(
                "context_expired",
                luggage_id=missing_id,
                state=context.state.name,
                age_s=round(current_time - context.last_seen, 3),
                ttl_s=round(float(ttl_seconds), 3),
            )
            del _contexts[missing_id]
        # else: continue with existing context in case ID reappears
