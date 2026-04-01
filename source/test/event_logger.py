from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AbandonmentEvent:
    """A single detected abandonment event."""

    luggage_id: int
    frame_num: int
    bbox: tuple[float, float, float, float]  # x, y, w, h


class EventLogger:
    """
    Tracks per-frame luggage states and records the first frame
    where each luggage item transitions to 'Abandoned'.

    Usage:
        logger = EventLogger()
        # In the probe callback, after process_frame():
        logger.log_frame(frame_num, luggage_info, luggage_bboxes)
        # After pipeline EOS:
        events = logger.events
    """

    def __init__(self):
        self._prev_states: dict[int, str] = {}
        self.events: list[AbandonmentEvent] = []
        self._abandoned_ids: set[int] = set()

    def reset(self) -> None:
        self._prev_states.clear()
        self.events.clear()
        self._abandoned_ids.clear()

    def log_frame(
        self,
        frame_num: int,
        luggage_info: dict[int, dict[str, str | int | None]],
        luggage_bboxes: dict[int, tuple[float, float, float, float]],
    ) -> None:
        """
        Call once per frame with the output of process_frame() and the
        bounding boxes of all tracked luggage items.

        Args:
            frame_num: The current frame number.
            luggage_info: {luggage_id: {"state": str, "owner_id": ...}}
            luggage_bboxes: {luggage_id: (x, y, w, h)}
        """
        for luggage_id, info in luggage_info.items():
            state = info["state"]
            prev = self._prev_states.get(luggage_id)

            if (
                state == "Abandoned"
                and prev != "Abandoned"
                and luggage_id not in self._abandoned_ids
            ):
                bbox = luggage_bboxes.get(luggage_id, (0, 0, 0, 0))
                self.events.append(AbandonmentEvent(luggage_id, frame_num, bbox))
                self._abandoned_ids.add(luggage_id)

            self._prev_states[luggage_id] = state
