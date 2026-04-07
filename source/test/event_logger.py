from __future__ import annotations

from dataclasses import dataclass

from .. import constants as c


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
        self._recent_events: list[AbandonmentEvent] = []

    def reset(self) -> None:
        self._prev_states.clear()
        self.events.clear()
        self._abandoned_ids.clear()
        self._recent_events.clear()

    @staticmethod
    def _bbox_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax1, ay1 = box_a[0], box_a[1]
        ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
        bx1, by1 = box_b[0], box_b[1]
        bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_center_distance(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        ax = box_a[0] + box_a[2] / 2
        ay = box_a[1] + box_a[3] / 2
        bx = box_b[0] + box_b[2] / 2
        by = box_b[1] + box_b[3] / 2
        return ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

    @staticmethod
    def _bbox_area(box: tuple[float, float, float, float]) -> float:
        return max(0.0, box[2]) * max(0.0, box[3])

    @staticmethod
    def _bbox_area_ratio(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        area_a = EventLogger._bbox_area(box_a)
        area_b = EventLogger._bbox_area(box_b)
        min_area = min(area_a, area_b)
        if min_area <= 0.0:
            return float("inf")
        return max(area_a, area_b) / min_area

    def _is_duplicate_event(
        self,
        frame_num: int,
        bbox: tuple[float, float, float, float],
    ) -> bool:
        min_frame = frame_num - c.ABANDONED_EVENT_DEDUP_FRAMES
        self._recent_events = [
            event for event in self._recent_events if event.frame_num >= min_frame
        ]

        for event in self._recent_events:
            iou = self._bbox_iou(event.bbox, bbox)
            distance = self._bbox_center_distance(event.bbox, bbox)
            area_ratio = self._bbox_area_ratio(event.bbox, bbox)

            if area_ratio > c.ABANDONED_EVENT_DEDUP_MAX_AREA_RATIO:
                continue

            tight_distance_px = min(
                c.ABANDONED_EVENT_DEDUP_TIGHT_DISTANCE_PX,
                c.ABANDONED_EVENT_DEDUP_DISTANCE_PX,
            )

            if distance <= tight_distance_px:
                return True

            if (
                iou >= c.ABANDONED_EVENT_DEDUP_IOU
                and distance <= c.ABANDONED_EVENT_DEDUP_DISTANCE_PX
            ):
                return True
        return False

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
                event = AbandonmentEvent(luggage_id, frame_num, bbox)
                if not self._is_duplicate_event(frame_num, bbox):
                    self.events.append(event)
                    self._recent_events.append(event)
                self._abandoned_ids.add(luggage_id)

            self._prev_states[luggage_id] = state
