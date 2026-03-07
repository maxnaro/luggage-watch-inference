from __future__ import annotations

from dataclasses import dataclass, field

from logic.state_machine.states import Attended, Unattended, Abandoned, State
from logic.helpers.bbox import BBox
from constants import (
    MOVEMENT_THRESHOLD_PX,
    OWNER_RADIUS_PX,
)  # TODO: replace these with some sort of config or args


@dataclass
class LuggageContext:
    """Context for evaluating the state of luggage"""

    id: int
    owner_id: int | None = None
    state: State = field(default_factory=Attended)
    last_bbox: BBox | None = None

    def transition_to(self, new_state: State) -> None:
        self.state = new_state

    def update(self, luggage_bbox: BBox, person_bboxes: dict[int, BBox]) -> str:
        """
        Called once per frame for a given luggage item.
        Returns the name of the current state after evaluation.
        """
        is_moving = True
        if self.last_bbox is not None:
            is_moving = (
                luggage_bbox.distance_to(self.last_bbox) >= MOVEMENT_THRESHOLD_PX
            )
        self.last_bbox = luggage_bbox

        is_attended = False
        if self.owner_id is not None and self.owner_id in person_bboxes:
            is_attended = (
                luggage_bbox.distance_to(person_bboxes[self.owner_id])
                <= OWNER_RADIUS_PX
            )
        elif self.owner_id is None:
            self.owner_id = self._get_nearest_person_id(luggage_bbox, person_bboxes)
            is_attended = self.owner_id is not None

        self.state.evaluate(self, is_attended, is_moving)
        return self.state.name

    @staticmethod
    def _get_nearest_person_id(
        luggage_bbox: BBox, person_bboxes: dict[int, BBox]
    ) -> int | None:
        nearest_id = None
        nearest_distance = float("inf")
        for person_id, person_bbox in person_bboxes.items():
            distance = luggage_bbox.distance_to(person_bbox)
            if distance < nearest_distance and distance <= OWNER_RADIUS_PX:
                nearest_distance = distance
                nearest_id = person_id
        return nearest_id
