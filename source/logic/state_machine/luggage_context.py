from __future__ import annotations

import time
from dataclasses import dataclass, field

from .states import Attended, Unattended, Abandoned, State
from ..helpers.bbox import BBox
from ... import constants as c


@dataclass
class LuggageContext:
    """Context for evaluating the state of luggage"""

    id: int
    owner_id: int | None = None
    state: State = field(default_factory=Attended)
    last_bbox: BBox | None = None
    last_seen: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.monotonic)
    owner_last_bbox: BBox | None = None
    owner_missing_since: float | None = None
    owner_confirmed_once: bool = field(default=False)
    owner_observed_seconds: float = field(default=0.0)
    _last_unattended_elapsed_seconds: float = field(default=0.0, init=False, repr=False)
    _last_unattended_exit_at: float | None = field(default=None, init=False, repr=False)
    _owner_candidate_id: int | None = field(default=None, init=False, repr=False)
    _owner_candidate_frames: int = field(default=0, init=False, repr=False)
    _unattended_entry_frames: int = field(default=0, init=False, repr=False)
    _unattended_reset_frames: int = field(default=0, init=False, repr=False)
    _last_update_time_s: float | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _now(now_s: float | None) -> float:
        return time.monotonic() if now_s is None else now_s

    def transition_to(self, new_state: State, now_s: float | None = None) -> None:
        now = self._now(now_s)

        if isinstance(self.state, Unattended) and isinstance(new_state, Attended):
            self._last_unattended_elapsed_seconds = self.state.elapsed_time(now)
            self._last_unattended_exit_at = now

        if isinstance(new_state, Unattended) and self._should_resume_unattended_timer(now):
            resumed_entered_at = now - self._last_unattended_elapsed_seconds
            new_state = Unattended(entered_at=resumed_entered_at)

        if self.state.name != new_state.name:
            self._unattended_entry_frames = 0
            self._unattended_reset_frames = 0

            if self.state.name != "Unattended" and new_state.name == "Unattended":
                self._last_unattended_elapsed_seconds = 0.0
                self._last_unattended_exit_at = None

        self.state = new_state

    def _should_resume_unattended_timer(self, now: float) -> bool:
        if self._last_unattended_exit_at is None:
            return False

        if self._last_unattended_elapsed_seconds <= 0:
            return False

        return (
            now - self._last_unattended_exit_at
            <= c.UNATTENDED_TIMER_RESUME_WINDOW_SECONDS
        )

    @property
    def unattended_reset_confirmed(self) -> bool:
        return (
            self._unattended_reset_frames
            >= c.UNATTENDED_RESET_CONFIRM_FRAMES
        )

    @property
    def unattended_entry_confirmed(self) -> bool:
        return (
            self._unattended_entry_frames
            >= c.UNATTENDED_ENTRY_CONFIRM_FRAMES
        )

    def can_transition_to_abandoned(self, now_s: float | None = None) -> bool:
        if c.REQUIRE_OWNER_HISTORY_FOR_ABANDONED and not self.owner_confirmed_once:
            return False

        context_age = self._now(now_s) - self.created_at
        return context_age >= c.MIN_CONTEXT_AGE_FOR_ABANDONED_SECONDS

    def update(
        self,
        luggage_bbox: BBox,
        person_bboxes: dict[int, BBox],
        now_s: float | None = None,
    ) -> dict[str, str | int | None]:
        """
        Called once per frame for a given luggage item.
        Returns the name of the current state and `owner_id` after evaluation.
        """
        now = self._now(now_s)
        if self._last_update_time_s is None:
            delta_s = 0.0
        else:
            delta_s = max(0.0, now - self._last_update_time_s)
        self._last_update_time_s = now

        is_moving = True
        if self.last_bbox is not None:
            is_moving = (
                luggage_bbox.distance_to(self.last_bbox) >= c.MOVEMENT_THRESHOLD_PX
            )
        self.last_bbox = luggage_bbox

        is_attended = self._evaluate_owner_attendance(
            luggage_bbox,
            person_bboxes,
            now,
        )

        owner_visible = False
        if self.owner_id is not None and self.owner_id in person_bboxes:
            owner_visible = self._is_within_owner_range(
                luggage_bbox,
                person_bboxes[self.owner_id],
            )

        if owner_visible:
            self.owner_observed_seconds += delta_s
            if (
                self.owner_observed_seconds
                >= c.OWNER_CONFIRM_MIN_OBSERVED_SECONDS
            ):
                self.owner_confirmed_once = True

        if self.state.name == "Attended":
            if not is_attended and not is_moving:
                self._unattended_entry_frames += 1
            else:
                self._unattended_entry_frames = 0
            self._unattended_reset_frames = 0
        elif self.state.name == "Unattended":
            self._unattended_entry_frames = 0
            if is_attended:
                self._unattended_reset_frames += 1
            else:
                self._unattended_reset_frames = 0
        else:
            self._unattended_entry_frames = 0
            self._unattended_reset_frames = 0

        self.state.evaluate(self, is_attended, is_moving, now)
        return {"state": self.state.name, "owner_id": self.owner_id}

    def _evaluate_owner_attendance(
        self,
        luggage_bbox: BBox,
        person_bboxes: dict[int, BBox],
        now: float,
    ) -> bool:
        if self.owner_id is None:
            return self._attempt_initial_owner_assignment(
                luggage_bbox, person_bboxes
            )

        if self.owner_id in person_bboxes:
            owner_bbox = person_bboxes[self.owner_id]

            # Do not blindly trust the owner ID if its trajectory suddenly jumps.
            # A person ID flip can keep the same numeric ID but point to a
            # different person.
            if self._is_owner_consistent(owner_bbox):
                self.owner_last_bbox = owner_bbox
                self.owner_missing_since = None
                self._reset_owner_candidate()
                return self._is_within_owner_range(luggage_bbox, owner_bbox)

            if self.owner_missing_since is None:
                self.owner_missing_since = now

            if self._attempt_owner_reassignment(
                luggage_bbox, person_bboxes, now
            ):
                owner_bbox = person_bboxes[self.owner_id]
                return self._is_within_owner_range(luggage_bbox, owner_bbox)

            return (now - self.owner_missing_since) <= c.OWNER_LOST_GRACE_SECONDS

        # Keep owner fixed during brief tracker losses and only allow strict
        # reassociation to preserve ownership consistency.
        if self.owner_missing_since is None:
            self.owner_missing_since = now

        if self._attempt_owner_reassignment(
            luggage_bbox, person_bboxes, now
        ):
            owner_bbox = person_bboxes[self.owner_id]
            return self._is_within_owner_range(luggage_bbox, owner_bbox)

        return (now - self.owner_missing_since) <= c.OWNER_LOST_GRACE_SECONDS

    def _attempt_initial_owner_assignment(
        self, luggage_bbox: BBox, person_bboxes: dict[int, BBox]
    ) -> bool:
        candidate_id = self._get_best_owner_candidate(luggage_bbox, person_bboxes)
        if not self._confirm_owner_candidate(
            candidate_id, c.OWNER_ASSIGN_CONFIRM_FRAMES
        ):
            return False

        self.owner_id = candidate_id
        self.owner_last_bbox = person_bboxes[candidate_id]
        self.owner_missing_since = None
        self._reset_owner_candidate()
        return True

    def _attempt_owner_reassignment(
        self,
        luggage_bbox: BBox,
        person_bboxes: dict[int, BBox],
        now: float,
    ) -> bool:
        if not self._can_reassign_for_owner_id_flip(now):
            self._reset_owner_candidate()
            return False

        candidate_id = self._get_reassignment_candidate(
            luggage_bbox, person_bboxes
        )
        if not self._confirm_owner_candidate(
            candidate_id, c.OWNER_REASSIGN_CONFIRM_FRAMES
        ):
            return False

        self.owner_id = candidate_id
        self.owner_last_bbox = person_bboxes[candidate_id]
        self.owner_missing_since = None
        self._reset_owner_candidate()
        return True

    def _get_reassignment_candidate(
        self, luggage_bbox: BBox, person_bboxes: dict[int, BBox]
    ) -> int | None:
        if self.owner_last_bbox is None:
            return None

        nearest_id = None
        nearest_key: tuple[float, float, float] | None = None
        for person_id, person_bbox in person_bboxes.items():
            if not self._is_within_owner_range(luggage_bbox, person_bbox):
                continue

            owner_shift = person_bbox.distance_to(self.owner_last_bbox)
            if owner_shift > c.OWNER_ID_FLIP_MAX_SHIFT_PX:
                continue

            edge_distance = luggage_bbox.centre_to_box_distance(person_bbox)
            centre_distance = luggage_bbox.distance_to(person_bbox)
            key = (owner_shift, edge_distance, centre_distance)

            if nearest_key is None or key < nearest_key:
                nearest_key = key
                nearest_id = person_id

        return nearest_id

    def _can_reassign_for_owner_id_flip(self, now: float) -> bool:
        if self.owner_missing_since is None:
            return False

        return (
            now - self.owner_missing_since
            <= c.OWNER_ID_FLIP_REASSIGN_WINDOW_SECONDS
        )

    def _confirm_owner_candidate(
        self, candidate_id: int | None, required_frames: int
    ) -> bool:
        if candidate_id is None:
            self._reset_owner_candidate()
            return False

        if self._owner_candidate_id == candidate_id:
            self._owner_candidate_frames += 1
        else:
            self._owner_candidate_id = candidate_id
            self._owner_candidate_frames = 1

        return self._owner_candidate_frames >= required_frames

    def _reset_owner_candidate(self) -> None:
        self._owner_candidate_id = None
        self._owner_candidate_frames = 0

    def _is_owner_consistent(self, owner_bbox: BBox) -> bool:
        if self.owner_last_bbox is None:
            return True
        return (
            owner_bbox.distance_to(self.owner_last_bbox)
            <= c.OWNER_REASSIGN_MAX_SHIFT_PX
        )

    @staticmethod
    def _is_within_owner_range(luggage_bbox: BBox, person_bbox: BBox) -> bool:
        edge_distance = luggage_bbox.centre_to_box_distance(person_bbox)
        if edge_distance <= c.OWNER_EDGE_RADIUS_PX:
            return True
        return luggage_bbox.distance_to(person_bbox) <= c.OWNER_RADIUS_PX

    def _get_best_owner_candidate(
        self, luggage_bbox: BBox, person_bboxes: dict[int, BBox]
    ) -> int | None:
        nearest_id = None
        nearest_key: tuple[float, float] | None = None
        for person_id, person_bbox in person_bboxes.items():
            if not self._is_within_owner_range(luggage_bbox, person_bbox):
                continue

            edge_distance = luggage_bbox.centre_to_box_distance(person_bbox)
            centre_distance = luggage_bbox.distance_to(person_bbox)
            key = (edge_distance, centre_distance)
            if nearest_key is None or key < nearest_key:
                nearest_key = key
                nearest_id = person_id

        return nearest_id
