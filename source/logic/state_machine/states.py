from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ... import constants as c

if TYPE_CHECKING:
    # HACK: avoid circular import for type hints, this is very ugly
    from .luggage_context import LuggageContext


class State(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, State):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    @abstractmethod
    def evaluate(
        self,
        context: LuggageContext,
        is_attended: bool,
        is_moving: bool,
        now_s: float | None = None,
    ) -> None:
        """Determines transitions and updates the context."""
        ...


class Attended(State):
    @property
    def name(self) -> str:
        return "Attended"

    def evaluate(
        self,
        context: LuggageContext,
        is_attended: bool,
        is_moving: bool,
        now_s: float | None = None,
    ) -> None:
        if context.unattended_entry_confirmed:
            context.transition_to(Unattended(now_s=now_s), now_s=now_s)


class Unattended(State):
    def __init__(self, entered_at: float | None = None, now_s: float | None = None):
        self._abandonment_timeout = c.ABANDONMENT_TIMEOUT_SECONDS
        now = time.monotonic() if now_s is None else now_s
        self._entered_at = now if entered_at is None else entered_at

    @property
    def name(self) -> str:
        return "Unattended"

    def elapsed_time(self, now_s: float | None = None) -> float:
        now = time.monotonic() if now_s is None else now_s
        return now - self._entered_at

    def evaluate(
        self,
        context: LuggageContext,
        is_attended: bool,
        is_moving: bool,
        now_s: float | None = None,
    ) -> None:
        abandonment_timeout = self._abandonment_timeout
        if not context.owner_confirmed_once:
            abandonment_timeout *= c.OWNERLESS_ABANDONMENT_TIMEOUT_MULTIPLIER

        if is_moving:
            context.transition_to(Attended(), now_s=now_s)
        elif context.unattended_reset_confirmed:
            context.transition_to(Attended(), now_s=now_s)
        elif (
            self.elapsed_time(now_s) >= abandonment_timeout
            and context.can_transition_to_abandoned(now_s)
        ):
            context.transition_to(Abandoned(), now_s=now_s)


class Abandoned(State):
    @property
    def name(self) -> str:
        return "Abandoned"

    def evaluate(
        self,
        context: LuggageContext,
        is_attended: bool,
        is_moving: bool,
        now_s: float | None = None,
    ) -> None:
        pass
