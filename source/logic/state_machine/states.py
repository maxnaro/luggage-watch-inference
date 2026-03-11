from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from constants import (
    ABANDONMENT_TIMEOUT_SECONDS,
)  # TODO: replace this with some sort of config or args

if TYPE_CHECKING:
    # HACK: avoid circular import for type hints, this is very ugly
    from logic.state_machine.luggage_context import LuggageContext


class State(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    def __eq__(self, other: str):
        return self.name() == other

    @abstractmethod
    def evaluate(
        self, context: LuggageContext, is_attended: bool, is_moving: bool
    ) -> None:
        """Determines transitions and updates the context."""
        ...


class Attended(State):
    @property
    def name(self) -> str:
        return "Attended"

    def evaluate(
        self, context: LuggageContext, is_attended: bool, is_moving: bool
    ) -> None:
        if not is_attended and not is_moving:
            context.transition_to(Unattended())


class Unattended(State):
    def __init__(self):
        self._abandonment_timeout = ABANDONMENT_TIMEOUT_SECONDS
        self._entered_at: float = time.monotonic()

    @property
    def name(self) -> str:
        return "Unattended"

    @property
    def elapsed_time(self) -> float:
        return time.monotonic() - self._entered_at

    def evaluate(
        self, context: LuggageContext, is_attended: bool, is_moving: bool
    ) -> None:
        if is_attended or is_moving:
            context.transition_to(Attended())
        elif self.elapsed_time >= self._abandonment_timeout:
            context.transition_to(Abandoned())


class Abandoned(State):
    @property
    def name(self) -> str:
        return "Abandoned"

    def evaluate(
        self, context: LuggageContext, is_attended: bool, is_moving: bool
    ) -> None:
        pass
