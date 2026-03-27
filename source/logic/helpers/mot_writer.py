import csv
from pathlib import Path


class MotWriter:
    """Writes MOT Challenge tracking rows to a file."""

    def __init__(self, output_path: Path):
        self._output_path = output_path
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)

    @property
    def output_path(self) -> Path:
        return self._output_path

    def write(
        self,
        frame_number: int,
        track_id: int,
        left: float,
        top: float,
        width: float,
        height: float,
        confidence: float,
        class_id: int,
        visibility: float,
    ) -> None:
        self._writer.writerow(
            [
                frame_number,
                track_id,
                round(left, 2),
                round(top, 2),
                round(width, 2),
                round(height, 2),
                round(confidence, 4),
                class_id,
                round(visibility, 3),
            ]
        )

    def close(self) -> None:
        self._fh.close()
