from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """Evaluation result for a single video."""

    video: str
    expected_abandonment: bool
    detected_abandonment: bool
    frame_error: int | None = None  # absolute frame difference (TP only)
    iou: float | None = None  # bounding box IoU (TP only)


@dataclass
class EvalSummary:
    """Aggregate metrics across all videos."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    frame_errors: list[int] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    results: list[EvalResult] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def mean_frame_error(self) -> float | None:
        return (
            sum(self.frame_errors) / len(self.frame_errors)
            if self.frame_errors
            else None
        )

    @property
    def mean_iou(self) -> float | None:
        return sum(self.ious) / len(self.ious) if self.ious else None


def compute_iou(box_a: tuple | list, box_b: tuple | list) -> float:
    """IoU between two (x, y, w, h) boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def _scale_bbox(
    bbox: tuple, muxer_w: int, muxer_h: int, video_w: int, video_h: int
) -> tuple[float, float, float, float]:
    """Scale a bbox from muxer resolution to original video resolution."""
    x, y, w, h = bbox
    sx = video_w / muxer_w
    sy = video_h / muxer_h
    return (x * sx, y * sy, w * sx, h * sy)


def evaluate_video(
    video_name: str,
    gt_entry: dict,
    events: list,
    temporal_tolerance: int = 90,
    muxer_size: tuple[int, int] = (640, 640),
    video_size: tuple[int, int] | None = None,
) -> EvalResult:
    """
    Evaluate pipeline output for a single video against ground truth.

    Args:
        video_name: Filename of the video.
        gt_entry: Ground truth dict with has_abandonment, true_abandon_frame, bag_roi.
        events: List of AbandonmentEvent from the EventLogger.
        temporal_tolerance: Allowed frame deviation for a true positive.
        muxer_size: (width, height) of the DeepStream muxer output.
        video_size: (width, height) of the original video. If provided,
            detected bboxes are scaled to match the ground truth coordinate space.
    """
    expected = gt_entry["has_abandonment"]
    detected = len(events) > 0

    if expected and detected:
        # Match the earliest detected event within temporal tolerance
        best_event = None
        best_error = float("inf")

        for event in events:
            error = abs(event.frame_num - gt_entry["true_abandon_frame"])
            if error < best_error:
                best_error = error
                best_event = event

        if best_error <= temporal_tolerance:
            detected_bbox = best_event.bbox
            if video_size is not None:
                detected_bbox = _scale_bbox(
                    detected_bbox, muxer_size[0], muxer_size[1],
                    video_size[0], video_size[1],
                )
            iou = compute_iou(detected_bbox, gt_entry["bag_roi"])
            return EvalResult(
                video=video_name,
                expected_abandonment=True,
                detected_abandonment=True,
                frame_error=int(best_error),
                iou=iou,
            )
        else:
            # Detected, but too far from the true frame — counts as FP + FN
            return EvalResult(
                video=video_name,
                expected_abandonment=True,
                detected_abandonment=True,
                frame_error=int(best_error),
            )

    return EvalResult(
        video=video_name,
        expected_abandonment=expected,
        detected_abandonment=detected,
    )


def summarise(results: list[EvalResult]) -> EvalSummary:
    """Aggregate per-video results into an EvalSummary."""
    summary = EvalSummary(results=results)

    for r in results:
        if r.expected_abandonment and r.frame_error is not None and r.iou is not None:
            summary.tp += 1
            summary.frame_errors.append(r.frame_error)
            summary.ious.append(r.iou)
        elif r.expected_abandonment and not r.detected_abandonment:
            summary.fn += 1
        elif not r.expected_abandonment and r.detected_abandonment:
            summary.fp += 1
        elif not r.expected_abandonment and not r.detected_abandonment:
            summary.tn += 1
        else:
            # Detected but outside tolerance (FP + FN)
            summary.fp += 1
            summary.fn += 1

    return summary
