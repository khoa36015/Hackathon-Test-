"""
Data models for Human Detection and Pose Skeleton Analytics.
Adheres to Strict Structured Output rules of Urban Safety AI-SOC.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from datetime import datetime


@dataclass
class Keypoint:
    """Represents a single anatomical landmark detected on a human body."""
    id: int
    name: str
    x: int             # Pixel X coordinate
    y: int             # Pixel Y coordinate
    z: float           # Relative depth value
    visibility: float  # Confidence/visibility score (0.0 to 1.0)


@dataclass
class HumanPose:
    """Represents full pose estimation for one detected human subject."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    confidence: float
    keypoints: List[Keypoint] = field(default_factory=list)
    posture_label: str = "Standing"  # Standing, Walking, Sitting, Low Mobility


@dataclass
class FrameAnalysisResult:
    """Complete analysis metadata for a single video frame."""
    frame_index: int
    timestamp: str  # ISO-8601 UTC or local formatted
    human_count: int
    humans: List[HumanPose] = field(default_factory=list)
    inference_time_ms: float = 0.0
    is_recording_evidence: bool = False

    @property
    def has_human(self) -> bool:
        return self.human_count > 0


@dataclass
class HumanIncidentClip:
    """Metadata record for an extracted video clip containing human presence."""
    event_id: str
    source_video: str
    start_time: str
    end_time: str
    start_frame: int
    end_frame: int
    duration_seconds: float
    output_filepath: str
    max_confidence: float = 0.0

