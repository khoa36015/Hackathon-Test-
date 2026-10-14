"""
Data models for Human Detection, YOLO11-Pose, ByteTrack, Behavior Analytics, and Audit Logging.
Adheres to Strict Structured Output rules of Urban Safety AI-SOC.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class BehaviorType(str, Enum):
    NORMAL = "NORMAL"
    CROUCHING = "CROUCHING"
    LOITERING = "LOITERING"
    HANDS_UP = "HANDS_UP"
    RUNNING = "RUNNING"
    FIGHTING = "FIGHTING"
    FALLEN = "FALLEN"


# =============================================================================
# Modern COCO-17 & Behavior Recognition Schemas
# =============================================================================

@dataclass
class KeypointCOCO:
    """Represents a single anatomical landmark detected via YOLO11-Pose (COCO-17 format)."""
    id: int               # 0 to 16
    name: str             # "nose", "left_shoulder", etc.
    x: float              # Pixel X coordinate
    y: float              # Pixel Y coordinate
    confidence: float     # Confidence/visibility score (0.0 to 1.0)


@dataclass
class BehaviorPrediction:
    """Action & behavior prediction output for one tracked human subject."""
    primary_behavior: BehaviorType
    confidence: float
    threat_level: ThreatLevel
    threat_score: float   # 0.00 to 1.00
    reasoning: str        # Auditability / Explainability string
    interacting_with_track_id: Optional[int] = None


@dataclass
class TrackHistoryState:
    """Snapshot of kinematics stored in temporal rolling window for one track."""
    timestamp: float
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    keypoints: List[KeypointCOCO]
    spine_angle: float
    wrist_velocity: float
    body_velocity: float


@dataclass
class TrackKinematicsTelemetry:
    """Live biomechanical telemetry used for diagnostic Alert Box Debug HUD."""
    track_id: int
    spine_angle: float
    aspect_ratio: float
    speed: float
    wrist_energy: float
    dwell_time: float
    confirmed_behavior: str
    threat_level: ThreatLevel
    threat_score: float


@dataclass
class BehaviorAuditLogEntry:
    """Formal audit record saved into JSONL and log file upon action detection."""
    event_id: str                      # EVT_YYYYMMDD_HHMMSS_TRACKID
    timestamp: str                     # ISO-8601: YYYY-MM-DD HH:MM:SS.mmm
    frame_index: int
    track_id: int
    event_type: str                    # STATE_TRANSITION | THREAT_ESCALATION | CRITICAL_ALERT | HEARTBEAT
    previous_behavior: Optional[str]
    current_behavior: str
    threat_level: str
    threat_score: float
    telemetry: Dict[str, float]
    reasoning: str
    dwell_time_seconds: float


@dataclass
class HumanPoseTrack:
    """Full pose, track identity, and behavioral analytics for one detected subject."""
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_conf: float
    keypoints: List[KeypointCOCO] = field(default_factory=list)
    behavior: BehaviorPrediction = field(default_factory=lambda: BehaviorPrediction(
        primary_behavior=BehaviorType.NORMAL,
        confidence=1.0,
        threat_level=ThreatLevel.LOW,
        threat_score=0.1,
        reasoning="Normal activity"
    ))
    dwell_time_seconds: float = 0.0

    @property
    def posture_label(self) -> str:
        return self.behavior.primary_behavior.value

    @property
    def confidence(self) -> float:
        return self.detection_conf


@dataclass
class SOCFrameAnalysisResult:
    """Complete analysis metadata for a single video frame in Urban Safety AI-SOC."""
    frame_index: int
    timestamp: str  # Formatted timestamp string: YYYY-MM-DD HH:MM:SS.mmm
    human_count: int
    humans: List[HumanPoseTrack] = field(default_factory=list)
    highest_threat_level: ThreatLevel = ThreatLevel.LOW
    highest_threat_score: float = 0.0
    critical_events: List[str] = field(default_factory=list)
    telemetry_list: List[TrackKinematicsTelemetry] = field(default_factory=list)
    recent_alerts: List[str] = field(default_factory=list)
    inference_time_ms: float = 0.0
    tracking_time_ms: float = 0.0
    behavior_time_ms: float = 0.0
    is_recording_evidence: bool = False

    @property
    def has_human(self) -> bool:
        return self.human_count > 0

    @property
    def requires_operator_attention(self) -> bool:
        return self.highest_threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)


# =============================================================================
# Legacy Schemas for Full Backward Compatibility
# =============================================================================

@dataclass
class Keypoint:
    """Legacy anatomical landmark (MediaPipe format)."""
    id: int
    name: str
    x: int
    y: int
    z: float
    visibility: float


@dataclass
class HumanPose:
    """Legacy pose estimation model."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    keypoints: List[Keypoint] = field(default_factory=list)
    posture_label: str = "Standing"


@dataclass
class FrameAnalysisResult:
    """Legacy frame analysis result."""
    frame_index: int
    timestamp: str
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
    primary_behavior: str = "NORMAL"
    threat_level: str = "LOW"
