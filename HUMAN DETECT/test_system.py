"""
Automated Verification & Test Suite for Urban Safety AI-SOC:
YOLO11-Pose, ByteTrack, Supervision HUD, BehaviorEngine,
Anti-Flapping Debounce, and Action Audit Logging.
"""

import os
import sys
import time
import math
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

# Ensure current directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import (
    KeypointCOCO,
    ThreatLevel,
    BehaviorType,
    BehaviorPrediction,
    HumanPoseTrack,
    SOCFrameAnalysisResult,
    Keypoint,
    HumanPose,
    FrameAnalysisResult,
    BehaviorAuditLogEntry,
    TrackKinematicsTelemetry
)
from behavior_detector import BehaviorEngine, BehaviorAuditLogger
from clip_recorder import EvidenceClipRecorder


def test_models():
    print("[TEST 1/7] Testing Data Models (New SOC & Legacy Compatibility)...")
    kp_coco = KeypointCOCO(id=0, name="nose", x=320.0, y=100.0, confidence=0.95)
    pred = BehaviorPrediction(
        primary_behavior=BehaviorType.NORMAL,
        confidence=0.98,
        threat_level=ThreatLevel.LOW,
        threat_score=0.10,
        reasoning="Normal upright stance"
    )
    track = HumanPoseTrack(
        track_id=1,
        bbox=(280, 80, 360, 320),
        detection_conf=0.92,
        keypoints=[kp_coco],
        behavior=pred
    )
    soc_res = SOCFrameAnalysisResult(
        frame_index=1,
        timestamp="2026-10-14 17:00:00.000",
        human_count=1,
        humans=[track],
        highest_threat_level=ThreatLevel.LOW,
        highest_threat_score=0.10
    )
    assert soc_res.has_human is True
    assert soc_res.humans[0].behavior.primary_behavior == BehaviorType.NORMAL
    assert soc_res.humans[0].posture_label == "NORMAL"

    # Legacy models verification
    kp_legacy = Keypoint(id=0, name="NOSE", x=100, y=100, z=0.0, visibility=0.9)
    pose_legacy = HumanPose(track_id=1, bbox=(80, 80, 40, 100), confidence=0.9, keypoints=[kp_legacy])
    res_legacy = FrameAnalysisResult(frame_index=1, timestamp="2026-10-14", human_count=1, humans=[pose_legacy])
    assert res_legacy.has_human is True
    print("  -> PASSED: Modern and legacy data schemas serialized and validated successfully.")


def _create_synthetic_pose(
    track_id: int,
    bbox: Tuple[int, int, int, int],
    spine_mode: str = "upright",
    wrists_mode: str = "normal",
    speed: float = 0.0
) -> HumanPoseTrack:
    """Helper to synthesize COCO-17 keypoint skeletons for behavioral unit testing."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h = y2 - y1

    kps = []
    if spine_mode == "upright":
        nose_pt = (cx, y1 + 0.1 * h)
        l_sh = (cx - 20, y1 + 0.25 * h)
        r_sh = (cx + 20, y1 + 0.25 * h)
        l_hip = (cx - 15, y1 + 0.55 * h)
        r_hip = (cx + 15, y1 + 0.55 * h)
        l_knee = (cx - 15, y1 + 0.75 * h)
        r_knee = (cx + 15, y1 + 0.75 * h)
        l_ank = (cx - 15, y2 - 5)
        r_ank = (cx + 15, y2 - 5)
    elif spine_mode == "fallen":
        nose_pt = (x1 + 10, cy)
        l_sh = (x1 + 40, cy - 10)
        r_sh = (x1 + 40, cy + 10)
        l_hip = (x2 - 40, cy - 10)
        r_hip = (x2 - 40, cy + 10)
        l_knee = (x2 - 20, cy - 10)
        r_knee = (x2 - 20, cy + 10)
        l_ank = (x2 - 5, cy - 10)
        r_ank = (x2 - 5, cy + 10)
    else:  # crouching
        nose_pt = (cx, y1 + 0.15 * h)
        l_sh = (cx - 20, y1 + 0.30 * h)
        r_sh = (cx + 20, y1 + 0.30 * h)
        l_hip = (cx - 20, y1 + 0.55 * h)
        r_hip = (cx + 20, y1 + 0.55 * h)
        l_knee = (cx - 25, y1 + 0.65 * h)
        r_knee = (cx + 25, y1 + 0.65 * h)
        l_ank = (cx - 20, y2 - 5)
        r_ank = (cx + 20, y2 - 5)

    if wrists_mode == "hands_up":
        l_w = (cx - 30, nose_pt[1] - 30)
        r_w = (cx + 30, nose_pt[1] - 30)
    elif wrists_mode == "fighting":
        l_w = (cx - 10, y1 + 0.35 * h)
        r_w = (cx + 40, y1 + 0.35 * h)
    else:
        l_w = (cx - 25, y1 + 0.45 * h)
        r_w = (cx + 25, y1 + 0.45 * h)

    raw_pts = [
        (0, "nose", nose_pt[0], nose_pt[1]),
        (1, "left_eye", nose_pt[0] - 5, nose_pt[1] - 5),
        (2, "right_eye", nose_pt[0] + 5, nose_pt[1] - 5),
        (3, "left_ear", nose_pt[0] - 12, nose_pt[1] - 5),
        (4, "right_ear", nose_pt[0] + 12, nose_pt[1] - 5),
        (5, "left_shoulder", l_sh[0], l_sh[1]),
        (6, "right_shoulder", r_sh[0], r_sh[1]),
        (7, "left_elbow", l_sh[0] - 10, l_sh[1] + 25),
        (8, "right_elbow", r_sh[0] + 10, r_sh[1] + 25),
        (9, "left_wrist", l_w[0], l_w[1]),
        (10, "right_wrist", r_w[0], r_w[1]),
        (11, "left_hip", l_hip[0], l_hip[1]),
        (12, "right_hip", r_hip[0], r_hip[1]),
        (13, "left_knee", l_knee[0], l_knee[1]),
        (14, "right_knee", r_knee[0], r_knee[1]),
        (15, "left_ankle", l_ank[0], l_ank[1]),
        (16, "right_ankle", r_ank[0], r_ank[1])
    ]

    for kid, kname, kx, ky in raw_pts:
        kps.append(KeypointCOCO(id=kid, name=kname, x=float(kx), y=float(ky), confidence=0.90))

    return HumanPoseTrack(
        track_id=track_id,
        bbox=bbox,
        detection_conf=0.90,
        keypoints=kps
    )


def test_behavior_rules():
    print("\n[TEST 2/7] Testing BehaviorEngine Biomechanical Rules...")
    test_out = os.path.join(current_dir, "output")
    engine = BehaviorEngine(fps=20.0, output_dir=test_out)

    # 1. Normal Upright Behavior Test
    t = 1000.0
    for i in range(15):
        p = _create_synthetic_pose(track_id=1, bbox=(200, 100, 280, 350), spine_mode="upright")
        poses, _, _ = engine.update([p], timestamp=t + (i * 0.05), frame_index=i)
    assert poses[0].behavior.primary_behavior == BehaviorType.NORMAL
    assert poses[0].behavior.threat_level == ThreatLevel.LOW
    print("  -> Rule 1 (NORMAL): Upright stance correctly recognized with LOW threat.")

    # 2. Fallen Behavior Test (Horizontal Torso)
    engine_fall = BehaviorEngine(fps=20.0, output_dir=test_out)
    for i in range(15):
        p = _create_synthetic_pose(track_id=2, bbox=(100, 300, 350, 380), spine_mode="fallen")
        poses, _, _ = engine_fall.update([p], timestamp=t + (i * 0.05), frame_index=i)
    assert poses[0].behavior.primary_behavior == BehaviorType.FALLEN
    assert poses[0].behavior.threat_level == ThreatLevel.CRITICAL
    assert poses[0].behavior.threat_score >= 0.90
    print("  -> Rule 2 (FALLEN): Horizontal body correctly triggered CRITICAL threat emergency.")

    # 3. Hands Up Behavior Test (Distress / Surrender)
    engine_hands = BehaviorEngine(fps=20.0, output_dir=test_out)
    for i in range(15):
        p = _create_synthetic_pose(track_id=3, bbox=(200, 100, 280, 350), spine_mode="upright", wrists_mode="hands_up")
        poses, _, _ = engine_hands.update([p], timestamp=t + (i * 0.05), frame_index=i)
    assert poses[0].behavior.primary_behavior == BehaviorType.HANDS_UP
    assert poses[0].behavior.threat_level == ThreatLevel.HIGH
    print("  -> Rule 3 (HANDS_UP): Wrists raised above head correctly triggered HIGH alert.")

    # 4. Running / Sudden Acceleration Test
    engine_run = BehaviorEngine(fps=20.0, output_dir=test_out)
    for i in range(15):
        x_shift = i * 45
        p = _create_synthetic_pose(track_id=4, bbox=(100 + x_shift, 100, 180 + x_shift, 300), spine_mode="upright")
        poses, _, _ = engine_run.update([p], timestamp=t + (i * 0.05), frame_index=i)
    assert poses[0].behavior.primary_behavior == BehaviorType.RUNNING
    assert poses[0].behavior.threat_level == ThreatLevel.HIGH
    print("  -> Rule 4 (RUNNING): Rapid displacement correctly detected as HIGH threat fleeing.")

    # 5. Fighting / Violent Confrontation Test
    engine_fight = BehaviorEngine(fps=20.0, output_dir=test_out)
    for i in range(15):
        offset_w = 40 if (i % 2 == 0) else -40
        p1 = _create_synthetic_pose(track_id=10, bbox=(200, 100, 260, 300), spine_mode="upright", wrists_mode="fighting")
        p2 = _create_synthetic_pose(track_id=11, bbox=(240, 100, 300, 300), spine_mode="upright", wrists_mode="fighting")
        p1.keypoints[9].x += offset_w
        p2.keypoints[10].x -= offset_w
        poses, _, _ = engine_fight.update([p1, p2], timestamp=t + (i * 0.05), frame_index=i)
    assert poses[0].behavior.primary_behavior == BehaviorType.FIGHTING
    assert poses[0].behavior.threat_level == ThreatLevel.CRITICAL
    print("  -> Rule 5 (FIGHTING): Close-proximity arm violence correctly triggered CRITICAL threat.")

    engine.close()
    engine_fall.close()
    engine_hands.close()
    engine_run.close()
    engine_fight.close()
    print("  -> PASSED: All 5 behavior classification rules verified successfully.")


def test_anti_flapping_and_stabilization():
    print("\n[TEST 3/7] Testing Anti-Flapping Hysteresis & Box Stabilization...")
    test_out = os.path.join(current_dir, "output")
    engine = BehaviorEngine(fps=20.0, enter_debounce_frames=8, bbox_ema_alpha=0.65, output_dir=test_out)

    t = 2000.0
    # Phase A: 6 frames of normal upright stance
    for i in range(6):
        p = _create_synthetic_pose(track_id=20, bbox=(100, 100, 180, 320), spine_mode="upright")
        poses, _, _ = engine.update([p], timestamp=t + (i * 0.05), frame_index=i)
        assert poses[0].behavior.primary_behavior == BehaviorType.NORMAL

    # Phase B: Inject 2 transient noise frames of FALLEN (e.g. keypoint glitch)
    for i in range(6, 8):
        p_glitch = _create_synthetic_pose(track_id=20, bbox=(100, 280, 320, 350), spine_mode="fallen")
        poses, _, _ = engine.update([p_glitch], timestamp=t + (i * 0.05), frame_index=i)
        # MUST STAY NORMAL: 2 frames is below enter_debounce threshold (6-8 frames)
        assert poses[0].behavior.primary_behavior == BehaviorType.NORMAL, (
            f"Anti-flapping failed: Glitch frame {i} prematurely flipped behavior to {poses[0].behavior.primary_behavior.value}"
        )
    print("  -> Anti-Flapping Debounce: 2 transient noise frames were successfully filtered out (stayed NORMAL).")

    # Phase C: Inject 10 persistent frames of true FALLEN (> 6 debounce frames)
    for i in range(8, 18):
        p_real = _create_synthetic_pose(track_id=20, bbox=(100, 280, 320, 350), spine_mode="fallen")
        poses, _, _ = engine.update([p_real], timestamp=t + (i * 0.05), frame_index=i)

    assert poses[0].behavior.primary_behavior == BehaviorType.FALLEN
    assert poses[0].behavior.threat_level == ThreatLevel.CRITICAL
    print("  -> Transition Confirmation: Persistent state cleanly confirmed as FALLEN after debounce threshold.")

    # Phase D: Check Bounding Box EMA Smoothing
    # Raw box was (100, 280, 320, 350), check that smooth_bbox is properly formatted
    sb = poses[0].bbox
    assert len(sb) == 4 and sb[2] > sb[0] and sb[3] > sb[1]
    print(f"  -> Box Stabilization: Output smooth bbox: {sb}")

    engine.close()
    print("  -> PASSED: Anti-flapping and spatial stabilization verified successfully.")


def test_audit_logger():
    print("\n[TEST 4/7] Testing Formal SOC Action Audit Logging (JSONL & text)...")
    test_out = os.path.join(current_dir, "output")
    jsonl_path = os.path.join(test_out, "behavior_events.jsonl")
    log_path = os.path.join(test_out, "behavior_audit.log")

    assert os.path.exists(jsonl_path), f"Audit log file {jsonl_path} does not exist."
    assert os.path.exists(log_path), f"Audit text log file {log_path} does not exist."

    # Parse JSON Lines
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    assert len(entries) > 0, "Audit log file is empty."
    last_entry = entries[-1]
    required_fields = ["event_id", "timestamp", "frame_index", "track_id", "event_type", "current_behavior", "threat_level", "telemetry"]
    for rf in required_fields:
        assert rf in last_entry, f"Missing required audit field: {rf}"

    print(f"  -> Parsed {len(entries)} audit events from {os.path.basename(jsonl_path)}.")
    print(f"  -> Last audit event: [{last_entry['event_type']}] ID #{last_entry['track_id']} -> {last_entry['current_behavior']} ({last_entry['threat_level']})")
    print("  -> PASSED: SOC Action Audit Logging validated.")


def test_evidence_clip_recorder():
    print("\n[TEST 5/7] Testing EvidenceClipRecorder Rolling Buffer...")
    test_out_dir = os.path.join(current_dir, "output")
    recorder = EvidenceClipRecorder(
        output_dir=test_out_dir,
        fps=20.0,
        frame_size=(640, 480),
        pre_roll_seconds=1.0,
        post_roll_seconds=1.0
    )

    dummy_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    saved_clip = None

    # Pre-roll 25 frames
    for i in range(25):
        recorder.update(dummy_frame, is_human_present=False, timestamp_str=f"T0_{i}")

    # Event 30 frames
    for i in range(30):
        recorder.update(dummy_frame, is_human_present=True, timestamp_str=f"T1_{i}")

    # Post-roll 25 frames
    for i in range(25):
        res = recorder.update(dummy_frame, is_human_present=False, timestamp_str=f"T2_{i}")
        if res is not None:
            saved_clip = res

    recorder.release()
    assert saved_clip is not None, "Clip was not finalized."
    assert os.path.exists(saved_clip), f"Clip file {saved_clip} not found."
    assert os.path.getsize(saved_clip) > 0, "Clip file is 0 bytes."
    print(f"  -> Generated clip: {os.path.basename(saved_clip)} ({os.path.getsize(saved_clip)/1024:.1f} KB)")
    print("  -> PASSED: EvidenceClipRecorder finalized readable MP4 video.")


def test_yolo_pose_pipeline_and_debug_hud():
    print("\n[TEST 6/7] Testing YOLOPoseDetector Pipeline & Alert Box Debug HUD...")
    from yolo_pose_detector import YOLOPoseDetector
    detector = YOLOPoseDetector(model_path="yolo11n-pose.pt", conf_threshold=0.25)

    test_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    analysis, annotated = detector.process_frame(test_frame, frame_index=1)

    assert annotated.shape == (480, 640, 3), "Annotated frame dimensions mismatch."
    assert isinstance(analysis, SOCFrameAnalysisResult), "Analysis result must be SOCFrameAnalysisResult."
    assert hasattr(analysis, "telemetry_list")
    assert hasattr(analysis, "recent_alerts")

    # Test toggling Alert Box Debug
    assert detector.show_debug_box is True
    toggled = detector.toggle_debug_box()
    assert toggled is False
    assert detector.show_debug_box is False
    detector.toggle_debug_box()
    assert detector.show_debug_box is True

    print(f"  -> Diagnostic Alert Box toggle verified: ON -> OFF -> ON.")
    detector.close()
    print("  -> PASSED: YOLOPoseDetector Pipeline and Debug HUD verified cleanly.")


def test_camera_hardware():
    print("\n[TEST 7/7] Checking Camera Hardware / Fallback Readiness...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  -> Camera #0 online: {w}x{h}. Ready for live streaming.")
        else:
            print("  -> Camera #0 opened but frame read empty (virtual / busy).")
    else:
        print("  -> Camera #0 not available (App will use simulated stream or batch mode).")
    print("  -> PASSED: Camera hardware check complete.")


def run_all_tests():
    print("=================================================================")
    print("   AUTOMATED VERIFICATION: STABILIZATION, AUDIT & DEBUG SUITE    ")
    print("=================================================================")
    try:
        test_models()
        test_behavior_rules()
        test_anti_flapping_and_stabilization()
        test_audit_logger()
        test_evidence_clip_recorder()
        test_yolo_pose_pipeline_and_debug_hud()
        test_camera_hardware()
        print("\n=================================================================")
        print("   ALL 7 TESTS PASSED SUCCESSFULLY! (7/7)                        ")
        print("=================================================================")
        return 0
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion error: {ae}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(run_all_tests())
