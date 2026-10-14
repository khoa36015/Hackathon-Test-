"""
YOLO11-Pose & Supervision ByteTrack Pipeline with Integrated Behavior Recognition.
Designed for Urban Safety AI-SOC real-time monitoring and threat scoring.
Enhanced with Anti-Flapping Box Stabilization, Telemetry Diagnostic Feed, and Alert Box Debug HUD.
"""

import os
import sys
import time
import math
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from models import (
    KeypointCOCO,
    ThreatLevel,
    BehaviorType,
    BehaviorPrediction,
    HumanPoseTrack,
    SOCFrameAnalysisResult,
    TrackKinematicsTelemetry
)
from behavior_detector import BehaviorEngine


# COCO-17 Keypoint Names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# COCO-17 Skeleton Connections (start_idx, end_idx)
COCO_SKELETON_PAIRS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left Arm
    (5, 7), (7, 9),
    # Right Arm
    (6, 8), (8, 10),
    # Left Leg
    (11, 13), (13, 15),
    # Right Leg
    (12, 14), (14, 16)
]

# Threat-aware Color Palettes (BGR format)
THREAT_COLORS = {
    ThreatLevel.LOW: {"box": (0, 220, 0), "bone": (0, 255, 127), "joint": (0, 255, 0), "label": (0, 180, 0)},
    ThreatLevel.MEDIUM: {"box": (0, 215, 255), "bone": (0, 215, 255), "joint": (0, 255, 255), "label": (0, 165, 255)},
    ThreatLevel.HIGH: {"box": (0, 120, 255), "bone": (0, 140, 255), "joint": (0, 165, 255), "label": (0, 100, 255)},
    ThreatLevel.CRITICAL: {"box": (0, 0, 255), "bone": (50, 50, 255), "joint": (0, 0, 255), "label": (0, 0, 220)}
}


def compute_xyxy_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two (x1, y1, x2, y2) bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class YOLOPoseDetector:
    """
    High-performance real-time human detector, pose estimator, and tracker.
    Combines YOLO11-Pose, Supervision ByteTrack, BehaviorEngine, and Diagnostic Debug HUD.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 45,            # Increased for ID persistence (Anti-Flapping)
        minimum_matching_threshold: float = 0.80, # Tighter association threshold
        fps: float = 20.0,
        output_dir: str = "output"
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.fps_target = fps
        self.output_dir = output_dir

        # Device selection: use CUDA if available, else CPU
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[YOLOPoseDetector] Initializing YOLO11-Pose on device: '{self.device}'...")

        self.model = None
        self.tracker = None
        self.use_fallback = False
        self.show_debug_box = True  # Toggleable diagnostic overlay

        self._init_yolo_and_bytetrack()

        # Initialize Behavior Recognition Engine
        self.behavior_engine = BehaviorEngine(
            fps=self.fps_target,
            enter_debounce_frames=8,
            exit_debounce_frames=12,
            bbox_ema_alpha=0.65,
            output_dir=self.output_dir
        )

        # FPS calculation state
        self._fps_history = []
        self._last_process_time = time.time()
        self.fps = 0.0

    def _init_yolo_and_bytetrack(self) -> None:
        """Loads Ultralytics YOLO11-Pose model and Supervision ByteTrack with stabilization."""
        try:
            from ultralytics import YOLO
            import supervision as sv

            # Load YOLO11-Pose model
            self.model = YOLO(self.model_path)
            try:
                self.model.to(self.device)
            except Exception as de:
                print(f"[YOLOPoseDetector] Device switch note: {de}, keeping default.")

            # Initialize ByteTrack (with anti-flapping parameters)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=self.track_activation_threshold,
                    lost_track_buffer=self.lost_track_buffer,
                    minimum_matching_threshold=self.minimum_matching_threshold,
                    frame_rate=int(self.fps_target)
                )
            print("[YOLOPoseDetector] YOLO11-Pose and Supervision ByteTrack (stabilized) loaded successfully.")

        except Exception as e:
            print(f"[YOLOPoseDetector] Warning: Could not initialize YOLO11 / ByteTrack ({e}). Operating in resilient fallback mode.")
            self.use_fallback = True

    def toggle_debug_box(self) -> bool:
        """Toggles visibility of the diagnostic Alert Box Debug HUD."""
        self.show_debug_box = not self.show_debug_box
        return self.show_debug_box

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0
    ) -> Tuple[SOCFrameAnalysisResult, np.ndarray]:
        """
        Processes a single BGR video frame:
        1. YOLO11-Pose inference
        2. ByteTrack multi-object tracking
        3. Anti-flapping temporal behavior recognition & action audit logging
        4. Render SOC HUD dashboard, indicators, and Alert Box Debug panel
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        height, width, _ = frame_bgr.shape

        raw_poses: List[HumanPoseTrack] = []
        t_infer = 0.0
        t_track = 0.0
        t_beh = 0.0

        if not self.use_fallback and self.model is not None and self.tracker is not None:
            import supervision as sv

            # 1. YOLO11-Pose Inference
            t_inf_start = time.time()
            results = self.model(
                frame_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )[0]
            t_infer = (time.time() - t_inf_start) * 1000.0

            # 2. Convert to Supervision Detections & ByteTrack
            t_trk_start = time.time()
            detections = sv.Detections.from_ultralytics(results)
            tracked_detections = self.tracker.update_with_detections(detections)
            t_track = (time.time() - t_trk_start) * 1000.0

            # 3. Associate Keypoints with Tracked Detections
            raw_kpts_data = None
            if hasattr(results, 'keypoints') and results.keypoints is not None and results.keypoints.data is not None:
                raw_kpts_data = results.keypoints.data.cpu().numpy()

            raw_boxes_xyxy = results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else np.empty((0, 4))

            for i, (xyxy, mask, confidence, class_id, tracker_id, data) in enumerate(tracked_detections):
                if tracker_id is None:
                    continue

                tx1, ty1, tx2, ty2 = [int(v) for v in xyxy]

                best_iou = -1.0
                best_kp_idx = -1
                for orig_idx, obox in enumerate(raw_boxes_xyxy):
                    iou = compute_xyxy_iou((tx1, ty1, tx2, ty2), tuple(obox))
                    if iou > best_iou:
                        best_iou = iou
                        best_kp_idx = orig_idx

                kps_list: List[KeypointCOCO] = []
                if best_kp_idx >= 0 and raw_kpts_data is not None and best_kp_idx < len(raw_kpts_data):
                    kps_arr = raw_kpts_data[best_kp_idx]
                    for kidx in range(min(17, len(kps_arr))):
                        kx, ky, kconf = kps_arr[kidx]
                        kname = COCO_KEYPOINT_NAMES[kidx] if kidx < len(COCO_KEYPOINT_NAMES) else f"kp_{kidx}"
                        kps_list.append(KeypointCOCO(
                            id=kidx,
                            name=kname,
                            x=float(kx),
                            y=float(ky),
                            confidence=float(kconf)
                        ))

                raw_poses.append(HumanPoseTrack(
                    track_id=int(tracker_id),
                    bbox=(tx1, ty1, tx2, ty2),
                    detection_conf=float(confidence) if confidence is not None else 0.85,
                    keypoints=kps_list
                ))

        # 4. Behavioral & Kinematics Analysis with Anti-Flapping & Audit Logger
        t_beh_start = time.time()
        tracked_poses, telemetry_list, recent_alerts = self.behavior_engine.update(
            raw_poses,
            timestamp=time.time(),
            frame_index=frame_index
        )
        t_beh = (time.time() - t_beh_start) * 1000.0

        # Determine highest threat level and critical events
        highest_threat = ThreatLevel.LOW
        highest_score = 0.0
        critical_events: List[str] = []

        threat_rank = {ThreatLevel.LOW: 0, ThreatLevel.MEDIUM: 1, ThreatLevel.HIGH: 2, ThreatLevel.CRITICAL: 3}

        for p in tracked_poses:
            if threat_rank[p.behavior.threat_level] > threat_rank[highest_threat]:
                highest_threat = p.behavior.threat_level
                highest_score = p.behavior.threat_score

            if p.behavior.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                critical_events.append(
                    f"ID #{p.track_id} [{p.behavior.primary_behavior.value}]: {p.behavior.reasoning}"
                )

        # Update FPS moving average
        curr_t = time.time()
        dt = curr_t - self._last_process_time
        self._last_process_time = curr_t
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_history.append(inst_fps)
            if len(self._fps_history) > 15:
                self._fps_history.pop(0)
            self.fps = sum(self._fps_history) / len(self._fps_history)

        analysis = SOCFrameAnalysisResult(
            frame_index=frame_index,
            timestamp=now_str,
            human_count=len(tracked_poses),
            humans=tracked_poses,
            highest_threat_level=highest_threat,
            highest_threat_score=highest_score,
            critical_events=critical_events,
            telemetry_list=telemetry_list,
            recent_alerts=recent_alerts,
            inference_time_ms=t_infer,
            tracking_time_ms=t_track,
            behavior_time_ms=t_beh
        )

        annotated_frame = self.draw_soc_hud_and_indicators(frame_bgr.copy(), analysis)

        return analysis, annotated_frame

    def draw_soc_hud_and_indicators(
        self,
        frame: np.ndarray,
        analysis: SOCFrameAnalysisResult
    ) -> np.ndarray:
        """Renders threat-aware skeleton bones, sleek boxes, behavior labels, top HUD, and Alert Box Debug HUD."""
        h, w, _ = frame.shape

        for human in analysis.humans:
            x1, y1, x2, y2 = human.bbox
            bw = x2 - x1
            bh = y2 - y1

            colors = THREAT_COLORS.get(human.behavior.threat_level, THREAT_COLORS[ThreatLevel.LOW])
            box_color = colors["box"]
            bone_color = colors["bone"]
            joint_color = colors["joint"]

            # 1. Draw COCO Skeleton Bones
            if human.keypoints:
                kp_dict = {kp.id: kp for kp in human.keypoints}
                for s_idx, e_idx in COCO_SKELETON_PAIRS:
                    if s_idx in kp_dict and e_idx in kp_dict:
                        kp1 = kp_dict[s_idx]
                        kp2 = kp_dict[e_idx]
                        if kp1.confidence > 0.25 and kp2.confidence > 0.25:
                            bone_len = math.hypot(kp1.x - kp2.x, kp1.y - kp2.y)
                            if bone_len < max(50.0, 0.85 * math.hypot(bw, bh)):
                                cv2.line(
                                    frame,
                                    (int(kp1.x), int(kp1.y)),
                                    (int(kp2.x), int(kp2.y)),
                                    bone_color,
                                    2,
                                    cv2.LINE_AA
                                )

                # 2. Draw Keypoint Joints
                for kp in human.keypoints:
                    if kp.confidence > 0.25:
                        cv2.circle(frame, (int(kp.x), int(kp.y)), 3, joint_color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (int(kp.x), int(kp.y)), 5, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Draw Corner-Accented Bounding Box (Stabilized via EMA)
            self._draw_corner_box(frame, x1, y1, x2, y2, box_color)

            # 4. Draw Behavior & Threat Tag Header
            b_label = human.behavior.primary_behavior.value
            t_score_pct = int(human.behavior.threat_score * 100)
            tag_text = f"ID #{human.track_id} | {b_label} [{t_score_pct}%]"
            if human.behavior.interacting_with_track_id:
                tag_text += f" vs #{human.behavior.interacting_with_track_id}"

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), (15, 15, 15), -1)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), box_color, 1)
            cv2.putText(
                frame,
                tag_text,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 5. Top HUD Dashboard
        self._draw_hud_dashboard(frame, analysis)

        # 6. Diagnostic Alert Box Debug Panel (Toggleable with 'd')
        if self.show_debug_box:
            self._draw_debug_alert_box(frame, analysis.telemetry_list, analysis.recent_alerts)

        return frame

    def _draw_corner_box(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
        """Draws a sleek boundary box with highlighted corner brackets."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        w = x2 - x1
        h = y2 - y1
        c_len = min(20, max(8, int(min(w, h) * 0.15)))
        t = 2

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), color, t)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), color, t)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), color, t)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), color, t)

    def _draw_hud_dashboard(self, frame: np.ndarray, analysis: SOCFrameAnalysisResult):
        """Draws top HUD bar displaying SOC Title, Visible Clock, Threat Badge, and Latencies."""
        w = frame.shape[1]

        # Top dark bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (12, 12, 16), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
        cv2.line(frame, (0, 44), (w, 44), (55, 55, 65), 1)

        # Title
        cv2.putText(
            frame,
            "URBAN SAFETY SOC | YOLO11-POSE + BYTETRACK + BEHAVIOR ENGINE",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Visible Clock Overlay (SOC Requirement)
        cv2.putText(
            frame,
            f"TIME: {analysis.timestamp}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Threat Alert Badge
        if analysis.highest_threat_level == ThreatLevel.CRITICAL:
            badge_text = f"[ CRITICAL: {analysis.critical_events[0] if analysis.critical_events else 'EMERGENCY'} ]"
            badge_color = (0, 0, 255)
        elif analysis.highest_threat_level == ThreatLevel.HIGH:
            badge_text = f"[ HIGH ALERT: {analysis.critical_events[0] if analysis.critical_events else 'SUSPICIOUS'} ]"
            badge_color = (0, 140, 255)
        elif analysis.highest_threat_level == ThreatLevel.MEDIUM:
            badge_text = f"[ ELEVATED: {analysis.human_count} PERSON(S) MONITORED ]"
            badge_color = (0, 215, 255)
        else:
            badge_text = f"[ NORMAL: {analysis.human_count} PERSON(S) CLEAR ]" if analysis.human_count > 0 else "[ CLEAR: NO ACTIVITY ]"
            badge_color = (0, 220, 0)

        if len(badge_text) > 48:
            badge_text = badge_text[:45] + "... ]"

        cv2.putText(
            frame,
            badge_text,
            (w - 550, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            badge_color,
            2,
            cv2.LINE_AA
        )

        # Latency & FPS metrics
        metrics = f"FPS: {self.fps:.1f} | INFER: {analysis.inference_time_ms:.0f}ms | BEH: {analysis.behavior_time_ms:.0f}ms"
        cv2.putText(
            frame,
            metrics,
            (w - 180, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # Recording Indicator
        if analysis.is_recording_evidence:
            cv2.circle(frame, (w - 20, 15), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (w - 50, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

    def _draw_debug_alert_box(
        self,
        frame: np.ndarray,
        telemetry_list: List[TrackKinematicsTelemetry],
        recent_alerts: List[str]
    ) -> None:
        """
        Renders a floating Diagnostic Alert Box HUD at the bottom-left corner of the stream.
        Shows per-subject biomechanical telemetry and latest action state transitions.
        """
        h, w, _ = frame.shape
        box_w = 410
        box_h = 165
        box_x = 12
        box_y = h - box_h - 12

        # 1. Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (8, 8, 12), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (65, 65, 75), 1)

        # Header Title
        cv2.putText(
            frame,
            "[ DIAGNOSTIC TELEMETRY - TOGGLE 'd' ]",
            (box_x + 10, box_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.line(frame, (box_x + 5, box_y + 24), (box_x + box_w - 5, box_y + 24), (55, 55, 65), 1)

        # 2. Per-Track Telemetry Rows (up to 4 tracks)
        line_y = box_y + 40
        if not telemetry_list:
            cv2.putText(
                frame,
                "No subjects in field of view.",
                (box_x + 12, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (140, 140, 140),
                1,
                cv2.LINE_AA
            )
        else:
            for t in telemetry_list[:4]:
                color = (0, 0, 255) if t.threat_level == ThreatLevel.CRITICAL else (
                    (0, 140, 255) if t.threat_level == ThreatLevel.HIGH else (
                        (0, 215, 255) if t.threat_level == ThreatLevel.MEDIUM else (0, 220, 0)
                    )
                )
                row_str = (
                    f"ID#{t.track_id}: Ang={t.spine_angle:.0f}° | Spd={t.speed:.1f} | "
                    f"H/W={t.aspect_ratio:.1f} | {t.confirmed_behavior}"
                )
                cv2.putText(
                    frame,
                    row_str,
                    (box_x + 10, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    color,
                    1,
                    cv2.LINE_AA
                )
                line_y += 18

        # 3. Live Alert Ticker at bottom
        ticker_y = box_y + box_h - 10
        cv2.line(frame, (box_x + 5, ticker_y - 14), (box_x + box_w - 5, ticker_y - 14), (55, 55, 65), 1)

        latest_msg = recent_alerts[0] if recent_alerts else "STATUS: ALL MONITORED SUBJECTS STABLE"
        if len(latest_msg) > 52:
            latest_msg = latest_msg[:49] + "..."

        ticker_color = (0, 0, 255) if "CRITICAL" in latest_msg else (0, 215, 255) if "HIGH" in latest_msg else (0, 220, 0)
        cv2.putText(
            frame,
            f"ALERT: {latest_msg}",
            (box_x + 10, ticker_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            ticker_color,
            1,
            cv2.LINE_AA
        )

    def close(self) -> None:
        """Safely cleans up AI model, tracker, and audit log resources."""
        self.model = None
        self.tracker = None
        if hasattr(self, 'behavior_engine'):
            self.behavior_engine.close()
