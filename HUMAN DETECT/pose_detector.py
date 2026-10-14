"""
Pose Skeleton Detector Module for Urban Safety AI-SOC.
Performs real-time human detection and pose keypoint/skeleton rendering.
Utilizes modern MediaPipe PoseLandmarker (v1.1.0+ Task API) with automatic
OpenCV HOG fallback for maximum resilience.
"""

import os
import sys
import time
import math
import urllib.request
import cv2
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from models import Keypoint, HumanPose, FrameAnalysisResult


# Landmark names in MediaPipe Pose
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# Anatomical bone connections: (start_idx, end_idx, color_bgr)
SKELETON_CONNECTIONS = [
    # Torso (Cyan)
    (11, 12, (0, 255, 255)),   # Left Shoulder <-> Right Shoulder
    (11, 23, (0, 255, 255)),   # Left Shoulder <-> Left Hip
    (12, 24, (0, 255, 255)),   # Right Shoulder <-> Right Hip
    (23, 24, (0, 255, 255)),   # Left Hip <-> Right Hip

    # Left Arm (Lime Green)
    (11, 13, (0, 255, 128)),   # Left Shoulder <-> Left Elbow
    (13, 15, (0, 255, 128)),   # Left Elbow <-> Left Wrist
    (15, 17, (0, 255, 128)),   # Left Wrist <-> Left Pinky
    (15, 19, (0, 255, 128)),   # Left Wrist <-> Left Index

    # Right Arm (Orange)
    (12, 14, (0, 165, 255)),   # Right Shoulder <-> Right Elbow
    (14, 16, (0, 165, 255)),   # Right Elbow <-> Right Wrist
    (16, 18, (0, 165, 255)),   # Right Wrist <-> Right Pinky
    (16, 20, (0, 165, 255)),   # Right Wrist <-> Right Index

    # Left Leg (Magenta)
    (23, 25, (255, 0, 255)),   # Left Hip <-> Left Knee
    (25, 27, (255, 0, 255)),   # Left Knee <-> Left Ankle
    (27, 29, (255, 0, 255)),   # Left Ankle <-> Left Heel
    (29, 31, (255, 0, 255)),   # Left Heel <-> Left Foot Index

    # Right Leg (Sky Blue)
    (24, 26, (255, 191, 0)),   # Right Hip <-> Right Knee
    (26, 28, (255, 191, 0)),   # Right Knee <-> Right Ankle
    (28, 30, (255, 191, 0)),   # Right Ankle <-> Right Heel
    (30, 32, (255, 191, 0)),   # Right Heel <-> Right Foot Index

    # Face (Yellow)
    (0, 1, (0, 215, 255)),     # Nose <-> Left Eye Inner
    (1, 2, (0, 215, 255)),     # Left Eye Inner <-> Left Eye
    (2, 3, (0, 215, 255)),     # Left Eye <-> Left Eye Outer
    (3, 7, (0, 215, 255)),     # Left Eye Outer <-> Left Ear
    (0, 4, (0, 215, 255)),     # Nose <-> Right Eye Inner
    (4, 5, (0, 215, 255)),     # Right Eye Inner <-> Right Eye
    (5, 6, (0, 215, 255)),     # Right Eye <-> Right Eye Outer
    (6, 8, (0, 215, 255)),     # Right Eye Outer <-> Right Ear
    (9, 10, (0, 215, 255))     # Mouth Left <-> Mouth Right
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


class PoseSkeletonDetector:
    """
    High-performance real-time human detector and pose skeleton estimator.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._fps_history = []
        self._last_process_time = time.time()
        self.fps = 0.0

        self.landmarker = None
        self.mp = None
        self.use_hog_fallback = False

        self._init_mediapipe_landmarker()

    def _init_mediapipe_landmarker(self) -> None:
        """Initializes modern MediaPipe PoseLandmarker task with model auto-download."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp

            # Ensure model file exists locally
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(curr_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, "pose_landmarker_lite.task")

            if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
                print(f"[PoseSkeletonDetector] Downloading pose model from Google CDN...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print(f"[PoseSkeletonDetector] Model saved: {model_path} ({os.path.getsize(model_path)} bytes)")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
                num_poses=3
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            print("[PoseSkeletonDetector] MediaPipe PoseLandmarker (XNNPACK CPU) initialized successfully.")

        except Exception as e:
            print(f"[PoseSkeletonDetector] Warning: MediaPipe Task error ({e}). Falling back to OpenCV HOG.")
            self.use_hog_fallback = True
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0
    ) -> Tuple[FrameAnalysisResult, np.ndarray]:
        """
        Process a single BGR image frame, detect human body poses, and render indicators.
        Returns:
            Tuple[FrameAnalysisResult, annotated_frame_bgr]
        """
        start_time = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        height, width, _ = frame_bgr.shape

        humans: List[HumanPose] = []

        if not self.use_hog_fallback and self.landmarker is not None:
            # Convert BGR to RGB format required by MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                for person_idx, person_landmarks in enumerate(detection_result.pose_landmarks):
                    keypoints: List[Keypoint] = []
                    x_coords, y_coords = [], []
                    conf_sum = 0.0
                    valid_count = 0

                    for idx, lm in enumerate(person_landmarks):
                        name = LANDMARK_NAMES[idx] if idx < len(LANDMARK_NAMES) else f"KP_{idx}"
                        px = int(lm.x * width)
                        py = int(lm.y * height)
                        vis = float(getattr(lm, 'visibility', 0.8))
                        keypoints.append(Keypoint(id=idx, name=name, x=px, y=py, z=float(lm.z), visibility=vis))

                        if vis > 0.4:
                            x_coords.append(px)
                            y_coords.append(py)
                            conf_sum += vis
                            valid_count += 1

                    avg_conf = (conf_sum / max(1, valid_count)) if valid_count > 0 else 0.7

                    # Compute Bounding Box with protective padding
                    if x_coords and y_coords:
                        min_x = max(0, min(x_coords) - 25)
                        min_y = max(0, min(y_coords) - 30)
                        max_x = min(width - 1, max(x_coords) + 25)
                        max_y = min(height - 1, max(y_coords) + 25)
                        w = max_x - min_x
                        h = max_y - min_y
                        bbox = (min_x, min_y, w, h)
                    else:
                        bbox = (0, 0, width, height)

                    posture = self._estimate_posture(keypoints)

                    humans.append(HumanPose(
                        track_id=person_idx + 1,
                        bbox=bbox,
                        confidence=float(avg_conf),
                        keypoints=keypoints,
                        posture_label=posture
                    ))
        else:
            # Fallback using OpenCV HOG detector
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for i, (x, y, w, h) in enumerate(boxes):
                conf = float(weights[i]) if i < len(weights) else 0.6
                humans.append(HumanPose(
                    track_id=i + 1,
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=conf,
                    keypoints=[],
                    posture_label="Observed"
                ))

        inference_time_ms = (time.time() - start_time) * 1000.0

        # Update FPS moving average
        curr_time = time.time()
        dt = curr_time - self._last_process_time
        self._last_process_time = curr_time
        if dt > 0:
            inst_fps = 1.0 / dt
            self._fps_history.append(inst_fps)
            if len(self._fps_history) > 15:
                self._fps_history.pop(0)
            self.fps = sum(self._fps_history) / len(self._fps_history)

        analysis = FrameAnalysisResult(
            frame_index=frame_index,
            timestamp=now_str,
            human_count=len(humans),
            humans=humans,
            inference_time_ms=inference_time_ms
        )

        annotated_frame = self.draw_skeleton_and_indicators(frame_bgr.copy(), analysis)

        return analysis, annotated_frame

    def draw_skeleton_and_indicators(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysisResult
    ) -> np.ndarray:
        """
        Renders skeleton bones, keypoint joints, bounding boxes, and SOC HUD overlay.
        """
        h, w, _ = frame.shape

        for human in analysis.humans:
            bx, by, bw, bh = human.bbox

            # 1. Draw Skeleton Bones
            if human.keypoints:
                kp_dict: Dict[int, Keypoint] = {kp.id: kp for kp in human.keypoints}

                for start_idx, end_idx, color in SKELETON_CONNECTIONS:
                    if start_idx in kp_dict and end_idx in kp_dict:
                        kp1 = kp_dict[start_idx]
                        kp2 = kp_dict[end_idx]

                        # Only draw bone if both keypoints have acceptable visibility
                        if kp1.visibility > 0.4 and kp2.visibility > 0.4:
                            cv2.line(frame, (kp1.x, kp1.y), (kp2.x, kp2.y), color, 2, cv2.LINE_AA)

                # 2. Draw Skeleton Joints (Keypoints)
                for kp in human.keypoints:
                    if kp.visibility > 0.4:
                        # Inner solid circle
                        cv2.circle(frame, (kp.x, kp.y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                        # Outer contrasting ring
                        cv2.circle(frame, (kp.x, kp.y), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Draw Bounding Box (Corner accented style for modern SOC look)
            box_color = (0, 255, 127) if human.confidence > 0.6 else (0, 165, 255)
            self._draw_sleek_bounding_box(frame, bx, by, bw, bh, box_color)

            # Label banner
            label = f"HUMAN #{human.track_id} | CONF: {human.confidence * 100:.0f}% | {human.posture_label}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (bx, max(0, by - 22)), (bx + lw + 10, by), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx, max(0, by - 22)), (bx + lw + 10, by), box_color, 1)
            cv2.putText(frame, label, (bx + 5, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Draw HUD Dashboard (Top Banner + Visible Clock + FPS + Alerts)
        self._draw_hud_dashboard(frame, analysis)

        return frame

    def _draw_sleek_bounding_box(self, frame: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """Draws a refined bounding box with highlighted corners."""
        # Main thin rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

        corner_len = min(20, max(8, int(min(w, h) * 0.15)))
        thick = 2
        # Top-Left
        cv2.line(frame, (x, y), (x + corner_len, y), color, thick)
        cv2.line(frame, (x, y), (x, y + corner_len), color, thick)
        # Top-Right
        cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, thick)
        cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, thick)
        # Bottom-Left
        cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, thick)
        cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, thick)
        # Bottom-Right
        cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, thick)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, thick)

    def _draw_hud_dashboard(self, frame: np.ndarray, analysis: FrameAnalysisResult):
        """Draws top status bar with visible clock, FPS, and threat indicators."""
        w = frame.shape[1]

        # Top overlay bar (semi-transparent dark background)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 42), (w, 42), (50, 50, 55), 1)

        # System Title
        cv2.putText(
            frame,
            "URBAN SAFETY SOC | REAL-TIME HUMAN & SKELETON DETECTOR",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Visible Clock Overlay (Crucial for SOC Auditing and Timing Sync)
        clock_text = f"TIME: {analysis.timestamp}"
        cv2.putText(
            frame,
            clock_text,
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Status Badge
        if analysis.human_count > 0:
            status_text = f"[ ALERT: {analysis.human_count} HUMAN(S) DETECTED ]"
            status_color = (0, 69, 255)  # Bright Orange-Red
        else:
            status_text = "[ MONITORING: CLEAR ]"
            status_color = (0, 220, 0)  # Green

        cv2.putText(
            frame,
            status_text,
            (w - 380, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            2,
            cv2.LINE_AA
        )

        # FPS & Inference time metrics
        metrics_text = f"FPS: {self.fps:.1f} | INFER: {analysis.inference_time_ms:.1f}ms"
        cv2.putText(
            frame,
            metrics_text,
            (w - 170, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

        # Evidence Recording Indicator Badge
        if analysis.is_recording_evidence:
            cv2.circle(frame, (w - 20, 15), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, "REC", (w - 48, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

    def _estimate_posture(self, keypoints: List[Keypoint]) -> str:
        """Estimates high-level posture: Standing, Sitting, or Moving."""
        kp_dict = {kp.id: kp for kp in keypoints}
        if 11 in kp_dict and 23 in kp_dict and 25 in kp_dict and 27 in kp_dict:
            hip_y = (kp_dict[23].y + kp_dict.get(24, kp_dict[23]).y) / 2
            knee_y = (kp_dict[25].y + kp_dict.get(26, kp_dict[25]).y) / 2
            ankle_y = (kp_dict[27].y + kp_dict.get(28, kp_dict[27]).y) / 2

            thigh_dist = abs(knee_y - hip_y)
            shin_dist = abs(ankle_y - knee_y)

            if thigh_dist < shin_dist * 0.45:
                return "Sitting"
            return "Standing"
        return "Observed"

    def close(self) -> None:
        """Releases AI model resources safely."""
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
