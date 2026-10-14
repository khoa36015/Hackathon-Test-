"""
Pose Skeleton Detector Module for Urban Safety AI-SOC.
Performs real-time MULTI-PERSON human detection with:
1. Spatial Non-Maximum Suppression (NMS) to eliminate duplicate/ghost detections.
2. Anatomical Bone Length & Proximity Validation to eliminate cross-person limb contamination.
3. Persistent Multi-Person Centroid & IoU Tracking to prevent ID flapping.
4. Individual color palettes and non-obtrusive corner-accented bounding boxes.
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

# Anatomical bone connections: (start_idx, end_idx)
SKELETON_PAIRS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (30, 32),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10)
]

# Distinct, high-contrast color palettes for each tracked person
PERSON_PALETTES = [
    # Person 1: Neon Cyan & Lime
    {"bone": (0, 255, 255), "joint": (0, 255, 0), "box": (0, 255, 127), "name": "CYAN"},
    # Person 2: Warm Amber & Orange
    {"bone": (0, 140, 255), "joint": (0, 215, 255), "box": (0, 165, 255), "name": "ORANGE"},
    # Person 3: Bright Magenta / Pink
    {"bone": (255, 0, 255), "joint": (255, 105, 180), "box": (255, 0, 255), "name": "MAGENTA"},
    # Person 4: Electric Sky Blue & Yellow
    {"bone": (255, 215, 0), "joint": (0, 255, 255), "box": (255, 191, 0), "name": "BLUE"},
    # Person 5: Spring Emerald Green
    {"bone": (50, 205, 50), "joint": (124, 252, 0), "box": (0, 255, 0), "name": "GREEN"},
    # Person 6: Coral Red & Gold
    {"bone": (80, 80, 255), "joint": (0, 215, 255), "box": (0, 69, 255), "name": "RED"}
]

MODEL_URL_FULL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
MODEL_URL_LITE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"


def compute_box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


class MultiPersonTracker:
    """
    Robust multi-person tracker using IoU and centroid distance matching.
    Maintains persistent IDs and prevents identity swapping when people cross paths.
    """

    def __init__(self, max_disappeared: int = 12) -> None:
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.tracks: Dict[int, Dict] = {}  # track_id -> {"bbox", "centroid", "disappeared", "palette_idx"}

    def update(self, detected_humans: List[HumanPose]) -> List[HumanPose]:
        """Matches current detections with existing tracks and assigns persistent track IDs."""
        if not detected_humans:
            # Mark all existing tracks as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        # If no tracks yet, register all detections
        if not self.tracks:
            for h in detected_humans:
                tid = self.next_id
                self.next_id += 1
                bx, by, bw, bh = h.bbox
                centroid = (bx + bw / 2.0, by + bh / 2.0)
                self.tracks[tid] = {
                    "bbox": h.bbox,
                    "centroid": centroid,
                    "disappeared": 0,
                    "palette_idx": (tid - 1) % len(PERSON_PALETTES)
                }
                h.track_id = tid
            return detected_humans

        # Pairwise matching between existing tracks and incoming detections
        track_ids = list(self.tracks.keys())
        cost_matrix = []

        for tid in track_ids:
            t_bbox = self.tracks[tid]["bbox"]
            t_cx, t_cy = self.tracks[tid]["centroid"]
            row = []
            for h in detected_humans:
                h_bx, h_by, h_bw, h_bh = h.bbox
                h_cx, h_cy = h_bx + h_bw / 2.0, h_by + h_bh / 2.0

                iou = compute_box_iou(t_bbox, h.bbox)
                dist = math.hypot(t_cx - h_cx, t_cy - h_cy)
                diag = math.hypot(h_bw, h_bh) + 1e-5
                norm_dist = min(2.0, dist / diag)

                # Cost combines IoU overlap and centroid distance (lower is better)
                cost = (0.65 * (1.0 - iou)) + (0.35 * norm_dist)
                row.append(cost)
            cost_matrix.append(row)

        assigned_tracks = set()
        assigned_detections = set()

        # Greedy matching sorted by lowest cost
        flat_costs = []
        for r_idx, row in enumerate(cost_matrix):
            for c_idx, val in enumerate(row):
                flat_costs.append((val, r_idx, c_idx))
        flat_costs.sort(key=lambda x: x[0])

        for cost, r_idx, c_idx in flat_costs:
            if cost > 0.85:  # Cost threshold for acceptable match
                break
            tid = track_ids[r_idx]
            if tid in assigned_tracks or c_idx in assigned_detections:
                continue

            # Assign track
            assigned_tracks.add(tid)
            assigned_detections.add(c_idx)

            h = detected_humans[c_idx]
            h.track_id = tid
            bx, by, bw, bh = h.bbox
            self.tracks[tid]["bbox"] = h.bbox
            self.tracks[tid]["centroid"] = (bx + bw / 2.0, by + bh / 2.0)
            self.tracks[tid]["disappeared"] = 0

        # Increment disappeared count for unassigned tracks
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Register brand new tracks for unmatched detections
        for c_idx, h in enumerate(detected_humans):
            if c_idx not in assigned_detections:
                tid = self.next_id
                self.next_id += 1
                h.track_id = tid
                bx, by, bw, bh = h.bbox
                self.tracks[tid] = {
                    "bbox": h.bbox,
                    "centroid": (bx + bw / 2.0, by + bh / 2.0),
                    "disappeared": 0,
                    "palette_idx": (tid - 1) % len(PERSON_PALETTES)
                }

        # Sort detections by track_id for consistent ordering
        detected_humans.sort(key=lambda x: x.track_id)
        return detected_humans


class PoseSkeletonDetector:
    """
    High-performance real-time multi-person human detector and pose skeleton estimator.
    Engineered to resolve multi-person occlusion, overlap, and ID confusion.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.25,
        min_tracking_confidence: float = 0.25,
        max_humans: int = 10,
        model_complexity: int = 1
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_humans = max_humans
        self.model_complexity = model_complexity
        self._fps_history = []
        self._last_process_time = time.time()
        self.fps = 0.0

        self.landmarker = None
        self.mp = None
        self.use_hog_fallback = False
        self.tracker = MultiPersonTracker(max_disappeared=10)

        self._init_mediapipe_landmarker()

    def _init_mediapipe_landmarker(self) -> None:
        """Initializes modern MediaPipe PoseLandmarker task configured for multi-person detection."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self.mp = mp

            curr_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(curr_dir, "models")
            os.makedirs(models_dir, exist_ok=True)

            model_path_full = os.path.join(models_dir, "pose_landmarker_full.task")
            model_path_lite = os.path.join(models_dir, "pose_landmarker_lite.task")

            active_model_path = None
            if os.path.exists(model_path_full) and os.path.getsize(model_path_full) > 1000:
                active_model_path = model_path_full
            elif os.path.exists(model_path_lite) and os.path.getsize(model_path_lite) > 1000:
                active_model_path = model_path_lite
            else:
                try:
                    print(f"[PoseSkeletonDetector] Downloading full multi-person pose model...")
                    urllib.request.urlretrieve(MODEL_URL_FULL, model_path_full)
                    active_model_path = model_path_full
                except Exception:
                    print(f"[PoseSkeletonDetector] Downloading lite pose model fallback...")
                    urllib.request.urlretrieve(MODEL_URL_LITE, model_path_lite)
                    active_model_path = model_path_lite

            base_options = python.BaseOptions(model_asset_path=active_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False,
                num_poses=self.max_humans
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            print(f"[PoseSkeletonDetector] Multi-Person PoseLandmarker (Max: {self.max_humans} people) initialized successfully.")

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
        Processes a video frame, detects all human poses, performs NMS overlap filtering,
        tracks persistent IDs, and draws clean, isolated skeleton overlays.
        """
        start_time = time.time()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        height, width, _ = frame_bgr.shape

        raw_humans: List[HumanPose] = []

        if not self.use_hog_fallback and self.landmarker is not None:
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

                        if vis > 0.25:
                            x_coords.append(px)
                            y_coords.append(py)
                            conf_sum += vis
                            valid_count += 1

                    # Need at least 4 valid keypoints to constitute a credible human detection
                    if valid_count < 4:
                        continue

                    avg_conf = conf_sum / max(1, valid_count)

                    # Compute Bounding Box with protective padding
                    if x_coords and y_coords:
                        min_x = max(0, min(x_coords) - 20)
                        min_y = max(0, min(y_coords) - 25)
                        max_x = min(width - 1, max(x_coords) + 20)
                        max_y = min(height - 1, max(y_coords) + 20)
                        w = max(10, max_x - min_x)
                        h = max(10, max_y - min_y)
                        bbox = (min_x, min_y, w, h)
                    else:
                        continue

                    posture = self._estimate_posture(keypoints)

                    raw_humans.append(HumanPose(
                        track_id=person_idx + 1,
                        bbox=bbox,
                        confidence=float(avg_conf),
                        keypoints=keypoints,
                        posture_label=posture
                    ))
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            boxes, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
            for i, (x, y, w, h) in enumerate(boxes):
                conf = float(weights[i]) if i < len(weights) else 0.6
                raw_humans.append(HumanPose(
                    track_id=i + 1,
                    bbox=(int(x), int(y), int(w), int(h)),
                    confidence=conf,
                    keypoints=[],
                    posture_label="Observed"
                ))

        # 1. Non-Maximum Suppression (NMS) to eliminate duplicate/ghost detections of the same person
        filtered_humans = self._filter_duplicate_poses(raw_humans)

        # 2. Multi-Person Tracking across frames to maintain stable IDs and stop frame swapping
        tracked_humans = self.tracker.update(filtered_humans)

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
            human_count=len(tracked_humans),
            humans=tracked_humans,
            inference_time_ms=inference_time_ms
        )

        annotated_frame = self.draw_skeleton_and_indicators(frame_bgr.copy(), analysis)

        return analysis, annotated_frame

    def _filter_duplicate_poses(self, humans: List[HumanPose], iou_thresh: float = 0.50) -> List[HumanPose]:
        """
        Removes overlapping ghost detections of the same human body.
        Retains the higher-confidence, higher-quality pose detection.
        """
        if len(humans) <= 1:
            return humans

        # Sort descending by confidence * valid_keypoints
        sorted_humans = sorted(
            humans,
            key=lambda h: h.confidence * len([kp for kp in h.keypoints if kp.visibility > 0.25]),
            reverse=True
        )

        accepted: List[HumanPose] = []
        for candidate in sorted_humans:
            is_duplicate = False
            for existing in accepted:
                iou = compute_box_iou(candidate.bbox, existing.bbox)
                if iou > iou_thresh:
                    is_duplicate = True
                    break

                # Also check torso center proximity (if both are within 15% distance, they are the same person)
                c_bx, c_by, c_bw, c_bh = candidate.bbox
                e_bx, e_by, e_bw, e_bh = existing.bbox
                c_cx, c_cy = c_bx + c_bw / 2.0, c_by + c_bh / 2.0
                e_cx, e_cy = e_bx + e_bw / 2.0, e_by + e_bh / 2.0
                dist = math.hypot(c_cx - e_cx, c_cy - e_cy)
                min_dim = min(c_bw, c_bh, e_bw, e_bh)

                if dist < 0.25 * min_dim:
                    is_duplicate = True
                    break

            if not is_duplicate:
                accepted.append(candidate)

        return accepted

    def draw_skeleton_and_indicators(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysisResult
    ) -> np.ndarray:
        """
        Renders isolated skeleton bones, joints, bounding boxes, and SOC HUD overlay.
        Includes Anatomical Bone Length Validation to prevent cross-person limb contamination.
        """
        h, w, _ = frame.shape

        for human in analysis.humans:
            bx, by, bw, bh = human.bbox
            palette_idx = (human.track_id - 1) % len(PERSON_PALETTES)
            palette = PERSON_PALETTES[palette_idx]

            bone_color = palette["bone"]
            joint_color = palette["joint"]
            box_color = palette["box"]

            # 1. Draw Skeleton Bones with Anatomical Validation
            if human.keypoints:
                kp_dict: Dict[int, Keypoint] = {kp.id: kp for kp in human.keypoints}

                for start_idx, end_idx in SKELETON_PAIRS:
                    if start_idx in kp_dict and end_idx in kp_dict:
                        kp1 = kp_dict[start_idx]
                        kp2 = kp_dict[end_idx]

                        # Check visibility
                        if kp1.visibility <= 0.25 or kp2.visibility <= 0.25:
                            continue

                        # ANATOMICAL VALIDATION: Prevent bone from jumping to another person!
                        if not self._is_valid_bone_segment(kp1, kp2, human.bbox):
                            continue

                        cv2.line(frame, (kp1.x, kp1.y), (kp2.x, kp2.y), bone_color, 2, cv2.LINE_AA)

                # 2. Draw Skeleton Joints (Keypoints)
                for kp in human.keypoints:
                    if kp.visibility > 0.25:
                        # Ensure joint is within reasonable proximity of this person's bounding box
                        if self._is_point_near_box(kp.x, kp.y, human.bbox):
                            cv2.circle(frame, (kp.x, kp.y), 4, joint_color, -1, cv2.LINE_AA)
                            cv2.circle(frame, (kp.x, kp.y), 6, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Draw Bounding Box with sleek corner accents
            self._draw_sleek_bounding_box(frame, bx, by, bw, bh, box_color)

            # 4. Label banner with ID, Confidence, and Posture
            label = f"ID #{human.track_id} | {human.confidence * 100:.0f}% | {human.posture_label}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (bx, max(0, by - 20)), (bx + lw + 8, by), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx, max(0, by - 20)), (bx + lw + 8, by), box_color, 1)
            cv2.putText(frame, label, (bx + 4, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Draw Top HUD Dashboard
        self._draw_hud_dashboard(frame, analysis)

        return frame

    def _is_point_near_box(self, x: int, y: int, bbox: Tuple[int, int, int, int]) -> bool:
        """Verifies a keypoint is not an extreme outlier far away from this person's body."""
        bx, by, bw, bh = bbox
        margin_x = max(25, int(bw * 0.30))
        margin_y = max(30, int(bh * 0.30))
        return (bx - margin_x <= x <= bx + bw + margin_x) and (by - margin_y <= y <= by + bh + margin_y)

    def _is_valid_bone_segment(self, kp1: Keypoint, kp2: Keypoint, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Validates that a bone segment is anatomically plausible for this person.
        Rejects erroneous connections that jump across to another person's body!
        """
        bx, by, bw, bh = bbox

        # 1. Both points must reside in reasonable proximity to this person's bounding box
        if not self._is_point_near_box(kp1.x, kp1.y, bbox) or not self._is_point_near_box(kp2.x, kp2.y, bbox):
            return False

        # 2. Bone length cannot exceed realistic physiological limit
        bone_length = math.hypot(kp1.x - kp2.x, kp1.y - kp2.y)
        max_allowed_bone = max(40.0, 0.70 * math.hypot(bw, bh))

        if bone_length > max_allowed_bone:
            # Outlier bone crossing over to another person
            return False

        return True

    def _draw_sleek_bounding_box(self, frame: np.ndarray, x: int, y: int, w: int, h: int, color: Tuple[int, int, int]):
        """Draws an isolated bounding box with highlighted corner brackets."""
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

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 18), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (0, 42), (w, 42), (50, 50, 55), 1)

        # System Title
        cv2.putText(
            frame,
            "URBAN SAFETY SOC | MULTI-PERSON SKELETON DETECTOR",
            (12, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        # Visible Clock Overlay
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

        # Multi-Person Status Alert Badge
        if analysis.human_count > 0:
            count_label = f"[ ALERT: {analysis.human_count} HUMAN(S) DETECTED ]"
            status_color = (0, 69, 255) if analysis.human_count > 1 else (0, 165, 255)
        else:
            count_label = "[ MONITORING: CLEAR ]"
            status_color = (0, 220, 0)

        cv2.putText(
            frame,
            count_label,
            (w - 410, 25),
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
        """Estimates high-level posture: Standing, Sitting, or Observed."""
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
