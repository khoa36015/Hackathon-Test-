"""
Temporal Behavior & Biomechanical Action Recognition Engine for Urban Safety AI-SOC.
Enhanced with:
1. Spatial Bounding Box & Kinematic EMA Smoothing (Anti-Jitter)
2. Hysteresis & Debounce State Machine (Anti-Flapping)
3. Formal SOC Action Audit Logging (JSONL & text log with anti-spam rate limiting)
4. Diagnostic Telemetry Feed for Alert Box Debug HUD
"""

import os
import time
import math
import json
from dataclasses import asdict
from collections import deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Deque

from models import (
    KeypointCOCO,
    BehaviorType,
    ThreatLevel,
    BehaviorPrediction,
    TrackHistoryState,
    HumanPoseTrack,
    BehaviorAuditLogEntry,
    TrackKinematicsTelemetry
)


# COCO-17 Keypoint Indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


class BehaviorAuditLogger:
    """
    SOC-Grade Action Audit Logger.
    Persists behavior detections and state transitions to JSONL and audit log file.
    Applies rate-limiting to prevent log flooding while guaranteeing non-repudiation.
    """

    def __init__(self, output_dir: str = "output", heartbeat_interval: float = 1.0) -> None:
        self.output_dir = output_dir
        self.heartbeat_interval = heartbeat_interval
        os.makedirs(self.output_dir, exist_ok=True)

        self.jsonl_path = os.path.join(self.output_dir, "behavior_events.jsonl")
        self.text_log_path = os.path.join(self.output_dir, "behavior_audit.log")

        # Open file handles in append mode with line buffering
        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8", buffering=1)
        self._text_file = open(self.text_log_path, "a", encoding="utf-8", buffering=1)

    def log_event(self, entry: BehaviorAuditLogEntry) -> None:
        """Appends audit event to JSONL and text log with instant flush."""
        try:
            # 1. Structured JSON Lines log
            json_line = json.dumps(asdict(entry), ensure_ascii=False)
            self._jsonl_file.write(json_line + "\n")

            # 2. Human-readable text audit log
            prev_str = entry.previous_behavior if entry.previous_behavior else "INIT"
            text_line = (
                f"[{entry.timestamp}] [{entry.event_type}] ID #{entry.track_id}: "
                f"{prev_str} -> {entry.current_behavior} "
                f"({entry.threat_level} {entry.threat_score * 100:.0f}%) | "
                f"Spine:{entry.telemetry.get('spine_angle', 0):.0f}° "
                f"Spd:{entry.telemetry.get('speed', 0):.2f} | "
                f"Reason: {entry.reasoning}\n"
            )
            self._text_file.write(text_line)

            self._jsonl_file.flush()
            self._text_file.flush()
        except Exception as e:
            print(f"[BehaviorAuditLogger] Warning: Could not write audit log ({e})")

    def close(self) -> None:
        """Safely closes file descriptors."""
        try:
            if not self._jsonl_file.closed:
                self._jsonl_file.close()
            if not self._text_file.closed:
                self._text_file.close()
        except Exception:
            pass


class BehaviorEngine:
    """
    Temporal Behavior Analysis Engine with Anti-Flapping & Bounding Box Stabilization.
    Employs an Exponential Moving Average (EMA) and a Hysteresis Debounce State Machine.
    """

    def __init__(
        self,
        fps: float = 20.0,
        history_window_size: int = 30,         # ~1.5 seconds at 20 FPS
        fall_angle_threshold: float = 55.0,     # Degrees from vertical
        fall_aspect_ratio_thresh: float = 0.90, # H/W < 0.90
        run_speed_multiplier: float = 1.75,     # Speed normalized by body height / sec
        loiter_time_threshold: float = 10.0,    # Seconds before loitering alert
        fight_wrist_energy_thresh: float = 1200.0,
        fight_distance_norm: float = 1.35,
        enter_debounce_frames: int = 8,        # Frames to confirm non-normal state (Anti-flapping)
        exit_debounce_frames: int = 12,        # Frames to confirm return to NORMAL
        bbox_ema_alpha: float = 0.65,          # Bbox smoothing weight (0.65 curr, 0.35 prev)
        output_dir: str = "output"
    ) -> None:
        self.fps = fps if fps > 0 else 20.0
        self.history_window_size = history_window_size
        self.fall_angle_threshold = fall_angle_threshold
        self.fall_aspect_ratio_thresh = fall_aspect_ratio_thresh
        self.run_speed_multiplier = run_speed_multiplier
        self.loiter_time_threshold = loiter_time_threshold
        self.fight_wrist_energy_thresh = fight_wrist_energy_thresh
        self.fight_distance_norm = fight_distance_norm
        self.enter_debounce_frames = enter_debounce_frames
        self.exit_debounce_frames = exit_debounce_frames
        self.bbox_ema_alpha = bbox_ema_alpha

        # Track State Store: track_id -> TrackMeta
        self.tracks_meta: Dict[int, Dict] = {}

        # Recent alerts buffer (up to 5 most recent events for live HUD ticker)
        self.recent_alerts: Deque[str] = deque(maxlen=5)

        # Audit Logger
        self.audit_logger = BehaviorAuditLogger(output_dir=output_dir)

    def update(
        self,
        tracked_poses: List[HumanPoseTrack],
        timestamp: Optional[float] = None,
        frame_index: int = 0
    ) -> Tuple[List[HumanPoseTrack], List[TrackKinematicsTelemetry], List[str]]:
        """
        1. Applies Spatial Bounding Box Smoothing (EMA).
        2. Computes Biomechanical Kinematics.
        3. Applies Hysteresis Debounce State Machine (Anti-Flapping).
        4. Logs State Transitions to Audit Log.
        5. Returns stabilized poses, live telemetry, and recent alert ticker.
        """
        curr_time = time.time() if timestamp is None else timestamp
        telemetry_list: List[TrackKinematicsTelemetry] = []

        for pose in tracked_poses:
            tid = pose.track_id

            # Initialize track metadata if new
            if tid not in self.tracks_meta:
                bx1, by1, bx2, by2 = pose.bbox
                c_init = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                self.tracks_meta[tid] = {
                    "first_seen": curr_time,
                    "last_seen": curr_time,
                    "initial_centroid": c_init,
                    "history": deque(maxlen=self.history_window_size),
                    "smooth_bbox": pose.bbox,
                    # Anti-Flapping Hysteresis State Machine
                    "confirmed_behavior": BehaviorType.NORMAL,
                    "confirmed_threat_level": ThreatLevel.LOW,
                    "confirmed_threat_score": 0.10,
                    "confirmed_since": curr_time,
                    "candidate_behavior": None,
                    "candidate_streak": 0,
                    "last_heartbeat_time": curr_time
                }

                # Log initial registration
                self.audit_logger.log_event(BehaviorAuditLogEntry(
                    event_id=f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tid}",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    frame_index=frame_index,
                    track_id=tid,
                    event_type="TRACK_INITIALIZED",
                    previous_behavior=None,
                    current_behavior=BehaviorType.NORMAL.value,
                    threat_level=ThreatLevel.LOW.value,
                    threat_score=0.10,
                    telemetry={"spine_angle": 0.0, "speed": 0.0},
                    reasoning="Initial track identity assigned by ByteTrack",
                    dwell_time_seconds=0.0
                ))

            meta = self.tracks_meta[tid]
            meta["last_seen"] = curr_time
            pose.dwell_time_seconds = curr_time - meta["first_seen"]

            # ---------------------------------------------------------
            # 1. SPATIAL BOUNDING BOX STABILIZATION (EMA Filter)
            # Eliminates high-frequency box jittering
            # ---------------------------------------------------------
            prev_sb = meta["smooth_bbox"]
            raw_b = pose.bbox
            a = self.bbox_ema_alpha
            smooth_b = (
                int(a * raw_b[0] + (1.0 - a) * prev_sb[0]),
                int(a * raw_b[1] + (1.0 - a) * prev_sb[1]),
                int(a * raw_b[2] + (1.0 - a) * prev_sb[2]),
                int(a * raw_b[3] + (1.0 - a) * prev_sb[3])
            )
            meta["smooth_bbox"] = smooth_b
            pose.bbox = smooth_b

            # Extract instantaneous kinematic state
            state = self._extract_state(pose, curr_time, meta["history"])
            meta["history"].append(state)

            # Analyze instantaneous biomechanics
            raw_pred, norm_speed, aspect_ratio = self._analyze_single_track(tid, pose, meta)

            # ---------------------------------------------------------
            # 2. HYSTERESIS DEBOUNCE STATE MACHINE (Anti-Flapping)
            # Rejects 1-2 frame transient classification noise
            # ---------------------------------------------------------
            new_candidate = raw_pred.primary_behavior

            if new_candidate == meta["confirmed_behavior"]:
                # Consistent with confirmed state: clear any transient candidate
                meta["candidate_behavior"] = None
                meta["candidate_streak"] = 0
            else:
                # Potential transition detected
                if new_candidate == meta["candidate_behavior"]:
                    meta["candidate_streak"] += 1
                else:
                    meta["candidate_behavior"] = new_candidate
                    meta["candidate_streak"] = 1

                # Determine threshold
                if new_candidate in (BehaviorType.FALLEN, BehaviorType.FIGHTING):
                    # Fast-track emergency events (6 frames ~0.3s)
                    req_streak = 6
                elif new_candidate == BehaviorType.NORMAL:
                    # Require solid proof to return to normal
                    req_streak = self.exit_debounce_frames
                else:
                    req_streak = self.enter_debounce_frames

                # If debounced streak threshold reached: commit transition!
                if meta["candidate_streak"] >= req_streak:
                    prev_b = meta["confirmed_behavior"]
                    meta["confirmed_behavior"] = new_candidate
                    meta["confirmed_threat_level"] = raw_pred.threat_level
                    meta["confirmed_threat_score"] = raw_pred.threat_score
                    meta["confirmed_since"] = curr_time
                    meta["candidate_behavior"] = None
                    meta["candidate_streak"] = 0

                    # 3. GHI LOG KIỂM TOÁN TỨC THỜI (Audit Action Logging)
                    event_type = "CRITICAL_ALERT" if raw_pred.threat_level == ThreatLevel.CRITICAL else (
                        "THREAT_ESCALATION" if raw_pred.threat_level == ThreatLevel.HIGH else "STATE_TRANSITION"
                    )
                    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    audit_entry = BehaviorAuditLogEntry(
                        event_id=f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tid}",
                        timestamp=now_iso,
                        frame_index=frame_index,
                        track_id=tid,
                        event_type=event_type,
                        previous_behavior=prev_b.value,
                        current_behavior=new_candidate.value,
                        threat_level=raw_pred.threat_level.value,
                        threat_score=raw_pred.threat_score,
                        telemetry={
                            "spine_angle": round(state.spine_angle, 1),
                            "aspect_ratio": round(aspect_ratio, 2),
                            "speed": round(norm_speed, 2),
                            "wrist_velocity": round(state.wrist_velocity, 1)
                        },
                        reasoning=raw_pred.reasoning,
                        dwell_time_seconds=round(curr_time - meta["first_seen"], 2)
                    )
                    self.audit_logger.log_event(audit_entry)

                    # Update live alert feed for Debug Alert Box
                    t_short = datetime.now().strftime("%H:%M:%S")
                    alert_badge = f"[{t_short}] ID #{tid} {prev_b.value} -> {new_candidate.value}"
                    self.recent_alerts.appendleft(alert_badge)

            # Periodic heartbeat logging if elevated threat continues
            if meta["confirmed_threat_level"] in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
                if curr_time - meta["last_heartbeat_time"] >= self.audit_logger.heartbeat_interval:
                    meta["last_heartbeat_time"] = curr_time
                    hb_entry = BehaviorAuditLogEntry(
                        event_id=f"HB_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tid}",
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        frame_index=frame_index,
                        track_id=tid,
                        event_type="HEARTBEAT_ALERT",
                        previous_behavior=meta["confirmed_behavior"].value,
                        current_behavior=meta["confirmed_behavior"].value,
                        threat_level=meta["confirmed_threat_level"].value,
                        threat_score=meta["confirmed_threat_score"],
                        telemetry={"spine_angle": round(state.spine_angle, 1), "speed": round(norm_speed, 2)},
                        reasoning=f"Ongoing {meta['confirmed_behavior'].value} event",
                        dwell_time_seconds=round(curr_time - meta["confirmed_since"], 2)
                    )
                    self.audit_logger.log_event(hb_entry)

            # Assign stabilized, debounced behavior to pose
            pose.behavior = BehaviorPrediction(
                primary_behavior=meta["confirmed_behavior"],
                confidence=raw_pred.confidence,
                threat_level=meta["confirmed_threat_level"],
                threat_score=meta["confirmed_threat_score"],
                reasoning=raw_pred.reasoning,
                interacting_with_track_id=raw_pred.interacting_with_track_id
            )

            # Compile telemetry entry for Alert Box Debug HUD
            telemetry_list.append(TrackKinematicsTelemetry(
                track_id=tid,
                spine_angle=round(state.spine_angle, 1),
                aspect_ratio=round(aspect_ratio, 2),
                speed=round(norm_speed, 2),
                wrist_energy=round(state.wrist_velocity, 1),
                dwell_time=round(curr_time - meta["confirmed_since"], 1),
                confirmed_behavior=meta["confirmed_behavior"].value,
                threat_level=meta["confirmed_threat_level"],
                threat_score=meta["confirmed_threat_score"]
            ))

        # Check multi-person interaction (fighting)
        self._analyze_multi_person_interactions(tracked_poses, curr_time, frame_index)

        # Cleanup tracks inactive for > 5.0 seconds
        stale_ids = [tid for tid, m in self.tracks_meta.items() if curr_time - m["last_seen"] > 5.0]
        for tid in stale_ids:
            del self.tracks_meta[tid]

        return tracked_poses, telemetry_list, list(self.recent_alerts)

    def _extract_state(
        self,
        pose: HumanPoseTrack,
        timestamp: float,
        history: Deque[TrackHistoryState]
    ) -> TrackHistoryState:
        """Extracts biomechanical kinematics from keypoints and bounding box with EMA smoothing."""
        bx1, by1, bx2, by2 = pose.bbox
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0

        kp_dict = {kp.id: kp for kp in pose.keypoints if kp.confidence > 0.25}

        # 1. Spine Angle calculation
        spine_angle = 0.0
        if (LEFT_SHOULDER in kp_dict or RIGHT_SHOULDER in kp_dict) and (LEFT_HIP in kp_dict or RIGHT_HIP in kp_dict):
            s_pts = [kp_dict[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER) if i in kp_dict]
            h_pts = [kp_dict[i] for i in (LEFT_HIP, RIGHT_HIP) if i in kp_dict]
            s_x = sum(p.x for p in s_pts) / len(s_pts)
            s_y = sum(p.y for p in s_pts) / len(s_pts)
            h_x = sum(p.x for p in h_pts) / len(h_pts)
            h_y = sum(p.y for p in h_pts) / len(h_pts)

            dx = abs(s_x - h_x)
            dy = abs(s_y - h_y)
            spine_angle = math.degrees(math.atan2(dx, max(1e-4, dy)))

        # EMA smoothing on spine angle to prevent angle jitter
        if len(history) > 0:
            spine_angle = 0.70 * spine_angle + 0.30 * history[-1].spine_angle

        # 2. Body Velocity & Wrist Velocity
        body_vel = 0.0
        wrist_vel = 0.0
        if len(history) > 0:
            prev = history[-1]
            dt = max(1e-4, timestamp - prev.timestamp)
            dist = math.hypot(cx - prev.centroid[0], cy - prev.centroid[1])
            body_vel = 0.70 * (dist / dt) + 0.30 * prev.body_velocity

            prev_kp_dict = {kp.id: kp for kp in prev.keypoints if kp.confidence > 0.25}
            w_disp_sum = 0.0
            w_count = 0
            for wid in (LEFT_WRIST, RIGHT_WRIST):
                if wid in kp_dict and wid in prev_kp_dict:
                    d_w = math.hypot(kp_dict[wid].x - prev_kp_dict[wid].x, kp_dict[wid].y - prev_kp_dict[wid].y)
                    w_disp_sum += (d_w / dt) ** 2
                    w_count += 1
            if w_count > 0:
                raw_wv = math.sqrt(w_disp_sum / w_count)
                wrist_vel = 0.70 * raw_wv + 0.30 * prev.wrist_velocity

        return TrackHistoryState(
            timestamp=timestamp,
            bbox=pose.bbox,
            centroid=(cx, cy),
            keypoints=pose.keypoints,
            spine_angle=spine_angle,
            wrist_velocity=wrist_vel,
            body_velocity=body_vel
        )

    def _analyze_single_track(
        self,
        track_id: int,
        pose: HumanPoseTrack,
        meta: Dict
    ) -> Tuple[BehaviorPrediction, float, float]:
        """Classifies instantaneous kinematics into candidate behavior."""
        history = meta["history"]
        bx1, by1, bx2, by2 = pose.bbox
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        aspect_ratio = bh / float(bw)

        kp_dict = {kp.id: kp for kp in pose.keypoints if kp.confidence > 0.25}

        curr_state = history[-1] if len(history) > 0 else None
        spine_angle = curr_state.spine_angle if curr_state else 0.0

        # Normalized Speed
        norm_speed = 0.0
        if len(history) >= 5:
            dt = max(1e-4, history[-1].timestamp - history[-5].timestamp)
            disp = math.hypot(
                history[-1].centroid[0] - history[-5].centroid[0],
                history[-1].centroid[1] - history[-5].centroid[1]
            )
            norm_speed = (disp / dt) / max(10.0, float(bh))

        # -------------------------------------------------------------
        # Rule 1: FALLEN / LYING DOWN (Critical Priority)
        # -------------------------------------------------------------
        is_horizontal = (spine_angle >= self.fall_angle_threshold) or (aspect_ratio < self.fall_aspect_ratio_thresh)

        if is_horizontal:
            reason = f"Torso angle {spine_angle:.0f}° (>= {self.fall_angle_threshold:.0f}°) or H/W {aspect_ratio:.2f} horizontal"
            return BehaviorPrediction(
                primary_behavior=BehaviorType.FALLEN,
                confidence=0.95,
                threat_level=ThreatLevel.CRITICAL,
                threat_score=0.98,
                reasoning=reason
            ), norm_speed, aspect_ratio

        # -------------------------------------------------------------
        # Rule 2: HANDS UP / DISTRESS (High Priority)
        # -------------------------------------------------------------
        head_y = None
        for hid in (NOSE, LEFT_EYE, RIGHT_EYE):
            if hid in kp_dict:
                head_y = kp_dict[hid].y if head_y is None else min(head_y, kp_dict[hid].y)

        if head_y is not None:
            w_above = [wid for wid in (LEFT_WRIST, RIGHT_WRIST) if wid in kp_dict and kp_dict[wid].y < head_y]
            if len(w_above) == 2:
                return BehaviorPrediction(
                    primary_behavior=BehaviorType.HANDS_UP,
                    confidence=0.90,
                    threat_level=ThreatLevel.HIGH,
                    threat_score=0.78,
                    reasoning="Both hands raised above head in distress / surrender gesture"
                ), norm_speed, aspect_ratio

        # -------------------------------------------------------------
        # Rule 3: RUNNING / FLEEING (High Priority)
        # -------------------------------------------------------------
        if norm_speed >= self.run_speed_multiplier:
            return BehaviorPrediction(
                primary_behavior=BehaviorType.RUNNING,
                confidence=0.88,
                threat_level=ThreatLevel.HIGH,
                threat_score=0.82,
                reasoning=f"High movement speed: {norm_speed:.1f} body-heights/sec (> {self.run_speed_multiplier})"
            ), norm_speed, aspect_ratio

        # -------------------------------------------------------------
        # Rule 4: CROUCHING / PROWLING (Medium Priority)
        # -------------------------------------------------------------
        if aspect_ratio < 1.30 and spine_angle < 45.0:
            if (LEFT_KNEE in kp_dict or RIGHT_KNEE in kp_dict) and (LEFT_HIP in kp_dict or RIGHT_HIP in kp_dict):
                return BehaviorPrediction(
                    primary_behavior=BehaviorType.CROUCHING,
                    confidence=0.75,
                    threat_level=ThreatLevel.MEDIUM,
                    threat_score=0.55,
                    reasoning=f"Compressed posture H/W {aspect_ratio:.2f} with bent knees"
                ), norm_speed, aspect_ratio

        # -------------------------------------------------------------
        # Rule 5: LOITERING (Medium to High Priority)
        # -------------------------------------------------------------
        dwell_time = pose.dwell_time_seconds
        if dwell_time >= self.loiter_time_threshold and len(history) >= 20:
            c_curr = curr_state.centroid if curr_state else (0, 0)
            c_init = meta["initial_centroid"]
            net_disp = math.hypot(c_curr[0] - c_init[0], c_curr[1] - c_init[1])

            if net_disp < max(120.0, bw * 2.5):
                threat_lvl = ThreatLevel.HIGH if dwell_time > 20.0 else ThreatLevel.MEDIUM
                threat_sc = 0.72 if dwell_time > 20.0 else 0.52
                return BehaviorPrediction(
                    primary_behavior=BehaviorType.LOITERING,
                    confidence=0.82,
                    threat_level=threat_lvl,
                    threat_score=threat_sc,
                    reasoning=f"Lingering in local area for {dwell_time:.1f}s (disp: {net_disp:.0f}px)"
                ), norm_speed, aspect_ratio

        # -------------------------------------------------------------
        # Rule 6: NORMAL WALKING / STANDING (Low Priority)
        # -------------------------------------------------------------
        return BehaviorPrediction(
            primary_behavior=BehaviorType.NORMAL,
            confidence=0.95,
            threat_level=ThreatLevel.LOW,
            threat_score=0.12,
            reasoning="Normal upright posture and standard gait speed"
        ), norm_speed, aspect_ratio

    def _analyze_multi_person_interactions(
        self,
        poses: List[HumanPoseTrack],
        curr_time: float,
        frame_index: int
    ) -> None:
        """Cross-analyzes pairs of humans for fighting / violent physical struggle."""
        if len(poses) < 2:
            return

        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                p1, p2 = poses[i], poses[j]
                if p1.behavior.primary_behavior == BehaviorType.FALLEN or p2.behavior.primary_behavior == BehaviorType.FALLEN:
                    continue

                b1 = p1.bbox
                b2 = p2.bbox
                c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                avg_w = ((b1[2] - b1[0]) + (b2[2] - b2[0])) / 2.0

                if dist < avg_w * self.fight_distance_norm:
                    hist1 = self.tracks_meta.get(p1.track_id, {}).get("history", [])
                    hist2 = self.tracks_meta.get(p2.track_id, {}).get("history", [])

                    w_vel1 = hist1[-1].wrist_velocity if len(hist1) > 0 else 0.0
                    w_vel2 = hist2[-1].wrist_velocity if len(hist2) > 0 else 0.0

                    if (w_vel1 + w_vel2) > self.fight_wrist_energy_thresh or (w_vel1 > 800.0 and w_vel2 > 800.0):
                        for target_p, opponent_id in ((p1, p2.track_id), (p2, p1.track_id)):
                            target_p.behavior = BehaviorPrediction(
                                primary_behavior=BehaviorType.FIGHTING,
                                confidence=0.92,
                                threat_level=ThreatLevel.CRITICAL,
                                threat_score=0.95,
                                reasoning=f"Violent physical confrontation with ID #{opponent_id}",
                                interacting_with_track_id=opponent_id
                            )

    def close(self) -> None:
        """Closes audit logger safely."""
        self.audit_logger.close()
