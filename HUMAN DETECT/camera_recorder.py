"""
Live Camera Recorder Application with Real-Time Human Detection & Pose Skeleton Indicator.
Enhanced version with YOLO11-Pose, ByteTrack, and Behavior Recognition Engine.
Urban Safety AI-SOC Edition.
"""

import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from typing import Tuple, Optional

# Prefer YOLO11-Pose + ByteTrack + BehaviorEngine
try:
    from yolo_pose_detector import YOLOPoseDetector
    DEFAULT_DETECTOR_CLS = YOLOPoseDetector
except Exception:
    from pose_detector import PoseSkeletonDetector
    DEFAULT_DETECTOR_CLS = PoseSkeletonDetector

from clip_recorder import EvidenceClipRecorder


class CameraRecorder:
    """
    Real-time Camera Streamer with YOLO11-Pose, ByteTrack, Behavior Recognition,
    and Automatic Evidence Rolling Buffer Extraction.
    """

    def __init__(
        self,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        record_full_session: bool = True,
        auto_extract_evidence: bool = True,
        model_weights: str = "yolo11n-pose.pt",
        device: Optional[str] = None,
        min_confidence: float = 0.25,
        output_dir: str = "output",
        records_dir: str = "records"
    ) -> None:
        self.camera_index = camera_index
        self.width, self.height = resolution
        self.record_full_session = record_full_session
        self.auto_extract_evidence = auto_extract_evidence
        self.model_weights = model_weights
        self.device = device
        self.min_confidence = min_confidence
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(base_dir, output_dir)
        self.records_dir = records_dir if os.path.isabs(records_dir) else os.path.join(base_dir, records_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)

        print("[CameraRecorder] Initializing camera stream...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w <= 0 or actual_h <= 0:
                actual_w, actual_h = resolution
            self.frame_size = (actual_w, actual_h)
            print(f"[CameraRecorder] Camera #{self.camera_index} connected: {actual_w}x{actual_h}")
        else:
            self.frame_size = resolution
            print(f"[CameraRecorder] Warning: Could not open camera #{self.camera_index}. Fallback to simulated stream.")

        # Initialize AI Detector
        try:
            self.detector = YOLOPoseDetector(
                model_path=self.model_weights,
                device=self.device,
                conf_threshold=self.min_confidence,
                fps=20.0
            )
            print("[CameraRecorder] Initialized YOLOPoseDetector with ByteTrack & BehaviorEngine.")
        except Exception as e:
            print(f"[CameraRecorder] Warning: YOLOPoseDetector init failed ({e}). Falling back to MediaPipe PoseSkeletonDetector.")
            from pose_detector import PoseSkeletonDetector
            self.detector = PoseSkeletonDetector(
                min_detection_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
                max_humans=10,
                model_complexity=1
            )

        # Initialize Evidence Clip Recorder
        self.evidence_recorder = EvidenceClipRecorder(
            output_dir=self.output_dir,
            fps=20.0,
            frame_size=self.frame_size,
            pre_roll_seconds=1.5,
            post_roll_seconds=2.0
        )

        # Full session recorder
        self.session_writer: Optional[cv2.VideoWriter] = None
        if self.record_full_session:
            self._start_session_recorder()

        self.show_skeleton = True

    def _start_session_recorder(self) -> None:
        """Starts recording the entire continuous camera session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.records_dir, f"SESSION_{timestamp}_FULL.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.session_writer = cv2.VideoWriter(output_file, fourcc, 20.0, self.frame_size)
        print(f"[CameraRecorder] Full continuous session recording started: {output_file}")

    def record_video(self) -> None:
        """
        Main application loop: reads frames, runs real-time behavior detection,
        records evidence clips, and displays live GUI.
        """
        frame_idx = 0
        window_title = "Urban Safety AI-SOC: Human Behavior & Pose Monitor"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        print("\n=======================================================")
        print(" URBAN SAFETY SOC: YOLO11-POSE & BEHAVIOR RECORDER")
        print(" Hotkeys:")
        print("   [q] - Quit and finalize all recordings")
        print("   [s] - Save high-res snapshot with skeleton overlay")
        print("   [h] - Toggle skeleton and HUD overlay ON/OFF")
        print("   [d] - Toggle Diagnostic Alert Box Debug HUD ON/OFF")
        print("   [r] - Toggle continuous session recording ON/OFF")
        print("=======================================================\n")

        try:
            while True:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("[CameraRecorder] End of stream or frame read failed.")
                        break
                else:
                    # Simulated test frame if camera is unavailable
                    frame = self._generate_simulated_frame(frame_idx)

                frame_idx += 1
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # 1. AI Pose & Behavior Inference
                if self.show_skeleton:
                    analysis, display_frame = self.detector.process_frame(frame, frame_idx)
                else:
                    display_frame = frame.copy()
                    analysis = None

                is_human = analysis.has_human if analysis is not None else False

                # 2. Automated Evidence Clip Recording
                if self.auto_extract_evidence:
                    if analysis is not None:
                        analysis.is_recording_evidence = self.evidence_recorder.is_recording

                    saved_clip = self.evidence_recorder.update(display_frame, is_human, now_str)
                    if saved_clip:
                        print(f"[CameraRecorder] Evidence clip finalized: {saved_clip}")

                # 3. Full Session Continuous Recording
                if self.session_writer is not None:
                    self.session_writer.write(display_frame)

                # 4. Display Frame
                cv2.imshow(window_title, display_frame)

                # 5. Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[CameraRecorder] Stopping camera recorder...")
                    break
                elif key == ord('s'):
                    snap_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snap_path = os.path.join(self.output_dir, f"snapshot_{snap_time}.jpg")
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[CameraRecorder] Snapshot saved: {snap_path}")
                elif key == ord('h'):
                    self.show_skeleton = not self.show_skeleton
                    status = "ON" if self.show_skeleton else "OFF"
                    print(f"[CameraRecorder] HUD overlay: {status}")
                elif key == ord('d'):
                    if hasattr(self.detector, 'toggle_debug_box'):
                        dbg_status = self.detector.toggle_debug_box()
                        print(f"[CameraRecorder] Diagnostic Alert Box Debug: {'ON' if dbg_status else 'OFF'}")
                elif key == ord('r'):
                    if self.session_writer is not None:
                        self.session_writer.release()
                        self.session_writer = None
                        print("[CameraRecorder] Continuous session recording stopped.")
                    else:
                        self._start_session_recorder()

        finally:
            self.cleanup()

    def _generate_simulated_frame(self, frame_idx: int) -> np.ndarray:
        """Generates synthetic surveillance video frame for testing without physical camera."""
        frame = np.full((self.height, self.width, 3), 35, dtype=np.uint8)
        # Background grid
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (45, 45, 50), 1)
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (45, 45, 50), 1)

        cv2.putText(
            frame,
            "SIMULATED CAMERA STREAM - NO PHYSICAL WEBCAM ATTACHED",
            (20, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA
        )
        return frame

    def cleanup(self) -> None:
        """Safely cleans up video writers, camera hardware, and OpenCV windows."""
        print("[CameraRecorder] Cleaning up resources...")
        if self.cap.isOpened():
            self.cap.release()

        if self.session_writer is not None:
            self.session_writer.release()
            self.session_writer = None

        if self.evidence_recorder is not None:
            self.evidence_recorder.release()

        if hasattr(self.detector, 'close'):
            self.detector.close()

        cv2.destroyAllWindows()
        print("[CameraRecorder] All resources cleanly released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera Human & Behavior Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11-Pose weights file")
    args = parser.parse_args()

    app = CameraRecorder(camera_index=args.camera, model_weights=args.weights)
    app.record_video()
