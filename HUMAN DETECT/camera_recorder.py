"""
Live Camera Recorder Application with Real-Time Human Detection & Pose Skeleton Indicator.
Enhanced version based on https://github.com/g4lb/camera-recorder-with-human-detection
Urban Safety AI-SOC Edition.
"""

import os
import cv2
import argparse
from datetime import datetime
from typing import Tuple, Optional

from pose_detector import PoseSkeletonDetector
from clip_recorder import EvidenceClipRecorder


class CameraRecorder:
    """
    Real-time Camera Streamer with Human Detection, Skeleton Indicator,
    and Automatic Evidence Extraction.
    """

    def __init__(
        self,
        camera_index: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        record_full_session: bool = False,
        auto_extract_evidence: bool = True,
        output_dir: str = "output",
        records_dir: str = "records"
    ) -> None:
        self.camera_index = camera_index
        self.width, self.height = resolution
        self.record_full_session = record_full_session
        self.auto_extract_evidence = auto_extract_evidence
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(base_dir, output_dir)
        self.records_dir = records_dir if os.path.isabs(records_dir) else os.path.join(base_dir, records_dir)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.records_dir, exist_ok=True)

        print("[CameraRecorder] Initializing camera stream...")
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            # Try default backend if DSHOW fails
            self.cap = cv2.VideoCapture(self.camera_index)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (actual_w, actual_h)
            print(f"[CameraRecorder] Camera #{self.camera_index} connected: {actual_w}x{actual_h}")
        else:
            self.frame_size = resolution
            print(f"[CameraRecorder] Warning: Could not open camera #{self.camera_index}. Fallback to simulated stream.")

        # Initialize AI Pose & Skeleton Detector
        self.detector = PoseSkeletonDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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

        # Full session recorder (compatible with reference repo)
        self.session_writer: Optional[cv2.VideoWriter] = None
        if self.record_full_session:
            self._start_session_recorder()

        self.show_skeleton = True

    def _start_session_recorder(self) -> None:
        """Starts recording the entire continuous camera session."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = os.path.join(self.records_dir, f"video_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.session_writer = cv2.VideoWriter(output_file, fourcc, 20.0, self.frame_size)
        print(f"[CameraRecorder] Full continuous session recording started: {output_file}")

    def record_video(self) -> None:
        """
        Main application loop: reads frames, runs real-time skeleton detection,
        records evidence clips, and displays live GUI.
        """
        frame_idx = 0
        window_title = "Urban Safety AI-SOC: Human Detection & Skeleton Indicator"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        print("\n=======================================================")
        print(" URBAN SAFETY SOC: REAL-TIME HUMAN & SKELETON RECORDER")
        print(" Hotkeys:")
        print("   [q] - Quit and save all video clips")
        print("   [s] - Save high-res snapshot with skeleton overlay")
        print("   [h] - Toggle skeleton and HUD overlay ON/OFF")
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

                # 1. AI Pose & Skeleton Inference
                if self.show_skeleton:
                    analysis, display_frame = self.detector.process_frame(frame, frame_idx)
                else:
                    display_frame = frame.copy()
                    analysis = None

                is_human = analysis.has_human if analysis is not None else False

                # 2. Automated Evidence Clip Recording
                if self.auto_extract_evidence:
                    # Mark recording state on HUD
                    if analysis is not None:
                        analysis.is_recording_evidence = self.evidence_recorder.is_recording

                    self.evidence_recorder.update(display_frame, is_human, now_str)

                # 3. Full Session Continuous Recording (if enabled)
                if self.session_writer is not None:
                    self.session_writer.write(frame)

                # 4. Display Frame
                cv2.imshow(window_title, display_frame)

                # 5. Handle Keyboard Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[CameraRecorder] Stopping camera recorder...")
                    break
                elif key == ord('s'):
                    snap_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    snap_path = os.path.join(self.output_dir, f"snapshot_{snap_time}.jpg")
                    cv2.imwrite(snap_path, display_frame)
                    print(f"[CameraRecorder] Snapshot saved: {snap_path}")
                elif key == ord('h'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"[CameraRecorder] Skeleton Overlay: {'ON' if self.show_skeleton else 'OFF'}")
                elif key == ord('r'):
                    if self.session_writer is None:
                        self._start_session_recorder()
                    else:
                        self.session_writer.release()
                        self.session_writer = None
                        print("[CameraRecorder] Full continuous session recording stopped.")

        finally:
            self._cleanup()

    def _generate_simulated_frame(self, frame_idx: int) -> np.ndarray:
        """Generates synthetic test pattern if camera hardware is not attached."""
        h, w = self.height, self.width
        frame = np.full((h, w, 3), 30, dtype=np.uint8)
        # Grid lines
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (45, 45, 45), 1)
        for y in range(0, h, 40):
            cv2.line(frame, (0, y), (w, y), (45, 45, 45), 1)

        cv2.putText(frame, "[SIMULATED CAMERA STREAM - NO HARDWARE DETECTED]",
                    (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        return frame

    def _cleanup(self) -> None:
        """Releases all camera, writer, and window resources cleanly."""
        if self.cap.isOpened():
            self.cap.release()
        if self.session_writer is not None:
            self.session_writer.release()
        self.evidence_recorder.release()
        self.detector.close()
        cv2.destroyAllWindows()
        print("[CameraRecorder] Resources released successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera Human & Skeleton Detector")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--record-all", action="store_true", help="Record entire camera session to records/")
    parser.add_argument("--no-clips", action="store_true", help="Disable automatic evidence clip extraction")
    args = parser.parse_args()

    recorder = CameraRecorder(
        camera_index=args.camera,
        record_full_session=args.record_all,
        auto_extract_evidence=not args.no_clips
    )
    recorder.record_video()

