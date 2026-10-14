"""
Evidence Clip Recorder Module for Urban Safety AI-SOC.
Implements a Non-blocking Rolling Buffer to automatically record and extract
evidence clips (pre-roll + event + post-roll) upon human detection.
Uses OpenCV VideoWriter natively without relying on external ffmpeg binary.
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime
from collections import deque
from typing import Tuple, Optional


class EvidenceClipRecorder:
    """
    Automated evidence recorder using a temporal sliding window buffer.
    """

    def __init__(
        self,
        output_dir: str = "output",
        fps: float = 20.0,
        frame_size: Tuple[int, int] = (640, 480),
        pre_roll_seconds: float = 1.5,
        post_roll_seconds: float = 2.0
    ) -> None:
        self.output_dir = output_dir
        self.fps = fps if fps > 0 else 20.0
        self.frame_size = frame_size

        os.makedirs(self.output_dir, exist_ok=True)

        self.pre_roll_limit = int(self.fps * pre_roll_seconds)
        self.post_roll_limit = int(self.fps * post_roll_seconds)

        self.pre_roll_buffer = deque(maxlen=self.pre_roll_limit)

        self.is_recording = False
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_output_path: Optional[str] = None
        self.cooldown_counter = 0
        self.clip_start_time: Optional[str] = None
        self.clip_frame_count = 0
        self.event_counter = 0

    def update(
        self,
        frame: np.ndarray,
        is_human_present: bool,
        timestamp_str: str
    ) -> Optional[str]:
        """
        Updates recorder state with current frame.
        Returns:
            output_filepath if a clip was just finalized and saved, else None.
        """
        finalized_clip = None

        if is_human_present:
            # Human is visible in current frame
            if not self.is_recording:
                # Trigger start recording
                self._start_recording(timestamp_str)

            # Write current frame
            self.writer.write(frame)
            self.clip_frame_count += 1
            self.cooldown_counter = 0  # Reset cooldown

        else:
            # No human detected in current frame
            if self.is_recording:
                # In cooldown / post-roll period
                self.writer.write(frame)
                self.clip_frame_count += 1
                self.cooldown_counter += 1

                if self.cooldown_counter >= self.post_roll_limit:
                    # Finalize clip
                    finalized_clip = self._finalize_clip(timestamp_str)
            else:
                # Normal idle monitoring: push into rolling pre-roll buffer
                self.pre_roll_buffer.append(frame.copy())

        return finalized_clip

    def _start_recording(self, timestamp_str: str) -> None:
        """Starts a new video recording session and flushes pre-roll buffer."""
        self.event_counter += 1
        safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_path = os.path.join(self.output_dir, f"EVIDENCE_{safe_time}_HUMAN.mp4")
        self.clip_start_time = timestamp_str
        self.clip_frame_count = 0

        # Try mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.current_output_path,
            fourcc,
            self.fps,
            self.frame_size
        )

        # Flush pre-roll buffer frames into video file
        while len(self.pre_roll_buffer) > 0:
            past_frame = self.pre_roll_buffer.popleft()
            self.writer.write(past_frame)
            self.clip_frame_count += 1

        self.is_recording = True
        print(f"[EvidenceClipRecorder] >>> TRIGGERED RECORDING: {self.current_output_path}")

    def _finalize_clip(self, end_timestamp_str: str) -> Optional[str]:
        """Closes video file and logs audit record."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        saved_path = self.current_output_path
        duration = self.clip_frame_count / self.fps if self.fps > 0 else 0.0

        # Write SOC Audit Event Log
        log_file = os.path.join(self.output_dir, "detection_events.jsonl")
        event_record = {
            "event_id": f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.event_counter:03d}",
            "type": "HUMAN_PRESENCE_DETECTED",
            "start_time": self.clip_start_time,
            "end_time": end_timestamp_str,
            "duration_seconds": round(duration, 2),
            "frame_count": self.clip_frame_count,
            "output_file": saved_path
        }

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record) + "\n")
        except Exception as e:
            print(f"[EvidenceClipRecorder] Warning: Could not write event log ({e})")

        print(f"[EvidenceClipRecorder] <<< SAVED EVIDENCE CLIP: {saved_path} ({duration:.1f}s, {self.clip_frame_count} frames)")

        self.is_recording = False
        self.current_output_path = None
        self.cooldown_counter = 0
        return saved_path

    def release(self) -> None:
        """Safely cleans up any active recording session."""
        if self.is_recording and self.writer is not None:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._finalize_clip(now_str)

