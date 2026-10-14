"""
Batch Human Detection & Behavior Processing for Recorded Videos.
Enhanced replacement supporting YOLO11-Pose, ByteTrack, and Behavior Recognition.
Scans records/ folder, detects humans with behavior indicators, and extracts clips to output/.
"""

import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from typing import List, Optional

try:
    from yolo_pose_detector import YOLOPoseDetector
    DEFAULT_DETECTOR = YOLOPoseDetector
except Exception:
    from pose_detector import PoseSkeletonDetector
    DEFAULT_DETECTOR = PoseSkeletonDetector


class BatchHumanDetector:
    """
    Processes pre-recorded video files from records/ directory,
    detects human presence with YOLO11-Pose & Behavior recognition,
    and extracts relevant video clips into output/.
    """

    def __init__(
        self,
        records_dir: str = "records",
        output_dir: str = "output",
        model_weights: str = "yolo11n-pose.pt",
        device: Optional[str] = None,
        annotate_output: bool = True
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_dir = records_dir if os.path.isabs(records_dir) else os.path.join(base_dir, records_dir)
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(base_dir, output_dir)
        self.model_weights = model_weights
        self.device = device
        self.annotate_output = annotate_output

        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            self.detector = YOLOPoseDetector(
                model_path=self.model_weights,
                device=self.device,
                conf_threshold=0.25,
                fps=20.0
            )
            print("[BatchHumanDetector] Initialized with YOLOPoseDetector.")
        except Exception as e:
            print(f"[BatchHumanDetector] Fallback to PoseSkeletonDetector ({e})")
            from pose_detector import PoseSkeletonDetector
            self.detector = PoseSkeletonDetector(
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
                model_complexity=1
            )

    def process_all_videos(self) -> List[str]:
        """Scans records/ and processes every supported video file."""
        supported_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
        video_files = [
            f for f in os.listdir(self.records_dir)
            if os.path.isfile(os.path.join(self.records_dir, f)) and f.lower().endswith(supported_extensions)
        ]

        if not video_files:
            print(f"[BatchHumanDetector] No video files found in '{self.records_dir}/'.")
            return []

        all_outputs = []
        for vf in video_files:
            full_path = os.path.join(self.records_dir, vf)
            outputs = self.process_single_video(full_path)
            all_outputs.extend(outputs)

        return all_outputs

    def process_single_video(self, video_path: str) -> List[str]:
        """
        Analyzes a single video file, finds segments containing humans and behaviors,
        and extracts them into output files using native OpenCV VideoWriter.
        """
        print(f"\n[BatchHumanDetector] Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[BatchHumanDetector] Error: Could not open {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        output_files: List[str] = []

        in_segment = False
        segment_frames: List[np.ndarray] = []
        segment_behaviors: List[str] = []
        segment_index = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            analysis, annotated_frame = self.detector.process_frame(frame, frame_idx)

            target_frame = annotated_frame if self.annotate_output else frame

            if analysis.has_human:
                in_segment = True
                segment_frames.append(target_frame)

                # Collect behavior labels
                if hasattr(analysis, 'humans'):
                    for h in analysis.humans:
                        if hasattr(h, 'behavior'):
                            segment_behaviors.append(h.behavior.primary_behavior.value)

                # Periodic progress log
                if frame_idx % 30 == 0:
                    pct = (frame_idx / max(1, total_frames)) * 100
                    h_count = analysis.human_count
                    threat = getattr(analysis, 'highest_threat_level', 'LOW')
                    print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%) -> {h_count} human(s) | Threat: {threat}")

            else:
                if in_segment:
                    # Finalize current segment
                    segment_index += 1
                    dominant_behavior = "ACTIVITY"
                    if segment_behaviors:
                        from collections import Counter
                        dominant_behavior = Counter(segment_behaviors).most_common(1)[0][0]

                    out_name = f"{video_basename}_seg{segment_index}_{dominant_behavior}.mp4"
                    out_path = os.path.join(self.output_dir, out_name)
                    self._save_frames_to_video(segment_frames, out_path, fps, (width, height))
                    output_files.append(out_path)

                    segment_frames = []
                    segment_behaviors = []
                    in_segment = False

        # Final segment flush if video ended with human present
        if in_segment and segment_frames:
            segment_index += 1
            dominant_behavior = "ACTIVITY"
            if segment_behaviors:
                from collections import Counter
                dominant_behavior = Counter(segment_behaviors).most_common(1)[0][0]

            out_name = f"{video_basename}_seg{segment_index}_{dominant_behavior}.mp4"
            out_path = os.path.join(self.output_dir, out_name)
            self._save_frames_to_video(segment_frames, out_path, fps, (width, height))
            output_files.append(out_path)

        cap.release()
        print(f"[BatchHumanDetector] Done: Extracted {len(output_files)} incident segment(s) to {self.output_dir}/")
        return output_files

    def _save_frames_to_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: float,
        frame_size: tuple
    ) -> None:
        """Writes buffered frames to an MP4 video file safely."""
        if not frames:
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"  -> Extracted incident segment: {os.path.basename(output_path)} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Video Pose & Behavior Extractor")
    parser.add_argument("--records-dir", type=str, default="records", help="Input directory containing video files")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory for incident clips")
    parser.add_argument("--weights", type=str, default="yolo11n-pose.pt", help="YOLO11 weights")
    args = parser.parse_args()

    detector = BatchHumanDetector(records_dir=args.records_dir, output_dir=args.output_dir, model_weights=args.weights)
    detector.process_all_videos()
