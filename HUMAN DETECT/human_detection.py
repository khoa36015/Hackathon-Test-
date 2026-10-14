"""
Batch Human Detection & Skeleton Pose Processing for Recorded Videos.
Enhanced replacement for original reference repo human_detection.py.
Scans records/ folder, detects humans with skeleton indicators, and extracts clips to output/.
"""

import os
import cv2
import argparse
from datetime import datetime
from typing import List, Optional

from pose_detector import PoseSkeletonDetector


class BatchHumanDetector:
    """
    Processes pre-recorded video files from records/ directory,
    detects human presence with pose skeletons, and extracts relevant video clips.
    """

    def __init__(
        self,
        records_dir: str = "records",
        output_dir: str = "output",
        annotate_output: bool = True
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_dir = records_dir if os.path.isabs(records_dir) else os.path.join(base_dir, records_dir)
        self.output_dir = output_dir if os.path.isabs(output_dir) else os.path.join(base_dir, output_dir)
        self.annotate_output = annotate_output

        os.makedirs(self.records_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.detector = PoseSkeletonDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
        Analyzes a single video file, finds segments containing humans,
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

        # Tracking segments
        in_segment = False
        segment_frames: List[cv2.Mat] = []
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
            else:
                if in_segment:
                    # Segment just ended
                    out_path = self._save_segment(segment_frames, video_basename, segment_index, fps, (width, height))
                    if out_path:
                        output_files.append(out_path)
                    segment_index += 1
                    segment_frames = []
                    in_segment = False

            if frame_idx % 30 == 0:
                print(f"[BatchHumanDetector] Progress: {frame_idx}/{total_frames} frames ({(frame_idx / max(1, total_frames)) * 100:.1f}%)")

        # Handle segment extending to the end of the video
        if in_segment and segment_frames:
            out_path = self._save_segment(segment_frames, video_basename, segment_index, fps, (width, height))
            if out_path:
                output_files.append(out_path)

        cap.release()
        print(f"[BatchHumanDetector] Finished {video_path}. Generated {len(output_files)} clip(s).")
        return output_files

    def _save_segment(
        self,
        frames: List[cv2.Mat],
        base_name: str,
        seg_idx: int,
        fps: float,
        size: tuple
    ) -> Optional[str]:
        """Saves accumulated frames into a new MP4 video file."""
        if not frames:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"EVIDENCE_{base_name}_{timestamp}_SEG{seg_idx:02d}.mp4"
        out_path = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, size)

        for f in frames:
            writer.write(f)

        writer.release()
        print(f"[BatchHumanDetector] Saved human segment: {out_path} ({len(frames)} frames)")
        return out_path


def process_videos():
    """Entrypoint function compatible with original reference repo."""
    detector = BatchHumanDetector()
    outputs = detector.process_all_videos()
    print("\nSummary of Generated Output Clips:")
    for out in outputs:
        print(f" -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Human Detection for Recorded Videos")
    parser.add_argument("--records", type=str, default="records", help="Path to records directory")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("--raw", action="store_true", help="Save raw frames instead of skeleton-annotated frames")
    args = parser.parse_args()

    batch_detector = BatchHumanDetector(
        records_dir=args.records,
        output_dir=args.output,
        annotate_output=not args.raw
    )
    batch_detector.process_all_videos()

