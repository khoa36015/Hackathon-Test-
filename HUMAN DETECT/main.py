"""
Unified CLI Entrypoint for Human Detection & Pose Skeleton Application.
Urban Safety AI-SOC Edition.
"""

import sys
import argparse

from camera_recorder import CameraRecorder
from human_detection import BatchHumanDetector


def main():
    parser = argparse.ArgumentParser(
        description="Urban Safety AI-SOC: Human Detection & Pose Skeleton Indicator System",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["live", "batch"],
        default="live",
        help="Operating mode:\n"
             "  live  : Real-time camera streaming with skeleton indicator & auto evidence clipping (default)\n"
             "  batch : Scan pre-recorded videos from records/ directory and extract human segments\n"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for live mode (default: 0)"
    )

    parser.add_argument(
        "--record-all",
        action="store_true",
        help="In live mode, also record the entire session continuously into records/"
    )

    parser.add_argument(
        "--records-dir",
        type=str,
        default="records",
        help="Directory containing input videos for batch mode (default: records)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save extracted evidence clips and snapshots (default: output)"
    )

    parser.add_argument(
        "--max-humans",
        type=int,
        default=10,
        help="Maximum number of people to detect and track simultaneously (default: 10)"
    )

    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence threshold for multi-person detection (default: 0.25)"
    )

    args = parser.parse_args()

    print("=================================================================")
    print("      URBAN SAFETY AI-SOC: MULTI-PERSON SKELETON INDICATOR       ")
    print("=================================================================")
    print(f" Mode       : {args.mode.upper()}")
    print(f" Max Humans : {args.max_humans}")
    print(f" Conf Thresh: {args.conf_thresh}")
    print(f" Output Dir : {args.output_dir}")
    print("=================================================================\n")

    if args.mode == "live":
        app = CameraRecorder(
            camera_index=args.camera,
            record_full_session=True,  # Ghi hình toàn bộ phiên có gắn khung xương
            max_humans=args.max_humans,
            min_confidence=args.conf_thresh,
            output_dir=args.output_dir,
            records_dir=args.records_dir
        )
        app.record_video()

    elif args.mode == "batch":
        detector = BatchHumanDetector(
            records_dir=args.records_dir,
            output_dir=args.output_dir
        )
        detector.process_all_videos()


if __name__ == "__main__":
    main()

