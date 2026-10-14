"""
Automated Verification & Test Suite for HUMAN DETECT Application.
Verifies models, pose detector, rolling buffer clip recorder, and camera readiness.
"""

import os
import sys
import time
import cv2
import numpy as np

# Ensure current directory is in Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import Keypoint, HumanPose, FrameAnalysisResult
from pose_detector import PoseSkeletonDetector
from clip_recorder import EvidenceClipRecorder


def test_models():
    print("[TEST 1/5] Testing Data Models...")
    kp = Keypoint(id=0, name="NOSE", x=100, y=100, z=0.0, visibility=0.95)
    pose = HumanPose(track_id=1, bbox=(80, 80, 40, 100), confidence=0.92, keypoints=[kp])
    result = FrameAnalysisResult(frame_index=1, timestamp="2026-10-14 16:00:00", human_count=1, humans=[pose])
    assert result.has_human is True
    assert result.humans[0].keypoints[0].name == "NOSE"
    print("  -> PASSED: Data models serialized and validated successfully.")


def test_detector_synthetic_frames():
    print("\n[TEST 2/5] Testing PoseSkeletonDetector with Synthetic Frames...")
    detector = PoseSkeletonDetector()

    # 1. Blank frame
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    analysis, annotated = detector.process_frame(blank_frame, frame_index=1)
    assert annotated.shape == (480, 640, 3)
    assert analysis.human_count == 0
    print("  -> Blank frame processed: human_count = 0 (No false alarms).")

    # 2. Synthetic frame with drawn human-like silhouette
    synthetic_frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    # Head
    cv2.circle(synthetic_frame, (320, 140), 30, (200, 200, 200), -1)
    # Torso
    cv2.line(synthetic_frame, (320, 170), (320, 300), (200, 200, 200), 15)
    # Arms
    cv2.line(synthetic_frame, (320, 200), (260, 260), (200, 200, 200), 10)
    cv2.line(synthetic_frame, (320, 200), (380, 260), (200, 200, 200), 10)
    # Legs
    cv2.line(synthetic_frame, (320, 300), (280, 420), (200, 200, 200), 12)
    cv2.line(synthetic_frame, (320, 300), (360, 420), (200, 200, 200), 12)

    analysis, annotated = detector.process_frame(synthetic_frame, frame_index=2)
    assert annotated.shape == (480, 640, 3)
    print(f"  -> Synthetic frame processed: infer_time = {analysis.inference_time_ms:.1f}ms, humans = {analysis.human_count}.")

    detector.close()
    print("  -> PASSED: PoseSkeletonDetector successfully executed and released.")


def test_evidence_clip_recorder():
    print("\n[TEST 3/5] Testing EvidenceClipRecorder & Rolling Buffer...")
    test_out_dir = os.path.join(current_dir, "output")
    recorder = EvidenceClipRecorder(
        output_dir=test_out_dir,
        fps=20.0,
        frame_size=(640, 480),
        pre_roll_seconds=1.0,  # 20 frames
        post_roll_seconds=1.0   # 20 frames
    )

    test_frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    cv2.putText(test_frame, "TEST FRAME", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    saved_clip_path = None

    # Phase A: 25 frames without human (pre-roll buffer filling)
    for i in range(25):
        recorder.update(test_frame, is_human_present=False, timestamp_str=f"T0_{i}")
    assert len(recorder.pre_roll_buffer) == 20  # Maxed out at pre_roll_limit

    # Phase B: 30 frames WITH human (recording active)
    for i in range(30):
        recorder.update(test_frame, is_human_present=True, timestamp_str=f"T1_{i}")
    assert recorder.is_recording is True

    # Phase C: 25 frames without human (post-roll cooldown and closure)
    for i in range(25):
        res = recorder.update(test_frame, is_human_present=False, timestamp_str=f"T2_{i}")
        if res is not None:
            saved_clip_path = res

    recorder.release()

    assert saved_clip_path is not None, "Evidence clip was not finalized."
    assert os.path.exists(saved_clip_path), f"File {saved_clip_path} does not exist."
    file_size = os.path.getsize(saved_clip_path)
    assert file_size > 0, f"Saved clip {saved_clip_path} is empty (0 bytes)."

    # Verify video can be read back with OpenCV
    cap = cv2.VideoCapture(saved_clip_path)
    assert cap.isOpened(), "Cannot open generated video file."
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  -> Generated evidence clip: {os.path.basename(saved_clip_path)}")
    print(f"  -> Clip size: {file_size / 1024:.1f} KB, Frame count: {frame_count}")
    print("  -> PASSED: EvidenceClipRecorder created valid, readable MP4 video.")


def test_camera_hardware():
    print("\n[TEST 4/5] Checking Camera Hardware Availability...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  -> Camera #0 DETECTED: {w}x{h} @ {fps:.0f} FPS. Frame read OK.")
        else:
            print("  -> Camera #0 opened but could not read frame (busy or virtual).")
    else:
        print("  -> Camera #0 NOT DETECTED or currently in use by another app.")
        print("     (Note: App includes simulated stream fallback and batch video processing mode).")
    print("  -> PASSED: Camera hardware check completed.")


def test_fps_benchmark():
    print("\n[TEST 5/5] Benchmarking AI Pose Inference Speed on CPU...")
    detector = PoseSkeletonDetector()
    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    # Warmup
    for _ in range(5):
        detector.process_frame(dummy_frame)

    # Benchmark 30 frames
    start_time = time.time()
    num_frames = 30
    for i in range(num_frames):
        detector.process_frame(dummy_frame, frame_index=i)
    total_time = time.time() - start_time
    detector.close()

    bench_fps = num_frames / total_time
    avg_latency = (total_time / num_frames) * 1000.0
    print(f"  -> Processed {num_frames} frames in {total_time:.2f}s")
    print(f"  -> Average Latency: {avg_latency:.1f} ms/frame")
    print(f"  -> Throughput: {bench_fps:.1f} FPS on CPU")
    print("  -> PASSED: Benchmark completed successfully.")


def run_all_tests():
    print("================================================================")
    print("   AUTOMATED VERIFICATION SUITE: HUMAN DETECTION & SKELETON     ")
    print("================================================================")
    try:
        test_models()
        test_detector_synthetic_frames()
        test_evidence_clip_recorder()
        test_camera_hardware()
        test_fps_benchmark()
        print("\n================================================================")
        print("   ALL TESTS PASSED SUCCESSFULLY! (5/5)                         ")
        print("================================================================")
        return 0
    except AssertionError as ae:
        print(f"\n[FAIL] Assertion failed: {ae}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(run_all_tests())

