# KẾ HOẠCH TRIỂN KHAI: ỨNG DỤNG PHÁT HIỆN CON NGƯỜI & VẼ KHUNG XƯƠNG REAL-TIME (STAGE 2 — PLAN)
**Mã tài liệu:** `PLAN-HUMAN-DETECT-01`  
**Dự án:** Hệ thống Giám sát An ninh & An toàn Đô thị Thông minh (Urban Safety AI-SOC)  
**Phân hệ:** Tầng 1 (Input & Ingestion) & Tầng 2 (AI/CV Detection & Skeleton Pose Indicator)  
**Thư mục làm việc:** `HUMAN DETECT/`  
**Căn cứ pháp lý & quy trình:** Tuân thủ `AGENTS.md` (Stage 2 SDD & Strict Architectural Lock) và bản đặc tả `SPEC.md` v1.1.

---

## 1. DANH MỤC TỆP TIN THỰC HIỆN (FILE MANIFEST)

| Thao tác | Đường dẫn tệp | Mục đích / Trách nhiệm chính |
| :--- | :--- | :--- |
| **Tạo mới** | `HUMAN DETECT/requirements.txt` | Khai báo các thư viện phụ thuộc (`mediapipe`, `opencv-python`, `numpy`). |
| **Tạo mới** | `HUMAN DETECT/models.py` | Định nghĩa schema dữ liệu chuẩn (`Keypoint`, `HumanPose`, `FrameAnalysisResult`, `HumanIncidentClip`). |
| **Tạo mới** | `HUMAN DETECT/pose_detector.py` | Lớp `PoseSkeletonDetector` tích hợp MediaPipe Pose, tính toán 33 khớp xương, kết nối khung xương và vẽ chỉ báo SOC. |
| **Tạo mới** | `HUMAN DETECT/clip_recorder.py` | Lớp `EvidenceClipRecorder` quản lý bộ đệm vòng (Rolling Buffer), tự động trích xuất clip MP4 độc lập không phụ thuộc ffmpeg. |
| **Tạo mới** | `HUMAN DETECT/camera_recorder.py` | Ứng dụng giám sát Camera thời gian thực (Live Stream + Skeleton Indicator + Tự động lưu bằng chứng khi có người). |
| **Tạo mới** | `HUMAN DETECT/human_detection.py` | Ứng dụng quét hàng loạt video ngoại tuyến trong `records/` và trích xuất đoạn có người (tương thích nâng cấp từ repo gốc). |
| **Tạo mới** | `HUMAN DETECT/main.py` | Entrypoint điều khiển giao diện dòng lệnh (CLI Runner) cho phép chọn chế độ Live hoặc Batch. |
| **Tạo mới** | `HUMAN DETECT/test_system.py` | Bộ kịch bản kiểm thử tự động (Unit Test, Synthetic Frame Test, Buffer Test, FPS Benchmark). |
| **Tạo mới** | `HUMAN DETECT/records/.gitkeep` | Thư mục chứa video đầu vào (tương thích cấu trúc repo gốc). |
| **Tạo mới** | `HUMAN DETECT/output/.gitkeep` | Thư mục chứa video clip và audit log đầu ra. |

---

## 2. DANH SÁCH CLASS, HÀM & CHỮ KÝ PHƯƠNG THỨC (FUNCTION SIGNATURES)

### 2.1. Module `models.py`
Đảm bảo tính chuẩn hóa dữ liệu theo nguyên tắc Strict Structured Output của AI-SOC:

```python
@dataclass
class Keypoint:
    id: int
    name: str
    x: int
    y: int
    z: float
    visibility: float

@dataclass
class HumanPose:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    keypoints: List[Keypoint] = field(default_factory=list)
    posture_label: str = "Standing"

@dataclass
class FrameAnalysisResult:
    frame_index: int
    timestamp: str
    human_count: int
    humans: List[HumanPose] = field(default_factory=list)
    inference_time_ms: float = 0.0
    is_recording_evidence: bool = False
```

### 2.2. Module `pose_detector.py` (`PoseSkeletonDetector`)
Phụ trách suy luận AI thời gian thực và vẽ chỉ báo hình học:

```python
class PoseSkeletonDetector:
    def __init__(
        self, 
        min_detection_confidence: float = 0.5, 
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ) -> None: ...

    def process_frame(
        self, 
        frame_bgr: np.ndarray, 
        frame_index: int = 0
    ) -> Tuple[FrameAnalysisResult, np.ndarray]:
        """
        Nhận frame BGR, suy luận qua MediaPipe Pose, 
        trả về kết quả phân tích cấu trúc và frame đã vẽ đầy đủ khung xương + chỉ báo.
        """
        ...

    def draw_skeleton_and_indicators(
        self, 
        frame_bgr: np.ndarray, 
        analysis: FrameAnalysisResult
    ) -> np.ndarray:
        """
        Vẽ:
        1. Các khớp (Keypoint Circles với màu sắc theo độ tin cậy).
        2. Các đoạn xương nối (Bones Line nối vai, tay, thân, hông, chân).
        3. Hộp bao quanh (Bounding Box) kèm nhãn 'HUMAN #ID | CONF: XX%'.
        4. Bảng điều khiển HUD (FPS, Human Count, Trạng thái SOC, Đồng hồ thời gian thực).
        """
        ...

    def _estimate_posture(self, keypoints: List[Keypoint]) -> str:
        """Phân tích góc hông - gối - vai để xác định trạng thái Đứng / Ngồi / Di chuyển."""
        ...

    def close(self) -> None:
        """Giải phóng tài nguyên mô hình."""
        ...
```

### 2.3. Module `clip_recorder.py` (`EvidenceClipRecorder`)
Quản lý bộ đệm xoay vòng và xuất file clip MP4 chuẩn SOC:

```python
class EvidenceClipRecorder:
    def __init__(
        self,
        output_dir: str = "output",
        fps: float = 20.0,
        frame_size: Tuple[int, int] = (640, 480),
        pre_roll_seconds: float = 1.5,
        post_roll_seconds: float = 2.0
    ) -> None: ...

    def update(
        self, 
        annotated_frame: np.ndarray, 
        is_human_present: bool, 
        timestamp_str: str
    ) -> Optional[str]:
        """
        Đẩy frame vào pipeline. 
        Nếu phát hiện người -> kích hoạt ghi hoặc xả pre-roll.
        Nếu mất dấu người -> đếm cooldown post-roll rồi đóng file MP4.
        Trả về đường dẫn file đã đóng nếu vừa hoàn tất một clip.
        """
        ...

    def release(self) -> None:
        """Đóng an toàn video writer nếu đang ghi dở khi tắt app."""
        ...
```

### 2.4. Module `camera_recorder.py` & `human_detection.py`
- `camera_recorder.py`:
  - `class LiveCameraSkeletonApp`:
    - `__init__(camera_index=0, width=640, height=480, record_all=False, auto_extract_clips=True)`
    - `run()`: Vòng lặp đọc khung hình $\to$ suy luận khung xương $\to$ vẽ HUD $\to$ ghi bằng chứng $\to$ hiển thị `cv2.imshow`. Phím `q` thoát, phím `s` chụp snapshot.
- `human_detection.py`:
  - `class BatchVideoPoseDetector`:
    - `__init__(records_dir="records", output_dir="output")`
    - `process_all_records() -> List[str]`
    - `process_video(video_path: str) -> List[str]`

---

## 3. DANH SÁCH THƯ VIỆN & GIẢI TRÌNH KIẾN TRÚC

1. **`mediapipe` (v1.1.0+):**
   - *Lý do:* Giải pháp hàng đầu thế giới về Pose Estimation trên CPU. Cung cấp 33 điểm mốc 3D (BlazePose), chạy đạt 25–30 FPS ngay trên vi xử lý thông thường không cần card đồ họa rời.
   - *Đã xác thực:* Có bản build tương thích hoàn toàn với Python 3.15 trên hệ điều hành Windows.
2. **`opencv-python` (v5.0.0+):**
   - *Lý do:* Thư viện nền tảng cho Computer Vision: đọc luồng webcam, chuyển đổi không gian màu, vẽ đồ họa thời gian thực (đường nối xương, text, HUD), mã hóa xuất file video MP4 mà không phụ thuộc vào công cụ `ffmpeg` cài ngoài hệ điều hành.
3. **`numpy` (v2.5.4+):**
   - *Lý do:* Thao tác mảng dữ liệu ảnh và tính toán hình học tọa độ điểm mốc.

---

## 4. KỊCH BẢN KIỂM THỬ TỪNG BƯỚC (VERIFICATION & TEST PLAN)

### Bước 1: Kiểm thử Môi trường & Dependencies (Phase 1 Test)
- Chạy cài đặt: `pip install -r "HUMAN DETECT/requirements.txt"`
- Thực thi script kiểm tra import: Đảm bảo `cv2`, `mediapipe`, `numpy` nạp thành công không bị lỗi DLL.

### Bước 2: Kiểm thử Đơn vị với Khung hình Tổng hợp (Synthetic Frame Test)
- Sử dụng module `test_system.py` để sinh khung hình giả lập không cần camera vật lý.
- Đưa qua `PoseSkeletonDetector.process_frame()` để kiểm chứng:
  - Khả năng xử lý trường hợp không có người (empty frame $\to$ `human_count == 0`).
  - Khả năng render giao diện HUD, đồng hồ timestamp mà không phát sinh lỗi index out of bounds.

### Bước 3: Kiểm thử Bộ đệm & Tự động Trích xuất Clip (Buffer & Clip Verification)
- Mô phỏng chuỗi 90 frames:
  - Khung 1–30: Không có người (nạp vào Pre-roll Buffer).
  - Khung 31–60: Giả lập xuất hiện người (kích hoạt mở file `output/video_{timestamp}_human.mp4`).
  - Khung 61–90: Không có người (kích hoạt Post-roll Cooldown và đóng file).
- Xác minh file MP4 sinh ra trong thư mục `output/` có kích thước $> 0$ bytes và mở đọc lại được bằng OpenCV.

### Bước 4: Kiểm thử Tích hợp Thực tế trên Camera (Live Camera & Skeleton Verification)
- Chạy lệnh: `python "HUMAN DETECT/main.py" --mode live`
- Kiểm tra các tiêu chí chấp nhận (Acceptance Criteria):
  1. Cửa sổ `Urban Safety Human & Skeleton Monitor` hiển thị luồng video từ webcam.
  2. Khi có người trước camera: Khung xương màu sắc (các đường nối vai, khuỷu tay, đầu gối, bàn tay) bám sát chuyển động cơ thể trong thời gian thực.
  3. Bounding box bao quanh cơ thể kèm nhãn độ tin cậy.
  4. Đồng hồ timestamp và chỉ số FPS ($\ge 20$ FPS) hiển thị rõ ràng góc màn hình.
  5. Khi nhấn phím `s`: Ảnh chụp bằng chứng được lưu ngay vào `output/snapshot_{timestamp}.jpg`.
  6. Khi nhấn phím `q`: Ứng dụng thoát sạch tài nguyên mà không bị treo tiến trình (hang process).

---

## 5. BƯỚC TIẾP THEO THEO QUY TRÌNH SDD

Theo quy định tại **Mục 2 — Stage 3 trong `AGENTS.md`**, Agent hiện đang ở trạng thái **ĐÓNG BĂNG SINH CODE (Code Generation Freeze)**.  
Agent chỉ được phép tiến hành tạo các file code và chạy kiểm thử ở **Bước 4 (Stage 4 — Code & Verify)** sau khi nhận được lệnh phê duyệt tường minh từ bạn.

---

[WAITING FOR PLAN APPROVAL]

