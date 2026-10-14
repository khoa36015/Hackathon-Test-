# KẾ HOẠCH TRIỂN KHAI: GHI LOG KIỂM TOÁN, ALERT BOX DEBUG & KHỬ NHẢY MARK (STAGE 2 — PLAN)
**Mã tài liệu:** `PLAN-BEHAVIOR-STABILIZATION-02`  
**Dự án:** Hệ thống Giám sát An ninh & An toàn Đô thị Thông minh (Urban Safety AI-SOC)  
**Phân hệ:** Tầng 2 (AI Analytics), Tầng 3 (Dynamic Threat Scoring), Tầng 5 (Audit & Observability)  
**Thư mục làm việc:** `HUMAN DETECT/`  
**Căn cứ pháp lý & quy trình:** Tuân thủ `AGENTS.md` (Stage 2 SDD) và `SPEC-BEHAVIOR-STABILIZATION-02`.

---

## 1. DANH MỤC TỆP TIN THỰC HIỆN (FILE MANIFEST)

| Thao tác | Đường dẫn tệp | Mục đích / Trách nhiệm chính |
| :--- | :--- | :--- |
| **Cập nhật** | `HUMAN DETECT/models.py` | Bổ sung dataclass: `BehaviorAuditLogEntry` (lưu trữ kiểm toán) và `TrackKinematicsTelemetry` (dữ liệu đo xa cho Debug Box). |
| **Cập nhật** | `HUMAN DETECT/behavior_detector.py` | 1. Thêm lớp `BehaviorAuditLogger`: Ghi log JSONL và text log an toàn, có rate-limiting chống spam.<br>2. Thêm bộ lọc EMA cho Bbox và góc/tốc độ.<br>3. Thêm Máy trạng thái trễ (Hysteresis Debounce: $N_{\text{enter}}=8, N_{\text{exit}}=12$) chống đổi nhãn đột ngột. |
| **Cập nhật** | `HUMAN DETECT/yolo_pose_detector.py` | 1. Tinh chỉnh tham số ByteTrack (`lost_track_buffer=45`, `match_thresh=0.80`).<br>2. Thêm phương thức `_draw_debug_alert_box()` render khung Diagnostic HUD ở góc dưới màn hình.<br>3. Thêm cờ `show_debug_box` và phương thức `toggle_debug_box()`. |
| **Cập nhật** | `HUMAN DETECT/camera_recorder.py` | Bổ sung phím tắt `[d]` để bật/tắt Alert Box Debug trong luồng camera trực tiếp, in hướng dẫn trên terminal banner. |
| **Cập nhật** | `HUMAN DETECT/human_detection.py` | Kích hoạt `BehaviorAuditLogger` cho chế độ quét batch video ngoại tuyến. |
| **Cập nhật** | `HUMAN DETECT/test_system.py` | Bổ sung các bài test tự động: kiểm chứng khử rung (Anti-Flapping), kiểm tra định dạng file `behavior_events.jsonl`, và kiểm tra render Debug Alert Box. |
| **Cập nhật** | `HUMAN DETECT/README.md` | Hướng dẫn sử dụng phím tắt `[d]`, định dạng file log kiểm toán, và giải thích cơ chế chống nhảy mark. |

---

## 2. CHỮ KÝ HÀM & THIẾT KẾ CHI TIẾT (FUNCTION SIGNATURES)

### 2.1. Module `models.py`
```python
@dataclass
class BehaviorAuditLogEntry:
    event_id: str                      # EVT_YYYYMMDD_HHMMSS_TRACKID
    timestamp: str                     # ISO-8601
    frame_index: int
    track_id: int
    event_type: str                    # STATE_TRANSITION | THREAT_ESCALATION | CRITICAL_ALERT | HEARTBEAT
    previous_behavior: Optional[str]
    current_behavior: str
    threat_level: str
    threat_score: float
    telemetry: Dict[str, float]
    reasoning: str
    dwell_time_seconds: float

@dataclass
class TrackKinematicsTelemetry:
    track_id: int
    spine_angle: float
    aspect_ratio: float
    speed: float
    wrist_energy: float
    dwell_time: float
    confirmed_behavior: str
    threat_level: ThreatLevel
```

### 2.2. Module `behavior_detector.py`
```python
class BehaviorAuditLogger:
    def __init__(self, output_dir: str = "output", heartbeat_interval: float = 1.0) -> None: ...
    def log_event(self, entry: BehaviorAuditLogEntry) -> None: ...
    def close(self) -> None: ...

class BehaviorEngine:
    def __init__(
        self,
        fps: float = 20.0,
        enter_debounce_frames: int = 8,    # Số frames liên tục để xác nhận hành vi mới
        exit_debounce_frames: int = 12,    # Số frames liên tục để hoàn nguyên về Normal
        bbox_ema_alpha: float = 0.65,      # Trọng số làm mượt hộp bao (0.0 - 1.0)
        output_dir: str = "output",
        ...
    ) -> None: ...

    def update(
        self,
        tracked_poses: List[HumanPoseTrack],
        timestamp: Optional[float] = None,
        frame_index: int = 0
    ) -> Tuple[List[HumanPoseTrack], List[TrackKinematicsTelemetry], List[str]]:
        """
        Trả về:
        1. Danh sách poses đã làm mượt bbox và gắn confirmed_behavior ổn định.
        2. Dữ liệu đo xa telemetry cho Alert Box Debug.
        3. Danh sách sự kiện cảnh báo mới nhất (recent_alerts).
        """
        ...
```

### 2.3. Module `yolo_pose_detector.py`
```python
class YOLOPoseDetector:
    def __init__(...):
        self.show_debug_box: bool = True
        ...

    def toggle_debug_box(self) -> bool:
        """Bật / Tắt hiển thị khung Alert Box Debug HUD."""
        self.show_debug_box = not self.show_debug_box
        return self.show_debug_box

    def _draw_debug_alert_box(
        self,
        frame: np.ndarray,
        telemetry_list: List[TrackKinematicsTelemetry],
        recent_alerts: List[str]
    ) -> None:
        """Vẽ khung Diagnostic HUD bán trong suốt ở góc dưới màn hình."""
        ...
```

---

## 3. KỊCH BẢN KIỂM THỬ XÁC MINH (VERIFICATION & TEST PLAN)

1. **Test 1: Kiểm thử Khử Nhảy Nhãn (Anti-Flapping Test):**
   - Bơm vào `BehaviorEngine` chuỗi 15 frames:
     - Frames 1–5: `NORMAL`
     - Frames 6–7: Giả lập nhiễu 2 frames cúi người `CROUCHING`
     - Frames 8–15: `NORMAL`
   - **Kỳ vọng:** `confirmed_behavior` phải giữ vững `NORMAL` 100%, không bị chập chờn sang `CROUCHING`.
2. **Test 2: Kiểm thử Xác nhận Chuyển Trạng thái (Hysteresis Debounce Test):**
   - Bơm vào `BehaviorEngine` chuỗi 10 frames nằm ngang liên tiếp ($\ge N_{\text{enter}}=8$).
   - **Kỳ vọng:** Trạng thái chuyển thành công sang `FALLEN`, kích hoạt ghi log `STATE_TRANSITION`.
3. **Test 3: Kiểm thử Ghi Nhật ký Kiểm toán (Audit Logger Test):**
   - Kiểm tra tệp `output/behavior_events.jsonl` được tạo, mở đọc và parse JSON từng dòng.
   - Xác nhận có đủ các trường: `event_id`, `timestamp`, `track_id`, `event_type`, `telemetry`.
4. **Test 4: Kiểm thử Alert Box Debug Renderer (Debug HUD Test):**
   - Gọi `_draw_debug_alert_box` với telemetry giả lập.
   - Xác nhận không phát sinh lỗi kích thước mảng ảnh, văn bản hiển thị đầy đủ, màu sắc chuẩn xác.
5. **Test 5: Kiểm thử Phím tắt `[d]` (Hotkeys Integration Test):**
   - Kiểm tra phương thức `toggle_debug_box()` chuyển đổi trạng thái `True <-> False` chính xác.

---

[WAITING FOR PLAN APPROVAL]
