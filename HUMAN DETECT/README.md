# URBAN SAFETY AI-SOC: HUMAN DETECTION, POSE & BEHAVIOR RECOGNITION
**Kiến trúc:** YOLO11-Pose + Supervision ByteTrack + Biomechanical Behavior Engine  
**Dự án:** Hệ thống Giám sát An ninh & An toàn Đô thị Thông minh (Urban Safety AI-SOC)  
**Tầng nghiệp vụ:** Tầng 1 (Input/Ingestion), Tầng 2 (AI/Pose Analytics), Tầng 3 (Dynamic Threat Scoring)

---

## 1. CÁC TÍNH NĂNG NỔI BẬT

1. **Bộ Nhận diện Khung xương Tiên tiến YOLO11-Pose (Single-Stage 17 COCO Keypoints):**
   - Định vị chính xác 17 khớp giải phẫu chuẩn quốc tế (Mũi, Mắt, Tai, Vai, Khuỷu tay, Cổ tay, Hông, Đầu gối, Cổ chân).
   - Tối ưu hóa cực đại cho cả vi xử lý CPU và card đồ họa GPU rời (Nvidia CUDA).
   - Không bị tráo xương, không bị chồng lấn chi giữa nhiều người đứng sát nhau.

2. **Thuật toán Bám vết Đa đối tượng Supervision ByteTrack:**
   - Liên kết phát hiện 2 giai đoạn (High & Low confidence association).
   - Bảo toàn mã định danh duy nhất (`track_id`), loại bỏ triệt để hiện tượng tráo ID (ID flipping) khi người đi cắt chéo nhau hoặc bị che khuất tạm thời (occlusion).

3. **Động cơ Nhận diện Hành vi Động học Chuỗi Thời gian (Temporal Behavior Engine):**
   - Duy trì bộ nhớ trượt 30 khung hình cho từng `track_id`, tính toán góc nghiêng cột sống ($\theta_{\text{spine}}$), vận tốc rơi hông, động năng cổ tay, và khoảng cách tương tác liên cá thể.
   - Nhận diện 7 hành vi đô thị với 4 cấp độ đe dọa (Dynamic Threat Levels):
     - **`FALLEN`** (Té ngã / Nằm bất động — Nguy cơ: **`CRITICAL`** [Đỏ]).
     - **`FIGHTING`** (Ẩu đả / Tấn công / Xô xát — Nguy cơ: **`CRITICAL`** [Đỏ]).
     - **`RUNNING` / `FLEEING`** (Bỏ chạy / Tăng tốc đột ngột — Nguy cơ: **`HIGH`** [Cam]).
     - **`HANDS_UP`** (Giơ tay cầu cứu / Đầu hàng — Nguy cơ: **`HIGH`** [Cam]).
     - **`LOITERING`** (Lảng vảng bất thường khu vực cấm — Nguy cơ: **`MEDIUM` $\to$ `HIGH`** [Vàng/Cam]).
     - **`CROUCHING`** (Cúi khom / Lén lút / Đột nhập — Nguy cơ: **`MEDIUM`** [Vàng]).
     - **`NORMAL`** (Đi đứng sinh hoạt bình thường — Nguy cơ: **`LOW`** [Xanh lá]).

4. **Bảng điều khiển HUD Giám sát & Đồng hồ Thời gian thực Kiểm toán:**
   - Hiển thị trực tiếp Timestamp (`YYYY-MM-DD HH:MM:SS.mmm`) trên từng khung hình phục vụ kiểm toán chuẩn SOC.
   - Hộp bao giới hạn viền góc hiện đại (Sleek corner-accented bounding boxes) phân màu theo mức độ đe dọa.
   - Huy hiệu cảnh báo khẩn cấp khi phát hiện sự kiện nguy hiểm trên màn hình.

5. **Bộ đệm Bằng chứng Tự động (Evidence Rolling Buffer):**
   - Duy trì bộ đệm vòng (1.5s trước sự kiện + 2.0s sau sự kiện).
   - Tự động khóa và trích xuất video clip MP4 vào thư mục `output/` khi có đối tượng đạt mức đe dọa cao hoặc khẩn cấp.
   - Sử dụng OpenCV native `VideoWriter` độc lập, không phụ thuộc vào `ffmpeg` cài ngoài.

---

## 2. CÀI ĐẶT MÔI TRƯỜNG

Môi trường khuyến nghị: **Python 3.10+**.

```bash
cd "HUMAN DETECT"
pip install -r requirements.txt
```

---

## 3. HƯỚNG DẪN SỬ DỤNG

### 3.1. Chế độ Giám sát Trực tiếp từ Camera (Live Mode)

Khởi chạy webcam mặc định (Camera 0) với mô hình YOLO11n-Pose:
```bash
python main.py --mode live
```

Chỉ định camera khác và thiết bị GPU CUDA:
```bash
python main.py --mode live --camera 1 --device cuda
```

Ghi toàn bộ phiên làm việc liên tục vào `records/` đồng thời tự động cắt clip bằng chứng khi có sự cố:
```bash
python main.py --mode live --record-all
```

#### Phím tắt điều khiển khi xem Camera:
| Phím tắt | Chức năng |
| :---: | :--- |
| **`q`** | Thoát ứng dụng và hoàn tất các video clip an toàn. |
| **`s`** | Chụp ảnh Snapshot độ nét cao kèm khung xương và nhãn hành vi lưu vào `output/snapshot_{timestamp}.jpg`. |
| **`h`** | Bật / Tắt lớp hiển thị Khung xương & HUD. |
| **`d`** | Bật / Tắt (Toggle) Bảng **Alert Box Debug HUD** hiển thị đo xa telemetry và sự kiện tức thời. |
| **`r`** | Bật / Tắt chế độ ghi hình toàn bộ phiên làm việc. |

---

### 3.2. Chế độ Quét Hàng loạt Video Ngoại tuyến (Batch Mode)

Sao chép các file video cần phân tích (`.mp4`, `.avi`, `.mov`, `.mkv`) vào thư mục `records/`, sau đó chạy:
```bash
python main.py --mode batch
```
Ứng dụng sẽ tự động phân tích từng video, bám vết các đối tượng, nhận diện hành vi và trích xuất các phân đoạn sự cố ra thư mục `output/` kèm nhãn hành vi (ví dụ: `video1_seg1_FALLEN.mp4`, `video1_seg2_FIGHTING.mp4`).

---

### 3.3. Chạy Bộ Kiểm thử Tự động (Automated Test Suite)

Để kiểm chứng toàn bộ data models, các quy tắc nhận diện hành vi động học, bộ đệm ghi hình, và pipeline:
```bash
python test_system.py
```

---

## 4. CẤU TRÚC THƯ MỤC

```
HUMAN DETECT/
├── records/              # Thư mục chứa video đầu vào hoặc video ghi liên tục
├── output/               # Thư mục lưu video clip bằng chứng (.mp4) và snapshot (.jpg)
├── models.py             # Data schema chuẩn SOC (KeypointCOCO, BehaviorPrediction, SOCFrameAnalysisResult)
├── behavior_detector.py  # Động cơ phân tích động học & nhận diện 7 hành vi đô thị (BehaviorEngine)
├── yolo_pose_detector.py # Bộ tích hợp YOLO11-Pose + Supervision ByteTrack + SOC HUD
├── pose_detector.py      # Module MediaPipe Pose dự phòng (Fallback engine)
├── clip_recorder.py      # Quản lý Rolling Buffer & Trích xuất video clip bằng chứng
├── camera_recorder.py    # Ứng dụng Camera Streamer thời gian thực
├── human_detection.py    # Bộ xử lý batch video ngoại tuyến
├── main.py               # Entrypoint điều khiển CLI thống nhất
├── test_system.py        # Kịch bản kiểm thử tự động toàn diện
├── SPEC.md               # Tài liệu Đặc tả Kỹ thuật (SPEC-HUMAN-BEHAVIOR-01)
├── PLAN.md               # Kế hoạch Triển khai (PLAN-HUMAN-BEHAVIOR-01)
└── requirements.txt      # Danh sách thư viện phụ thuộc
```
