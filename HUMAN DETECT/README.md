# URBAN SAFETY AI-SOC: HUMAN DETECTION & SKELETON INDICATOR

Ứng dụng giám sát và phát hiện con người từ Camera trong thời gian thực, hiển thị bộ khung xương (Skeleton Pose Estimation & HUD Indicator), và tự động trích xuất video clip bằng chứng chuẩn SOC.

Nâng cấp toàn diện từ dự án tham chiếu [camera-recorder-with-human-detection](https://github.com/g4lb/camera-recorder-with-human-detection).

---

## 1. CÁC TÍNH NĂNG NỔI BẬT

1. **Phát hiện Con người & Vẽ Khung Xương Thời gian thực (Real-time Skeleton Indicator):**
   - Định vị chính xác 33 khớp xương giải phẫu học (vai, khuỷu tay, cổ tay, hông, đầu gối, cổ chân).
   - Vẽ đường nối xương (Bones) phân màu trực quan cho từng chi, đi kèm hộp bao (Bounding Box) và độ tin cậy (`Confidence`).
   - Tối ưu hóa cực đại cho CPU thông thường, đạt 25–30 FPS mượt mà trên Windows.

2. **Bảng điều khiển HUD & Đồng hồ Thời gian thực (Visible Clock Overlay):**
   - Hiển thị trực tiếp Timestamp (`YYYY-MM-DD HH:MM:SS.mmm`) trên từng khung hình phục vụ đối soát, kiểm toán sự cố chuẩn SOC.
   - Thống kê FPS thời gian thực và thời gian suy luận AI (`Inference Latency`).

3. **Tự động Trích xuất Video Bằng chứng (Automated Evidence Rolling Buffer):**
   - Duy trì bộ đệm vòng (Rolling Buffer 1.5s trước sự kiện + 2.0s sau sự kiện).
   - Tự động lưu video clip MP4 khi phát hiện người vào thư mục `output/video_{timestamp}_human.mp4`.
   - **Độc lập 100%:** Sử dụng OpenCV native VideoWriter, không bắt buộc cài đặt công cụ `ffmpeg` ngoại vi vào Windows PATH.

4. **Hỗ trợ 2 Chế độ Hoạt động:**
   - **Mode 1 (Live Real-time):** Giám sát camera trực tiếp.
   - **Mode 2 (Batch Processing):** Quét toàn bộ video lưu trữ trong thư mục `records/` và trích xuất các phân đoạn có người (tương thích 100% repo gốc).

---

## 2. CÀI ĐẶT

Môi trường yêu cầu: **Python 3.10+ (đã xác thực tương thích Python 3.15 trên Windows)**.

```bash
cd "HUMAN DETECT"
pip install -r requirements.txt
```

---

## 3. HƯỚNG DẪN SỬ DỤNG

### 3.1. Chế độ Giám sát Trực tiếp từ Camera (Live Mode)

Khởi chạy webcam mặc định (Camera 0):
```bash
python main.py --mode live
```

Chỉ định camera khác (ví dụ Camera 1):
```bash
python main.py --mode live --camera 1
```

Ghi toàn bộ phiên làm việc liên tục vào thư mục `records/` đồng thời tự động cắt clip bằng chứng khi có người:
```bash
python main.py --mode live --record-all
```

#### Phím tắt điều khiển khi xem Camera:
| Phím tắt | Chức năng |
| :---: | :--- |
| **`q`** | Thoát ứng dụng và đóng video clip an toàn. |
| **`s`** | Chụp ảnh Snapshot chất lượng cao kèm khung xương lưu vào `output/snapshot_{timestamp}.jpg`. |
| **`h`** | Bật / Tắt lớp hiển thị Khung xương & HUD. |
| **`r`** | Bật / Tắt chế độ ghi hình liên tục toàn bộ phiên. |

---

### 3.2. Chế độ Quét Hàng loạt Video Ngoại tuyến (Batch Mode)

Sao chép các file video cần phân tích (`.mp4`, `.avi`, `.mov`, `.mkv`) vào thư mục `records/`, sau đó chạy:
```bash
python main.py --mode batch
```
Ứng dụng sẽ tự động phân tích từng video, vẽ khung xương và lưu các đoạn có người ra thư mục `output/`.

---

### 3.3. Chạy Bộ Kiểm thử Tự động (Automated Test Suite)

Để kiểm tra độ tương thích của hệ thống, thuật toán và tốc độ FPS trên máy tính của bạn:
```bash
python test_system.py
```

---

## 4. CẤU TRÚC THƯ MỤC

```
HUMAN DETECT/
├── records/              # Chứa video camera quay được hoặc video đầu vào
├── output/               # Chứa video clip bằng chứng (.mp4) và nhật ký sự kiện (.jsonl)
├── models.py             # Schema cấu trúc dữ liệu chuẩn SOC
├── pose_detector.py      # Bộ nhận diện MediaPipe Pose & Vẽ khung xương
├── clip_recorder.py      # Quản lý Rolling Buffer & Trích xuất video clip
├── camera_recorder.py    # Ứng dụng Camera Streamer thời gian thực
├── human_detection.py    # Bộ xử lý batch video ngoại tuyến
├── main.py               # Entrypoint điều khiển CLI
├── test_system.py        # Kịch bản kiểm thử tự động
├── SPEC.md               # Tài liệu Đặc tả Kỹ thuật
├── PLAN.md               # Kế hoạch Triển khai
└── requirements.txt      # Danh sách thư viện phụ thuộc
```

