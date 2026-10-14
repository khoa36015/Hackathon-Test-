# AGENTS.md — QUY TẮC ĐIỀU PHỐI AI CODING AGENT
**Dự án:** Hệ thống Giám sát An ninh & An toàn Đô thị Thông minh (Urban Safety AI-SOC)  
**Phương pháp luận:** Spec-Driven Development (SDD) & Strict Human-in-the-Loop (HITL)  
**Cấp độ thực thi:** BẮT BUỘC TUYỆT ĐỐI (ZERO-TOLERANCE ENFORCEMENT)

---

## 1. NGUYÊN TẮC BẤT KHẢ XÂM PHẠM (CORE DIRECTIVES)

Bạn là một **AI Coding Agent** làm việc dưới sự chỉ đạo trực tiếp của **Principal Architect / Lead Developer (Con người)**. Trong suốt vòng đời dự án Hackathon này, bạn phải tuân thủ tuyệt đối 5 nguyên tắc tối cao sau:

1. **[CD-01] Spec-First & Plan-First (Tuyệt đối không sinh code sớm):**
   - Nghiêm cấm viết bất kỳ dòng code thực thi nào (kể cả code mẫu, script thử nghiệm hay boilerplate) khi **Tài liệu Đặc tả Kỹ thuật (`SPEC.md`)** và **Kế hoạch Triển khai (`PLAN.md`)** của tính năng đó chưa được tạo, rà soát và nhận lệnh phê duyệt tường minh từ Con người.
2. **[CD-02] Strict Architectural Lock (Khóa kiến trúc nghiêm ngặt):**
   - Con người nắm quyền kiểm soát độc quyền đối với kiến trúc hệ thống, luồng dữ liệu (data pipeline), cấu trúc cơ sở dữ liệu, API contract và cơ chế chuyển đổi trạng thái (state machine).
   - Bạn chỉ đóng vai trò thực thi chính xác theo thiết kế đã duyệt, không được tự ý "sáng tạo" lại luồng nghiệp vụ.
3. **[CD-03] No Unauthorized Mutation (Cấm tự ý thay đổi hệ thống):**
   - Nghiêm cấm tự ý thêm/bớt thư viện lõi (`dependencies`), thay đổi mô hình AI/VLM, đổi cấu trúc thư mục, sửa đổi schema dữ liệu hoặc thay đổi thuật toán chấm điểm rủi ro khi chưa có sự đồng ý bằng văn bản từ Con người.
   - Mọi sự khác biệt giữa thực tế triển khai và bản thiết kế đều phải dừng lại để đệ trình **RFC (Request for Comments)**.
4. **[CD-04] Dual Human-in-the-Loop Integrity (Bảo toàn chốt chặn Con người ở 2 tầng):**
   - **Tầng Phát triển (Development HITL):** Không tự ý chuyển giai đoạn nếu chưa có lệnh duyệt của Developer.
   - **Tầng Nghiệp vụ (Operational HITL):** Trong code nghiệp vụ, tuyệt đối không viết bất kỳ nhánh logic nào cho phép hệ thống tự động phát lệnh báo động tới cơ quan chức năng mà bỏ qua bước xác thực của Vận hành viên (Human Operator).
5. **[CD-05] SOC-Grade Reliability & Traceability (Độ tin cậy và Truy vết chuẩn SOC):**
   - Mọi luồng xử lý video, suy luận AI và đánh giá rủi ro đều phải có cơ chế xử lý lỗi tường minh, giới hạn thời gian chờ (timeout), và lưu vết kiểm toán (audit log). Nghiêm cấm sử dụng các khối bắt lỗi rỗng (`silent catch/pass`) làm che giấu sự cố rớt khung hình hoặc lỗi mô hình.

---

## 2. QUY TRÌNH LÀM VIỆC 4 BƯỚC (4-STAGE SDD WORKFLOW)

Mọi yêu cầu phát triển tính năng hoặc module mới đều phải đi qua tuần tự 4 giai đoạn dưới đây. Bạn **không được phép gộp bước** hoặc **nhảy cóc** trong bất kỳ hoàn cảnh nào, kể cả khi bị áp lực thời gian của Hackathon.

### BƯỚC 1: ĐẶC TẢ KỸ THUẬT (STAGE 1 — SPECIFICATION)
- **Nhiệm vụ của Agent:** Phân tích yêu cầu từ Con người, làm rõ các điểm mờ (edge cases) và soạn thảo/cập nhật tài liệu đặc tả kỹ thuật.
- **Đầu ra bắt buộc:**
  - Định nghĩa rõ Input / Output, Data Schema (JSON/Pydantic/TypeScript interfaces).
  - Sơ đồ luồng dữ liệu (Data Flow) và máy trạng thái (State Machine) dạng Mermaid hoặc văn bản.
  - Các ràng buộc về hiệu năng (độ trễ tối đa, tốc độ khung hình xử lý) và cách xử lý lỗi (fallback mechanism).
- **Chốt chặn (Gate 1):** Kết thúc phản hồi bằng nhãn `[WAITING FOR SPEC APPROVAL]` và dừng lại chờ Con người duyệt.

### BƯỚC 2: LẬP KẾ HOẠCH TRIỂN KHAI (STAGE 2 — IMPLEMENTATION PLAN)
- **Điều kiện kích hoạt:** Chỉ thực hiện sau khi Con người đã duyệt Bước 1.
- **Nhiệm vụ của Agent:** Chuyển hóa bản Spec đã duyệt thành danh sách công việc (atomic tasks) có thể kiểm chứng độc lập.
- **Đầu ra bắt buộc:**
  - Danh sách chính xác các file sẽ tạo mới, sửa đổi hoặc xóa bỏ.
  - Danh sách các hàm/class/interface sẽ triển khai cùng chữ ký hàm (function signatures).
  - Danh sách các thư viện cần dùng (nếu có) — phải giải trình lý do vì sao không dùng thư viện có sẵn.
  - Kịch bản kiểm thử (Verification & Test Plan) cho từng bước.
- **Chốt chặn (Gate 2):** Kết thúc phản hồi bằng nhãn `[WAITING FOR PLAN APPROVAL]` và dừng lại.

### BƯỚC 3: RÀ SOÁT & PHÊ DUYỆT (STAGE 3 — REVIEW & EXPLICIT APPROVAL)
- **Cơ chế hoạt động:**
  - Agent ở trạng thái **ĐÓNG BĂNG SINH CODE (Code Generation Freeze)**.
  - Con người sẽ rà soát kế hoạch, yêu cầu chỉnh sửa hoặc cấp quyền thực thi.
  - Agent chỉ được phép mở khóa Bước 4 khi nhận được từ khóa phê duyệt rõ ràng từ Con người (ví dụ: `"Approved"`, `"Duyệt"`, `"Proceed to code"`, `"Triển khai theo plan"`).

### BƯỚC 4: TRIỂN KHAI CODE & KIỂM CHỨNG (STAGE 4 — CODE & VERIFY)
- **Điều kiện kích hoạt:** Đã có lệnh duyệt ở Bước 3.
- **Nhiệm vụ của Agent:**
  - Viết code tuân thủ 100% thiết kế và kế hoạch đã thống nhất.
  - Không tự ý "tiện tay" refactor các file không nằm trong danh sách của Bước 2.
  - Nếu phát sinh rào cản kỹ thuật không lường trước khi đang viết code, **LẬP TỨC DỪNG LẠI** và kích hoạt Giao thức RFC (Mục 4).

---

## 3. RÀNG BUỘC KIẾN TRÚC HỆ THỐNG URBAN SAFETY SOC (DOMAIN & PIPELINE CONSTRAINTS)

Hệ thống sản phẩm là một nền tảng **An toàn Đô thị (Urban Safety)** hoạt động theo thời gian thực. Mọi thiết kế và dòng code của Agent phải tuân thủ nghiêm ngặt kiến trúc 4 tầng nghiệp vụ sau:

```
[Camera Streams] 
      │ (Frame Ingestion & Rolling Buffer)
      ▼
[Tầng 1: Input Layer] ──► [Tầng 2: AI/VLM Analytics] ──► [Tầng 3: Tracking & Dynamic Threat Scoring]
                                                                       │
                                                       (Threat >= Ngưỡng cảnh báo)
                                                                       ▼
[Cơ quan chức năng] ◄── (Đóng gói & Gửi) ◄── [Operator Xác nhận] ◄── [Tầng 4: Operational HITL & Dispatch]
```

### 3.1. Tầng Nguồn Dữ Liệu (Input & Ingestion Layer)
- **Cảm biến chính:** Hệ thống Camera giám sát thu thập luồng video thời gian thực (RTSP, WebRTC, hoặc luồng giả lập chuẩn hóa từ video file theo thời gian thực).
- **Ràng buộc bất đồng bộ (Non-blocking Ingestion):** Luồng đọc khung hình (Frame Grabber) phải tách biệt hoàn toàn với luồng suy luận AI (Inference Worker) thông qua hàng đợi hoặc bộ đệm vòng (Ring Buffer). Tuyệt đối không để quá trình gọi VLM làm nghẽn (block) luồng thu nhận video.
- **Bộ đệm Bằng chứng (Evidence Rolling Buffer):** Hệ thống phải duy trì bộ đệm lưu trữ tạm thời các khung hình trước và sau sự kiện (pre-event & post-event buffer) để sẵn sàng trích xuất video clip bằng chứng khi có yêu cầu.

### 3.2. Tầng Phân Tích Hành Vi (AI / VLM Analytics Layer)
- **Kiến trúc Kết hợp (Hybrid CV + VLM):**
  - Sử dụng Computer Vision tốc độ cao để phát hiện đối tượng (Object Detection), khoanh vùng (Bounding Box) và lọc sự kiện cơ bản trên luồng video.
  - Sử dụng Multimodal LLM / VLM để phân tích sâu ngữ cảnh thị giác trên các khung hình trọng yếu (Keyframes / Event Windows).
- **Trọng tâm Nghiệp vụ:**
  - **Dấu hiệu tiền phạm tội (Pre-crime Indicators):** Lảng vảng bất thường tại khu vực nhạy cảm (loitering), che giấu hoặc mang theo vật nghi là hung khí, theo dõi mục tiêu, tụ tập có dấu hiệu gây rối, xâm nhập trái phép vùng cấm.
  - **Hành vi hậu phạm tội (Post-crime Actions):** Bỏ chạy tốc độ cao khỏi hiện trường, ẩu đả/tấn công, cướp giật, phi tang vật chứng, nạn nhân nằm bất động.
- **Chuẩn hóa Đầu ra (Strict Structured Output):** Mọi phản hồi từ VLM bắt buộc phải tuân thủ JSON Schema nghiêm ngặt (bao gồm: `detected_behaviors`, `pre_crime_flags`, `post_crime_flags`, `confidence_score`, `reasoning_summary`). Cấm phân tích cú pháp từ văn bản tự do (free-text parsing) thiếu kiểm soát.

### 3.3. Tầng Định Danh & Đánh Giá Rủi Ro Động (Tracking & Dynamic Threat Scoring)
- **Gắn thẻ & Theo vết (Tagging & Tracking):**
  - Khi phát hiện dấu hiệu bất thường, đối tượng phải được cấp mã định danh theo vết duy nhất (`track_id` / `subject_tag`) và duy trì bám vết liên tục qua các khung hình.
- **Máy Trạng Thái Đe Dọa Động (Dynamic Threat State Machine):**
  - Hệ thống đánh giá và cập nhật liên tục nhãn mức độ đe dọa (`Threat Level`) theo thời gian thực qua 4 cấp độ chuẩn:
    - `Low` (Bình thường / Quan sát cơ bản)
    - `Medium` (Có dấu hiệu nghi vấn / Tiền phạm tội mức nhẹ — bắt đầu tăng tần suất lấy mẫu VLM)
    - `High` (Rủi ro cao / Dấu hiệu tiền phạm tội rõ rệt hoặc chuẩn bị xung đột — kích hoạt cảnh báo lên màn hình SOC)
    - `Critical` (Khẩn cấp / Đang xảy ra hành vi phạm tội hoặc hậu phạm tội nguy hiểm)
- **Ràng buộc Chuyển đổi Trạng thái:**
  - Việc nâng hoặc hạ cấp `Threat Level` phải dựa trên hàm chấm điểm rủi ro theo chuỗi thời gian (Temporal Threat Scoring), kết hợp lịch sử hành vi của `track_id` chứ không chỉ dựa vào một khung hình đơn lẻ (tránh hiện tượng nhấp nháy nhãn / state flapping).
  - Mọi lần thay đổi `Threat Level` phải được ghi vào lịch sử sự kiện (`threat_history`) kèm theo `timestamp`, điểm số và lý giải (`reasoning`) từ mô hình.

### 3.4. Tầng Xác Thực & Phản Ứng (Operational HITL & Dispatch Layer)
- **Chốt chặn Vận hành viên (Mandatory Operator Verification):**
  - Khi một đối tượng hoặc sự kiện đạt ngưỡng rủi ro theo quy định (mặc định: `High` hoặc `Critical`), hệ thống chuyển trạng thái sự cố sang `AWAITING_OPERATOR_VERIFICATION` và đẩy lên giao diện điều hành.
  - Vận hành viên (Human Operator) có 3 quyền quyết định: `CONFIRM_AND_DISPATCH` (Xác nhận & Điều phối), `DOWNGRADE_MONITOR` (Hạ mức cảnh báo & Tiếp tục theo dõi), hoặc `DISMISS_FALSE_POSITIVE` (Hủy cảnh báo giả).
  - **Cấm tuyệt đối:** Không một lệnh gọi API gửi cơ quan chức năng nào được phép thực thi nếu trạng thái sự cố chưa được chuyển thành `OPERATOR_CONFIRMED` kèm theo `operator_id` và `verified_at`.
- **Đóng gói Hồ sơ Sự cố Tự động (Automated Incident Packaging):**
  - Ngay sau khi Vận hành viên xác nhận (`OPERATOR_CONFIRMED`), hệ thống tự động đóng gói một **Incident Evidence Package** hoàn chỉnh gồm:
    1. **Định danh & Tọa độ:** `incident_id`, `camera_id`, tên khu vực, tọa độ GPS (`latitude`, `longitude`), thời điểm phát hiện.
    2. **Nhãn & Đánh giá Rủi ro:** Mức độ đe dọa hiện tại (`High`/`Critical`), lịch sử leo thang rủi ro (`Low -> Medium -> High -> Critical`), tóm tắt phân tích hành vi từ VLM.
    3. **Bằng chứng Hình ảnh (Visual Evidence):** Ảnh chụp đối tượng rõ nét nhất đã vẽ khung định danh (Annotated Snapshot) và đoạn video clip trích xuất từ Rolling Buffer (bao gồm ngữ cảnh trước và trong khi xảy ra sự kiện).
    4. **Thông tin Kiểm toán (Audit Metadata):** ID của Vận hành viên xác nhận, ghi chú điều hành (nếu có), thời gian xác nhận.
- **Phát lệnh Cảnh báo (Dispatch):** Gửi gói dữ liệu đã đóng gói tới hệ thống tiếp nhận của cơ quan chức năng thông qua giao thức đã định nghĩa, đảm bảo cơ chế chống gửi trùng lặp (idempotency) và lưu log trạng thái phản hồi.

---

## 4. GIAO THỨC GIAO TIẾP KHI XIN PHÉP THAY ĐỔI THIẾT KẾ (ARCHITECTURAL RFC PROTOCOL)

Khi đang ở bất kỳ giai đoạn nào (đặc biệt là trong quá trình viết code ở Bước 4), nếu bạn phát hiện thiết kế hiện tại gặp nút thắt kỹ thuật, xung đột thư viện, độ trễ vượt ngưỡng cho phép, hoặc cần thay đổi cấu trúc dữ liệu/kiến trúc, bạn phải thực hiện đúng quy trình sau:

### 4.1. Nguyên tắc Kích hoạt
1. **Dừng ngay lập tức** việc chỉnh sửa code liên quan đến thay đổi đó.
2. **Không tự ý áp dụng giải pháp thay thế** (workaround) nếu giải pháp đó làm thay đổi luồng dữ liệu, thêm thư viện mới hoặc sửa đổi API contract.
3. Gửi một bản **RFC (Request for Comments)** ngắn gọn cho Developer theo đúng mẫu chuẩn dưới đây.

### 4.2. Mẫu Đệ trình RFC Bắt buộc

```markdown
### 🛑 [RFC - YÊU CẦU THAY ĐỔI THIẾT KẾ / KIẾN TRÚC]

- **Module / Tầng bị ảnh hưởng:** [Tầng 1: Input | Tầng 2: AI/VLM | Tầng 3: Tracking & Scoring | Tầng 4: HITL & Dispatch]
- **Vấn đề phát sinh (Root Cause):** 
  [Mô tả ngắn gọn, chính xác nguyên nhân kỹ thuật khiến thiết kế ban đầu không thể thực thi hoặc gây rủi ro hiệu năng/an toàn]
- **Đề xuất thay đổi (Proposed Mutation):**
  - *Hiện trạng (As-Is):* [Thiết kế/Thư viện/Cơ chế hiện tại trong Spec/Plan]
  - *Đề xuất mới (To-Be):* [Thiết kế/Thư viện/Cơ chế đề xuất thay thế]
- **Đánh giá tác động (Impact Analysis):**
  - *Luồng dữ liệu & Độ trễ:* [Có làm tăng latency hoặc đổi schema không?]
  - *Ràng buộc HITL:* [Khẳng định thay đổi này KHÔNG làm ảnh hưởng tới chốt chặn xác nhận của Operator]
  - *Các file bị ảnh hưởng:* [Danh sách các file cần sửa đổi nếu RFC được duyệt]
- **Phương án dự phòng (Alternative):** [1 phương án khác nếu Developer không muốn thay đổi kiến trúc]

**[STATUS: PAUSED — CHỜ QUYẾT ĐỊNH TỪ ARCHITECT (APPROVE / REJECT / MODIFY)]**
```

### 4.3. Quy tắc Xử lý Phản hồi RFC
- Nếu Con người phản hồi **`APPROVE`**: Cập nhật lại `SPEC.md` và `PLAN.md` để đồng bộ tài liệu trước, sau đó mới tiến hành viết code.
- Nếu Con người phản hồi **`REJECT`**: Tuyệt đối tuân thủ thiết kế ban đầu hoặc thực hiện đúng theo chỉ dẫn kỹ thuật mà Con người đưa ra.