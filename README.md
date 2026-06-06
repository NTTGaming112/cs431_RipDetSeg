# CS431 - Rip Current Detection & Segmentation

Đây là repository lưu trữ mã nguồn và tài liệu cho đồ án môn học CS431.

## Tóm tắt nội dung

Dòng chảy rút (Rip currents) là hiện tượng vận động nước phức tạp và là nguyên nhân hàng đầu gây tai nạn đuối nước tại các bãi biển trên thế giới. Việc phát hiện chúng bằng quan sát thủ công gặp nhiều khó khăn do đặc tính vô hình và biến đổi bất thường.

Báo cáo này trình bày hệ thống phát hiện và phân đoạn dòng chảy rút dựa trên các mô hình học sâu tiên tiến, đánh giá ba kiến trúc:
1. **YOLOv8n-seg**: Mô hình cơ sở (Baseline).
2. **YOLOv8n-seg + CBAM**: Tích hợp module chú ý CBAM (Convolutional Block Attention Module).
3. **YOLO26n-seg + CBAM**: Kiến trúc end-to-end, NMS-free.

Phương pháp đánh giá sử dụng Stratified K-Fold Cross Validation (5 folds) trên tập dữ liệu Irikos/RipVIS (22,738 ảnh). Kết quả trên tập test (4,349 ảnh) cho thấy **YOLO26+CBAM** đạt hiệu suất tốt nhất với Score 0.48 – cải thiện 14.3% so với YOLOv8 cơ sở (0.42), đồng thời giảm 16.8% số tham số (2.7M vs 3.2M) và tăng tốc suy luận 24% (11.56ms vs 15.22ms).


## Cấu trúc thư mục

- `src/notebooks/`: Các file thực nghiệm, huấn luyện và kiểm thử mô hình.
- `src/docs/`: Báo cáo (report) và slide trình bày.