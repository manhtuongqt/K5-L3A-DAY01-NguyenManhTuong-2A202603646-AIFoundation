# Bài nộp Day 01 của tôi

Tôi là Nguyen Manh Tuong, mã sinh viên 2A202603646, lớp L3A. Thư mục này là phần bài nộp cá nhân của tôi cho lab lọc SMS rác.

## Các tệp cần xem

- [`day01_lab.ipynb`](day01_lab.ipynb): notebook đã chạy, có output của từng thí nghiệm, dự đoán trước khi chạy và câu trả lời sau khi xem kết quả.
- [`lab1_report.json`](lab1_report.json): tệp do ô báo cáo cuối trong notebook tạo ra; có seed 885, kết quả kiểm tra và số đo của các mô hình.
- [`2A202603646_NguyenManhTuong.md`](2A202603646_NguyenManhTuong.md): báo cáo của tôi, gồm bảng kết quả, phân tích lỗi và AI log.
- `hand_calc.jpg`: ảnh tôi viết tay năm phép tính, đã gửi trong cuộc trò chuyện nhưng chưa lưu thành tệp trong repository. Tôi ghi thiếu hệ số 2 trong dòng công thức F1 trên giấy; kết quả 0,73 là kết quả theo công thức đầy đủ. [`hand_calc_steps.md`](hand_calc_steps.md) ghi rõ phép tính để đối chiếu.

## Nếu muốn chạy lại

Mở `day01_lab.ipynb` trên Colab và chạy các ô từ đầu đến cuối. Notebook cài các thư viện cần thiết ở đầu, dùng seed 885 và không import `sklearn` trong mã bài làm. Phần LLM cần thêm `CKEY_API_KEY` vào Colab Secrets và bật quyền truy cập cho notebook. Trên máy cá nhân, có thể đặt key trong file `.env` ở gốc repository rồi nạp thành biến môi trường trước khi chạy. Tôi không đưa key vào notebook hoặc Git.

Các lời gọi LLM có thể tốn tiền và mất thời gian. Nếu chỉ cần xem bài nộp, có thể đọc output đã lưu trong notebook và bảng trong báo cáo mà không cần chạy API. Số tiền trong báo cáo là ước tính theo bảng giá CKEY lúc thí nghiệm, không phải hóa đơn.

Tôi dùng tên repository có `L3A` theo hướng dẫn chia lớp của buổi học.
