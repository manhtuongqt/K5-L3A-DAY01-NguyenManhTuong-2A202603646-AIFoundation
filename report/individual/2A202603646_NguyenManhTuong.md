# Báo cáo cá nhân — Day 01: Lọc SMS rác

- **MSSV:** 2A202603646
- **Họ và tên:** Nguyen Manh Tuong
- **SEED:** 885 (thử thách #4)
- **Nguồn dữ liệu:** UCI SMS Spam Collection; 5.574 dòng gốc, bỏ 403 bản ghi trùng còn 5.171 dòng. Chia train/validation/test lần lượt 3.621/775/775, SUB 40 tin. Thí nghiệm chạy với NumPy, PyTorch và MiniLM; mã bài làm không gọi `sklearn`.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | ✅ Đã chạy, có output và trả lời câu hỏi; 15/15 kiểm tra đạt. |
| [`lab1_report.json`](lab1_report.json) | ✅ Sinh bởi ô báo cáo cuối; ghi seed, dự đoán, kiểm tra và kết quả thực đo. |
| [`hand_calc.jpg`](hand_calc.jpg) | ⏳ Chưa có; học viên sẽ viết tay và thêm ảnh sau. [`hand_calc_steps.md`](hand_calc_steps.md) lưu phép tính để đối chiếu. |

## 2. Kết quả chính

| Phương án | F1 test 775 | F1 SUB 40 | p95 ms/tin | Tin/s cả lô | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.906 | 0.974 | 0.051 | 48796 | 0.00 |
| 2. BoW + MLP (deep learning) | 0.948 | 1.000 | 0.470 | 38910 | 0.00 |
| 3a. Encoder (MiniLM) + kNN | 0.921 | 1.000 | 35.645 | 43 | 0.00 |
| 3b. Encoder (MiniLM) + LogReg | 0.959 | 1.000 | 30.051 | 109 | 0.00 |
| 3c. Encoder (MiniLM) + MLP head | 0.950 | 1.000 | 20.450 | 104 | 0.00 |

Độ trễ đo từ văn bản thô đến nhãn trên môi trường thử; thông lượng cả lô không đồng nghĩa khả năng nhận request đồng thời. Bảng chưa có hàng LLM vì Gemini trả 429 khi chạy batch; không có F1, độ trễ hay USD/1k thực đo để điền. Một yêu cầu chào ngắn qua Gemini 2.5 Flash-Lite thành công, nhưng điều đó chỉ chứng minh key và đường gọi hoạt động. Dự đoán đều được ghi trước thí nghiệm trong JSON.

## 3. Ba dự đoán lệch thực tế

1. **Learning rate:** Tôi dự đoán `lr=50` sẽ dao động/phân kỳ. Loss vọt lên lúc đầu nhưng cuối 300 epoch còn khoảng **0.003**; gradient và dữ liệu này vẫn cho phép hội tụ sau overshoot. `lr=0.001` chậm hơn rõ, cuối khoảng **0.625**.
2. **Bỏ ReLU:** Tôi dự đoán mạng hai Linear sẽ gần LogReg và kém MLP có ReLU. Thực tế F1 không ReLU **0.953**, cao hơn MLP **0.948** một chút. Một seed và chênh **0.005** chưa đủ chứng minh kiến trúc nào tốt hơn; tác động khởi tạo và tối ưu có thể lớn hơn.
3. **Pareto:** Tôi dự đoán Encoder + LogReg nằm trên đường Pareto. Trên **SUB 40**, nó F1 **1.000** như BoW + MLP nhưng p95 **30.051 ms** so với **0.470 ms**, nên bị lấn át. F1 toàn test của Encoder + LogReg vẫn cao hơn (0.959 so với 0.948), cho thấy Pareto SUB nhạy với mẫu nhỏ.

Thí nghiệm tìm kiếm embedding trả 6/6 hàng xóm spam, nhưng 12/13 token truy vấn đã có trong vocab BoW. Vì thế không thể dùng truy vấn này một mình để chứng minh lợi thế khi không trùng từ.

## 4. Phân tích 50 case khó

| Phương án | F1 SUB 40 | F1 50 case khó | Accuracy 50 case |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.974 | 0.294 | 0.52 |
| BoW + MLP | 1.000 | 0.450 | 0.56 |
| MiniLM + kNN | 1.000 | 0.737 | 0.70 |
| MiniLM + LogReg | 1.000 | 0.612 | 0.62 |
| MiniLM + MLP head | 1.000 | 0.583 | 0.60 |

BoW + LogReg tụt nhiều nhất (**0.680 F1** so với SUB). Ở nhóm **spam giọng tự nhiên**, BoW + LogReg đúng **0%** và MiniLM + MLP head chỉ **10%**. Ví dụ #14 kể chuyện làm ở nhà kiếm khoảng £300/ngày: cả năm mô hình đều bỏ sót vì văn phong giống lời kể cá nhân, ý định chào mời việc làm nằm ở ngữ cảnh. Ví dụ #21 giả làm mẹ có điện thoại mới rồi xin chuyển tiền: MiniLM + MLP head đoán ham vì vỏ bọc hội thoại thân mật che hành vi lừa đảo. Nhóm né từ khoá cho thấy BoW + LogReg chỉ đúng **17%**, còn MiniLM + MLP head **83%**. Không có kết quả LLM hoặc prompt phòng thủ: quota ngăn chạy 4a/4d. Bộ 50 ca cố ý khó giúp tìm lỗi và kiểm tra hồi quy, không đại diện cho lưu lượng SMS thật.

## 5. Thử thách cá nhân #4: thay k của kNN

Giữ MiniLM, train/test và mọi siêu tham số khác; chỉ thay `k` trong `knn_predict`:

| k | Precision | Recall | F1 test |
| ---: | ---: | ---: | ---: |
| 1 | 0.897 | 0.946 | 0.921 |
| 5 | 0.897 | 0.946 | 0.921 |
| 25 | 0.846 | 0.937 | 0.889 |

`k=1` có thể nhạy với điểm nhiễu, nhưng ở lần chạy này bằng `k=5`. `k=25` lấy thêm hàng xóm xa và ham là lớp đa số, làm precision/recall giảm. Không có `sklearn` trong mã kNN bài làm.

## 6. Phương án sản phẩm thật

Yêu cầu 10 triệu SMS/ngày tương đương **116 tin/giây trung bình**; tất cả năm phương án cục bộ đạt p95 < 50 ms trong phép đo từng tin. Chọn **MiniLM + LogReg** khi ưu tiên F1 toàn test cao nhất **0.959** với p95 **30.051 ms**. Thông lượng cả lô **109 tin/s** trên một môi trường thử chưa đủ 116 tin/s và lưu lượng đỉnh có thể cao hơn nhiều; cần nhiều replica, benchmark tải đồng thời và theo dõi drift trước triển khai. **BoW + MLP** (F1 **0.948**, p95 **0.470 ms**) phù hợp tầng lọc đầu; chuyển tin xác suất lưỡng lự hoặc có dấu hiệu né từ sang MiniLM + LogReg. Độ trễ hai tầng ước khoảng 30.52 ms trước overhead, cần đo end-to-end.

Chi phí LLM chưa đo được. Theo [giá Gemini chính thức](https://ai.google.dev/gemini-api/docs/pricing), nếu **giả sử** mỗi tin 100 input + 1 output token, 10 triệu tin/ngày, 30 ngày và giá paid standard: Flash-Lite khoảng **$3,120/tháng**, Flash khoảng **$9,750/tháng**. Đây là ước tính lý thuyết; prompt thật và thinking token có thể làm chi phí khác. Độ trễ và quota của LLM chưa cho phép đánh giá SLA.

## 7. AI log và điều đã học

| Phần | Yêu cầu với AI | AI hỗ trợ | Kiểm chứng / giới hạn |
| --- | --- | --- | --- |
| 0–3, thử thách | Hoàn thiện lab, không dùng `sklearn` trong mã bài làm | Viết hàm TODO, dự đoán, giải thích và thử kNN | Chạy notebook 15/15 kiểm tra; đối chiếu output, hình và JSON. Phần này do AI soạn; học viên cần tự đọc để giải thích được. |
| 4 | Thử Gemini bằng key trong `.env` bị Git bỏ qua | Cấu hình provider, wrapper và parser | Lời chào ngắn thành công; batch gặp 429/503, nên không có chỉ số LLM. |
| 5–6 | Tổng hợp báo cáo | Viết bảng và phân tích từ kết quả | Đối chiếu F1 và độ trễ với JSON; ảnh viết tay chưa được cung cấp. |

Tôi thấy F1 của lớp spam quan trọng hơn accuracy trên dữ liệu lệch lớp; SUB 40 dễ khiến chênh lệch vài điểm phần trăm trông chắc chắn quá mức. Embedding cải thiện nhiều ca né từ khoá nhưng vẫn bỏ sót spam viết như hội thoại. Kiểm tra p95 từng tin, throughput và độ khó của bộ test cần được xem cùng nhau trước quyết định triển khai.

**Việc còn thiếu để nộp đủ:** thêm ảnh `report/individual/hand_calc.jpg` do học viên tự viết; khi API Gemini hết giới hạn, chạy lại các ô 4.A/4.B và 4a/4d rồi cập nhật báo cáo. Tên repository hiện tại của học viên có `L3A`, khác mẫu `L3` trong `SUBMISSION.md`; nên đổi tên repository trước khi nộp LMS nếu giáo viên yêu cầu khớp tuyệt đối.
