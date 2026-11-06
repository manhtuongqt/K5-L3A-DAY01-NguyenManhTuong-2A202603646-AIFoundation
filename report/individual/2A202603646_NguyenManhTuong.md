# Báo cáo cá nhân — Day 01: Lọc SMS rác

- **MSSV:** 2A202603646
- **Họ tên:** Nguyen Manh Tuong
- **SEED:** 885 (thử thách #4)
- **Dữ liệu:** UCI SMS Spam Collection, 5.574 dòng gốc; bỏ 403 bản ghi trùng còn 5.171. Train/validation/test = 3.621/775/775; SUB = 40 tin cân bằng. Mã bài làm dùng NumPy, PyTorch và MiniLM; không import `sklearn`.

## 1. Bằng chứng

| Tệp | Trạng thái |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | Đã chạy hết, còn output, ghi dự đoán trước thí nghiệm và trả lời câu hỏi; 15/15 kiểm tra đạt. |
| [`lab1_report.json`](lab1_report.json) | Sinh từ ô cuối; ghi seed, log dự đoán, kiểm tra và kết quả thực đo. |
| [`hand_calc.jpg`](hand_calc.jpg) | Chưa có; học viên sẽ tự viết tay và bổ sung ảnh. [`hand_calc_steps.md`](hand_calc_steps.md) chỉ để đối chiếu phép tính. |

Phần 4 dùng `CKEY_API_KEY` trong `.env` bị Git bỏ qua hoặc Colab Secrets. API `https://api.xah.io/v1`, model lớn `gemini-3.8-flash`, model nhỏ `gemini-2.5-flash-lite`. Thời gian từng yêu cầu đã đo; cache ngoài repository chỉ để tiếp tục sau lỗi mạng và giữ thời gian đo ban đầu. Giá USD là **ước tính** từ số từ/0,75 token và giá CKEY niêm yết lúc chạy, không phải hóa đơn.

## 2. Test thường và SUB

| Phương án | F1 test 775 | F1 SUB 40 | p95 ms/tin | Tin/s cả lô | USD/1.000 tin ước tính |
| --- | ---: | ---: | ---: | ---: | ---: |
| BoW + LogReg tự code | 0.906 | 0.974 | 0.461 | 8.993 | 0 |
| BoW + MLP | 0.948 | 1.000 | 1.697 | 17.341 | 0 |
| MiniLM + kNN | 0.921 | 1.000 | 75.614 | 17.2 | 0 |
| MiniLM + LogReg | **0.959** | 1.000 | 38.031 | 78.3 | 0 |
| MiniLM + MLP head | 0.950 | 1.000 | 34.960 | 66.3 | 0 |
| CKEY Flash-Lite zero-shot | — | 0.976 | 7308 | 0.36 | 0.0084 |
| CKEY 3.8 Flash zero-shot | — | 1.000 | 54119 | 0.07 | 0.00123 |
| CKEY Flash-Lite few-shot | — | 1.000 | 5998 | 0.34 | 0.0281 |

Ba hàng LLM chỉ đo SUB 40; p95 ở bảng là **ms** (tương ứng 7,308; 54,119; 5,998 giây). 3.8 Flash có **một output rỗng/không hợp lệ** được quy về ham và tình cờ đúng nhãn; F1 1.000 không chứng minh API ổn định. Few-shot tăng F1 Flash-Lite từ 0.976 lên 1.000, nhưng chi phí ước tính tăng khoảng 3,35 lần. Thông lượng là gọi tuần tự, không phải benchmark tải đồng thời; độ trễ máy cục bộ thay đổi theo tải.

## 3. Dự đoán sai

1. **Learning rate:** Tôi dự đoán `lr=50` sẽ phân kỳ. Loss vọt lên đầu nhưng sau 300 epoch còn khoảng **0.003**; `lr=0.001` chậm hơn, cuối khoảng **0.625**.
2. **Bỏ ReLU:** Tôi dự đoán hai Linear kém MLP có ReLU. Thực tế F1 không ReLU **0.953**, nhỉnh hơn MLP **0.948**. Một seed và chênh 0.005 chưa đủ kết luận kiến trúc nào tốt hơn.
3. **Pareto:** Tôi dự đoán MiniLM + LogReg nằm trên đường Pareto. Trên SUB 40, BoW + MLP có cùng F1 **1.000** nhưng p95 **1.697 ms** thay vì **38.031 ms**, nên lấn át nó. Đường Pareto gồm BoW + LogReg và BoW + MLP. Trên toàn test, MiniLM + LogReg vẫn có F1 cao hơn (0.959 so với 0.948).

Tìm kiếm embedding trả 6/6 hàng xóm spam, nhưng 12/13 token truy vấn đã có trong vocab BoW; truy vấn này chưa chứng minh ưu thế khi không trùng từ. Một tin sai trên SUB này làm F1 BoW + LogReg đổi từ **0.974 lên 1.000** (2,56 điểm phần trăm), nên không thể xếp hạng chắc chắn bằng khoảng cách nhỏ trên 40 tin.

## 4. Stress test 50 ca khó

| Phương án | F1 SUB 40 | F1 50 ca khó | Accuracy 50 ca |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.974 | 0.294 | 0.52 |
| BoW + MLP | 1.000 | 0.450 | 0.56 |
| MiniLM + kNN | 1.000 | 0.737 | 0.70 |
| MiniLM + LogReg | 1.000 | 0.612 | 0.62 |
| MiniLM + MLP head | 1.000 | 0.583 | 0.60 |
| Flash-Lite zero-shot | 0.976 | **0.885** | 0.88 |
| Flash-Lite + prompt phòng thủ | 0.976* | **0.966** | **0.96** |

\* F1 SUB ở hàng phòng thủ là mốc zero-shot gốc; prompt phòng thủ chỉ đo trên 50 ca khó.

BoW + LogReg tụt **0.680 F1** từ SUB sang bộ khó. Với spam giọng tự nhiên, BoW + LogReg đúng **0/10**, MiniLM + MLP head **1/10**, LLM zero-shot **4/10**. Prompt phòng thủ nâng LLM lên **9/10** nhóm này, nhưng nhóm ham trông giống spam giảm từ **10/10 xuống 9/10**. Với prompt injection, cả hai prompt đúng **6/6**; không thấy model bị lừa ở sáu ca này, nên không thể quy cải thiện nhóm đó cho phòng thủ. Có **1/50** ca cả bảy phương án đều sai: #13 là tin bán ảnh viết như lời nhắc thân thiện. Ví dụ #21 giả làm mẹ có điện thoại mới để xin chuyển tiền cho thấy ý định lừa đảo có thể bị che bởi hội thoại.

Để cứu BoW trước kiểu né từ, có thể chuẩn hoá Unicode và ký tự đồng dạng/số trước vector hoá; phải đo false positive trên ham thật vì quy tắc có thể biến tên riêng hoặc mã hợp lệ thành từ khoá. Bộ 50 ca được soạn để tìm lỗi và kiểm tra hồi quy; không đại diện phân phối tin nhắn thực và không dùng một mình để ước lượng chất lượng sản xuất.

## 5. Thử thách #4: thay k của kNN

Giữ MiniLM, train/test và các tham số khác; chỉ thay `k`:

| k | Precision | Recall | F1 test |
| ---: | ---: | ---: | ---: |
| 1 | 0.897 | 0.946 | 0.921 |
| 5 | 0.897 | 0.946 | 0.921 |
| 25 | 0.846 | 0.937 | 0.889 |

`k=1` có thể nhạy với điểm nhiễu, nhưng ở lần chạy này bằng `k=5`. `k=25` lấy thêm hàng xóm xa và ham là lớp đa số, khiến precision/recall giảm. Mã kNN không dùng `sklearn`.

## 6. Phương án cho 10 triệu SMS/ngày

10 triệu/ngày tương đương khoảng **116 tin/giây trung bình**. Với SLA p95 < 50 ms/tin, hai BoW và hai head MiniLM đạt ngưỡng trong phép đo từng tin; kNN p95 **75.614 ms** không đạt. Tôi chọn **MiniLM + LogReg** nếu ưu tiên F1 toàn test **0.959**, p95 **38.031 ms**. Thông lượng một môi trường thử **78 tin/s** chưa đạt trung bình và tải đỉnh sẽ cao hơn; cần nhiều replica và benchmark đồng thời. **BoW + MLP** (F1 **0.948**, p95 **1.697 ms**) xử lý tin chắc chắn ở tầng đầu, chuyển ca lưỡng lự/nghi né từ sang MiniLM + LogReg. Tổng p95 riêng lẻ khoảng **39.728 ms** trước overhead; không thể cộng hai p95 để suy ra p95 end-to-end, nên phải đo cả router.

Ba cấu hình LLM đều vượt SLA trực tiếp. Có thể dùng LLM ở luồng bất đồng bộ cho ca cần xem xét thêm, kèm kiểm tra độc lập và theo dõi lỗi API. Ngoại suy chi phí CKEY ước tính ở bảng cho 10 triệu tin/ngày × 30 ngày: Flash-Lite zero-shot khoảng **$2.520/tháng**, 3.8 Flash zero-shot **$369/tháng**, Flash-Lite few-shot **$8.434/tháng**. Giá 3.8 Flash của CKEY được niêm yết thấp hơn Flash-Lite lúc thử; số này chưa tính retry, overhead, thay đổi giá và không phải hóa đơn vận hành.

## 7. AI log và tự kiểm chứng

| Phần | Yêu cầu với AI | AI hỗ trợ | Kiểm chứng / giới hạn |
| --- | --- | --- | --- |
| 0–3, thử thách | Hoàn thiện lab, không dùng `sklearn` trong mã bài làm | Viết TODO, dự đoán, giải thích và thử kNN | Chạy notebook 15/15 kiểm tra; đối chiếu output, hình và JSON. Học viên cần tự đọc để giải thích được mã. |
| 4 | Kết nối CKEY bằng key trong `.env` bị Git bỏ qua | Cấu hình provider, wrapper, retry/cache và parser | Chạy 40 tin cho ba cấu hình, 50 ca cho hai prompt; đối chiếu output. Một output 3.8 Flash không hợp lệ; chi phí chỉ ước tính. |
| 5–6 | Tổng hợp báo cáo | Viết bảng và phân tích từ kết quả | Đối chiếu F1, độ trễ với JSON và bộ khó; ảnh viết tay chưa được cung cấp. |

Tôi thấy F1 của lớp spam quan trọng hơn accuracy trên dữ liệu lệch lớp. Embedding cải thiện nhiều ca né từ nhưng vẫn bỏ sót spam viết như hội thoại. Prompt phòng thủ giúp một số ca khó nhưng có thể tăng false positive. Cần xem p95 từng tin, throughput, lỗi API và độ khó bộ test cùng nhau trước quyết định triển khai.

**Việc còn thiếu để nộp đủ:** học viên thêm ảnh `report/individual/hand_calc.jpg` tự viết tay. Tên repository hiện có `L3A`, khác mẫu `L3` trong `SUBMISSION.md`; cần đổi tên nếu giáo viên yêu cầu khớp tuyệt đối.
