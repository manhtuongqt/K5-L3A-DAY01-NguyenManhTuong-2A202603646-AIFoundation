# Lab 1 — Phân loại SMS spam

**Nguyen Manh Tuong · 2A202603646 · lớp L3A · seed 885**

Trong bài lab này, tôi dùng cùng một bộ tin nhắn để so sánh bốn hướng phân loại spam: Bag of Words với logistic regression, mạng MLP, embedding MiniLM với các bộ phân loại nhỏ, và LLM qua API. Tôi quan tâm cả F1 lẫn thời gian xử lý vì một mô hình có điểm số cao nhưng phản hồi chậm vẫn khó dùng để lọc tin nhắn theo thời gian thực.

Repository dùng `L3A` theo hướng dẫn chia lớp của buổi học. Mã các phần tôi thực hiện không import `sklearn`.

## Bài nộp

| Tệp | Nội dung |
| --- | --- |
| [Notebook đã chạy](report/individual/day01_lab.ipynb) | Mã, dự đoán trước thí nghiệm, output, biểu đồ và câu trả lời của từng phần. |
| [Báo cáo cá nhân](report/individual/2A202603646_NguyenManhTuong.md) | Bảng kết quả đầy đủ, phân tích lỗi, phương án triển khai và AI log. |
| [Kết quả dạng JSON](report/individual/lab1_report.json) | Seed, 10 dự đoán, 15 kiểm tra và chỉ số của các mô hình. |
| [Ảnh tính tay](report/individual/hand_calc.jpg) | Năm phép tính viết trên giấy. |
| [Đối chiếu phép tính](report/individual/hand_calc_steps.md) | Công thức đầy đủ và các giá trị làm tròn trong ảnh. |

Notebook gốc của lab nằm ở [`src/day01_spam_ml_dl_embedding_llm.ipynb`](src/day01_spam_ml_dl_embedding_llm.ipynb). Bản đã làm và có output là notebook trong `report/individual/`.

## Dữ liệu và cách đánh giá

Tôi dùng bộ [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection). Từ 5.574 dòng gốc, tôi bỏ 403 bản ghi trùng và còn 5.171 tin, trong đó 4.518 ham. Tôi chia thành 3.621 tin train, 775 tin validation và 775 tin test. Để so sánh với LLM mà không gọi API cho toàn bộ test, tôi lấy thêm SUB gồm 40 tin cân bằng. Seed 885 được tạo từ mã sinh viên theo cách của lab.

Tôi dùng precision, recall và F1 cho lớp spam thay vì chỉ nhìn accuracy. Nếu luôn đoán ham, accuracy vẫn khoảng 87% do ham chiếm đa số, nhưng mô hình sẽ bỏ sót toàn bộ spam. Bộ test thường dùng để đo chất lượng chung; bộ 50 ca khó được dùng riêng để tìm lỗi như spam viết tự nhiên, né từ khoá và prompt injection. Tôi không xem bộ 50 ca này là đại diện cho tin nhắn ngoài thực tế.

## Các thí nghiệm đã làm

### 1. Bag of Words và logistic regression

Tôi biến mỗi tin thành vector từ khoá, rồi huấn luyện logistic regression bằng NumPy. Mô hình tính một điểm từ vector đầu vào, đưa qua sigmoid để ra xác suất spam và so với ngưỡng 0,5. Tôi thử learning rate khác nhau để quan sát loss thay đổi. Kết quả F1 trên test 775 tin là **0,906**. Mô hình chạy nhanh, nhưng dễ bỏ sót spam khi cách viết không có những từ nó đã học.

### 2. Mạng MLP

Tôi dùng cùng biểu diễn Bag of Words nhưng thêm tầng ẩn và ReLU, huấn luyện bằng PyTorch. F1 test đạt **0,948**. Tôi cũng bỏ ReLU để kiểm tra vai trò của tính phi tuyến: ở seed này F1 của mạng không ReLU là **0,953**, nhỉnh hơn một chút. Chênh lệch nhỏ trên một lần chạy chưa đủ để kết luận mạng không ReLU tốt hơn.

### 3. Embedding MiniLM

Tôi dùng `all-MiniLM-L6-v2` để biểu diễn tin nhắn thành vector ngữ nghĩa và giữ encoder cố định. Trên vector này, tôi thử kNN, logistic regression và MLP head. F1 test lần lượt là **0,921**, **0,959** và **0,950**. Logistic regression trên embedding có F1 cao nhất của nhóm mô hình cục bộ. Đổi `k` của kNN từ 1 hoặc 5 sang 25 làm F1 giảm từ **0,921** xuống **0,889** trong lần chạy này.

### 4. LLM qua CKEY

Tôi phân loại 40 tin SUB bằng `gemini-2.5-flash-lite` zero-shot, `gemini-3.8-flash` zero-shot và Flash-Lite few-shot. Prompt yêu cầu trả lời một nhãn `spam` hoặc `ham`; hàm đọc kết quả kiểm tra nhãn trước khi tính điểm. Tôi còn so sánh prompt thường và prompt phòng thủ trên 50 ca khó. Một output của 3.8 Flash rỗng hoặc không hợp lệ được quy về ham và tình cờ đúng nhãn, nên F1 1,000 của model đó cần đọc kèm giới hạn này.

## Kết quả chính

| Phương án | F1 test 775 | F1 SUB 40 | p95 mỗi tin |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,906 | 0,974 | 0,461 ms |
| BoW + MLP | 0,948 | 1,000 | 1,697 ms |
| MiniLM + kNN | 0,921 | 1,000 | 75,614 ms |
| MiniLM + LogReg | **0,959** | 1,000 | 38,031 ms |
| MiniLM + MLP head | 0,950 | 1,000 | 34,960 ms |
| Flash-Lite zero-shot | — | 0,976 | 7,308 giây |
| 3.8 Flash zero-shot | — | 1,000 | 54,119 giây |
| Flash-Lite few-shot | — | 1,000 | 5,998 giây |

Các hàng LLM chỉ có F1 trên SUB 40, không có F1 trên toàn bộ 775 tin test. Số p95 của LLM được ghi bằng **giây**, còn mô hình cục bộ bằng **mili giây**. Độ trễ và thông lượng đo trong notebook phụ thuộc máy và điều kiện API lúc chạy. Chi phí API trong báo cáo là phép ước tính theo số token xấp xỉ và giá CKEY tại thời điểm thí nghiệm, không phải hóa đơn.

## Điều tôi rút ra từ 50 ca khó

BoW + LogReg có F1 **0,294** trên bộ khó, giảm mạnh so với **0,974** trên SUB. Với nhóm spam giọng tự nhiên, mô hình này không đoán đúng ca nào trong 10 ca. LLM Flash-Lite zero-shot đạt F1 **0,885** và accuracy **88%** trên 50 ca; prompt phòng thủ nâng lên F1 **0,966** và accuracy **96%**.

Prompt phòng thủ giúp nhóm spam giọng tự nhiên từ **4/10** lên **9/10**, nhưng làm sai thêm một tin ham trông giống spam. Cả prompt thường và prompt phòng thủ đều đúng **6/6** ca prompt injection trong bộ này, vì vậy tôi không thể nói prompt phòng thủ cải thiện riêng nhóm đó. Có một ca (#13) mà mọi phương án được thử đều sai: tin bán ảnh được viết như lời nhắc thân thiện.

## Năm phép tính viết tay

Ảnh trong `report/individual/hand_calc.jpg` ghi lại: accuracy nếu luôn đoán ham khoảng **0,87**; sigmoid tại 0 bằng **0,5**; F1 với TP = 80, FP = 20, FN = 40 khoảng **0,73**; cosine của `[1, 2, 0]` và `[2, 1, 2]` khoảng **0,60**; chi phí ví dụ một triệu SMS/ngày là **10,50 USD/ngày** theo đơn giá trong đề.

Ở dòng F1 trên giấy, tôi ghi thiếu hệ số **2** trong công thức nhưng ra kết quả đúng. Công thức đầy đủ là `F1 = 2 × precision × recall / (precision + recall) = 160 / 220 ≈ 0,7273`. Tôi giữ nguyên ảnh gốc và ghi chú lại lỗi ghi công thức này trong [báo cáo](report/individual/2A202603646_NguyenManhTuong.md).

## Kết luận

Nếu cần lọc 10 triệu SMS/ngày với yêu cầu p95 dưới 50 ms/tin, tôi chọn MiniLM + LogReg làm mô hình chính vì F1 test **0,959** và p95 đo được **38,031 ms**. Tôi có thể dùng BoW + MLP làm tầng đầu cho các tin dễ rồi chuyển những ca không chắc chắn sang MiniLM. Đây vẫn chỉ là kết quả trên môi trường thử: thông lượng một máy chưa đủ mức trung bình khoảng 116 tin/giây, và tôi chưa đo p95 của cả hệ thống hai tầng dưới tải đồng thời.

LLM cho kết quả tốt trên bộ ca khó nhưng độ trễ vài giây đến hàng chục giây không đáp ứng SLA trực tiếp. Tôi chỉ xem LLM là phương án hỗ trợ xử lý bất đồng bộ cho những ca cần xem xét thêm. Phần dự đoán, nhận xét chi tiết và AI log nằm trong [báo cáo cá nhân](report/individual/2A202603646_NguyenManhTuong.md).
