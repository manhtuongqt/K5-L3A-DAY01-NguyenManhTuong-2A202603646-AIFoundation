# Các bước tính để tự viết tay

Ảnh `hand_calc.jpg` phải là ảnh giấy do học viên tự viết. Tệp này chỉ ghi các phép tính để đối chiếu.

1. Luôn đoán `ham`: sau bỏ trùng có 4.518 ham và 5.171 tin, nên accuracy = 4.518 / 5.171 = 0,8737 ≈ **0,87**.
2. Sigmoid tại 0: σ(0) = 1 / (1 + e⁰) = 1 / 2 = **0,50**.
3. TP = 80, FP = 20, FN = 40. Precision = 80 / 100 = 0,80; recall = 80 / 120 = 0,6667; F1 = 2TP / (2TP + FP + FN) = 160 / 220 = 0,7273 ≈ **0,73**.
4. Cosine([1, 2, 0], [2, 1, 2]): tích vô hướng = 1·2 + 2·1 + 0·2 = 4; độ dài = √5 và 3; cosine = 4 / (3√5) = 0,5963 ≈ **0,60**.
5. Một triệu SMS/ngày, 100 token input/tin và 1 token output/tin: input = 100 triệu token = 10 USD; output = 1 triệu token = 0,50 USD; tổng = **10,50 USD/ngày** theo đơn giá trong đề.
