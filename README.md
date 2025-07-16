# Fake News Detection Project

## Giới thiệu

**Fake News Detection Project** là dự án xây dựng hệ thống phát hiện tin giả bằng các phương pháp học máy và xử lý ngôn ngữ tự nhiên. Hệ thống tự động nhận diện và phân loại các bài báo hoặc tin tức thành thật hoặc giả dựa trên nội dung văn bản.

---

## Quy trình thực hiện

Dự án gồm các bước chính:

1. **Xử lý dữ liệu (Data Preprocessing)**
2. **Xây dựng & huấn luyện mô hình (Model Training)**
3. **Đánh giá mô hình (Model Evaluation)**
4. **Áp dụng mô hình để dự đoán (Prediction)**

---

## 1. Xử lý dữ liệu

### a. Thu thập dữ liệu

- Dữ liệu được thu thập từ nguồn đáng tin cậy (Kaggle, API tin tức, v.v.).
- Định dạng thường là CSV hoặc JSON với các trường: `title`, `text`, `label`.

### b. Làm sạch và tiền xử lý dữ liệu

- Xóa dòng bị thiếu dữ liệu, dòng trùng lặp.
- Chuẩn hóa chữ thường toàn bộ văn bản.
- Loại bỏ ký tự đặc biệt, số, dấu câu.
- Loại bỏ stopwords (các từ không mang nhiều ý nghĩa).
- Tokenization: tách văn bản thành các từ.
- Stemming/Lemmatization: đưa về từ gốc.

### c. Vector hóa dữ liệu

- Sử dụng các phương pháp như Bag of Words, TF-IDF, hoặc Word Embedding để biến văn bản thành dạng số phục vụ học máy.

### d. Chia tập dữ liệu

- Chia thành tập huấn luyện (train) và kiểm tra (test), ví dụ 80% train, 20% test.

---

## 2. Xây dựng & Huấn luyện mô hình

- Sử dụng các thuật toán như Logistic Regression, Random Forest, SVM, hoặc các mô hình Deep Learning (LSTM, CNN).
- Huấn luyện mô hình với dữ liệu đã được xử lý và vector hóa.
- Lưu mô hình đã huấn luyện dưới dạng file (pickle, joblib).

---

## 3. Đánh giá mô hình

- Sử dụng các chỉ số: Accuracy, Precision, Recall, F1-score để đánh giá chất lượng mô hình.
- Có thể sử dụng Cross-validation để đánh giá tổng quát.

---

## 4. Áp dụng mô hình (Dự đoán)

- Sử dụng mô hình đã huấn luyện để dự đoán nhãn cho bài báo/tin tức mới.
- Đầu vào: nội dung văn bản; Đầu ra: nhãn "Giả" hoặc "Thật".



