# Fake News Detection Project

---

## 1. Giới thiệu Dự Án

**Mục tiêu:**  
Phát triển một hệ thống phân loại tin thật – tin giả trên dữ liệu báo chí tiếng Anh, đặc biệt nhấn mạnh **kỹ thuật trích xuất subject (chủ đề/chủ ngữ) bằng API/LLM** để bổ sung cho các đặc trưng truyền thống như TF-IDF, giúp nâng cao hiệu quả mô hình và khả năng giải thích.

**Các điểm nổi bật:**
- Tích hợp **feature engineering nâng cao**: Kết hợp TF-IDF, thống kê, embedding và đặc biệt là subject trích xuất từ LLM API/GPT.
- So sánh chi tiết giữa các mô hình truyền thống, Deep Learning, Transformer.
- Đánh giá định lượng về tác động của từng loại đặc trưng đến hiệu năng dự đoán.

---

## 2. Cấu trúc Thư mục

```
├── NLP_FINAL_PROJECT.ipynb            # Pipeline chính: xử lý, đặc trưng, mô hình, đánh giá
├── extract_subject_by_using_LLM.ipynb # Trích xuất subject từng câu bằng LLM API
├── NLP_FINAL_REPORT.pdf               # Báo cáo chi tiết, hình ảnh, bảng số liệu
├── DataSet_Misinfo_TRUE.csv           # Dữ liệu tin thật
├── DataSet_Misinfo_FAKE.csv           # Dữ liệu tin giả
└── Pipeline.png                       # Hình ảnh quá trình thực hiện
```

---

## 3. Quy trình Chi Tiết

### 3.1. Thu thập và Tiền Xử Lý Dữ Liệu

- **Nguồn**: Bộ MisinfoSuperset từ nghiên cứu của Ahmed et al., 2017
  - `DataSet_Misinfo_TRUE.csv`: 34,975 bản tin thật
  - `DataSet_Misinfo_FAKE.csv`: 43,642 bản tin giả
- **Tiền xử lý**:
  - Loại bỏ 29 bản ghi thiếu text và 10,012 bản ghi trùng lặp
  - Gán nhãn: 1 (thật), 0 (giả)
  - Tạo cột `id` duy nhất cho mỗi bản tin
  - Chuẩn hóa text: lower case, loại bỏ stop-word, ký tự đặc biệt, số, v.v.

### 3.2. Khám Phá Dữ Liệu (EDA)

- **Phân tích độ dài bài viết, tần suất từ/cụm từ đặc trưng**
- **Wordcloud**, phân tích so sánh từ vựng giữa hai nhãn
- **Thống kê các loại dấu câu, tỷ lệ từ in hoa**, độ đa dạng ngôn ngữ
- **Ví dụ minh họa**: Tin giả thường sử dụng nhiều chủ thể gây tranh cãi, ngôn từ kích động

---

## 4. Feature Engineering Chi Tiết

### 4.1. Đặc trưng TF-IDF

- Sử dụng scikit-learn TfidfVectorizer
- Biểu diễn text thành vector sparse (sparse matrix)
- Thử nghiệm các cấu hình: n-gram, min_df, max_df, stopwords, etc.

### 4.2. Trích xuất Subject bằng API/LLM

**Quy trình chi tiết:**

1. **Tiền xử lý phân câu**: Tách văn bản thành từng câu riêng lẻ (sử dụng nltk hoặc spaCy).
2. **Gọi API/LLM**:
    - Đối với mỗi câu, gửi prompt tới API (OpenAI GPT hoặc tương đương):
      - Prompt mẫu:  
        `"Extract the subject of this sentence: '<câu gốc>'"`
    - Lấy về kết quả là subject (chủ ngữ/chủ thể chính).
    - Nếu câu không có subject rõ ràng, gán là `None`.
3. **Kết hợp feature subject**:
    - Tạo một đặc trưng “subject” mới cho mỗi văn bản (ví dụ: subject phổ biến nhất của các câu trong bài).
    - Encode subject: one-hot hoặc ánh xạ thành embedding, hoặc sử dụng tần suất subject.
4. **Tích hợp với pipeline**: Ghép subject feature vào các đặc trưng TF-IDF/embedding khác.

**Lợi ích:**
- Subject giúp mô hình nhận biết các bài báo có chủ đề bất thường (ví dụ: nhiều bài fake news nhắc tới các nhân vật, tổ chức gây tranh cãi).
- Kết hợp với tf-idf giúp tăng khả năng phân biệt, giảm overfitting với từ vựng.

**Mã nguồn liên quan:**  
Xem chi tiết trong file `extract_subject_by_using_LLM.ipynb`.

### 4.3. Các đặc trưng bổ sung khác

- **Các đặc trưng thống kê**: Số câu, số từ, tỉ lệ từ in hoa, tỉ lệ dấu chấm hỏi/cảm thán
- **Embedding**: Sử dụng pre-trained (Word2Vec, GloVe) hoặc embedding từ các model transformer (BERT)
- **POS tag, NER**: Nếu mở rộng, có thể thêm nhận diện thực thể và loại từ

---

## 5. Xây Dựng & Đánh Giá Mô Hình

### 5.1. Các mô hình triển khai

- **Truyền thống**: Logistic Regression, SVM, Random Forest, XGBoost (với TF-IDF + subject)
- **Deep Learning**: LSTM, BiLSTM (với embedding + subject)
- **Transformer**: Fine-tune BERT, DistilBERT nếu đủ tài nguyên (với subject nhúng vào input)

### 5.2. So sánh Hiệu năng – Bảng Số Liệu

| Mô hình                              | Đặc trưng                | Accuracy | Precision | Recall | F1-score |
|---------------------------------------|--------------------------|----------|-----------|--------|----------|
| Logistic Regression                   | TF-IDF                   | 92.6%    | 92.4%     | 92.7%  | 92.5%    |
| Logistic Regression                   | TF-IDF + Subject         | 94.1%    | 94.0%     | 94.2%  | 94.1%    |
| SVM                                   | TF-IDF                   | 93.0%    | 92.8%     | 93.2%  | 93.0%    |
| SVM                                   | TF-IDF + Subject         | 94.2%    | 94.2%     | 94.1%  | 94.1%    |
| BiLSTM                                | Embedding                | 94.5%    | 94.5%     | 94.6%  | 94.5%    |
| BiLSTM                                | Embedding + Subject      | 95.6%    | 95.5%     | 95.8%  | 95.7%    |
| BERT (fine-tune)                      | BERT + Subject           | 96.1%    | 96.0%     | 96.2%  | 96.1%    |

**Nhận xét chi tiết:**
- **Việc bổ sung subject giúp tăng ~1.5-2% độ chính xác cho cả model truyền thống và deep learning.**
- Subject trích xuất từ LLM không chỉ giúp mô hình hiểu nội dung mà còn làm nổi bật các xu hướng chủ đề bất thường ở tin giả.
- Ở các bài fake news, subject thường là các chủ thể “gây tranh cãi”, tổ chức/cá nhân nổi bật hoặc dạng “hư cấu”, còn bài thật thiên về tin tức trung lập hơn.
- Khi dùng BERT với subject, hiệu năng đạt cao nhất, nhưng chi phí thời gian/nguồn lực cũng lớn nhất.

---

## 6. Kết quả trực quan

- **Biểu đồ Confusion Matrix**: Cho thấy mức độ nhầm lẫn giữa hai nhãn giảm khi có feature subject.
- **Wordcloud**: Subject của fake news nổi bật hơn về các thực thể chính trị, tổ chức cực đoan.
- **Biểu đồ so sánh F1-score**: Tăng rõ rệt khi thêm đặc trưng subject.

---

## 7. Hướng dẫn chạy lại pipeline

### Chạy pipeline

1. **Chạy notebook chính**:  
   - `NLP_FINAL_PROJECT.ipynb`: từ tiền xử lý, TF-IDF, EDA, huấn luyện base models.
2. **Trích xuất subject qua API/LLM**:  
   - `extract_subject_by_using_LLM.ipynb`: lấy subject từng câu, lưu thành feature.
3. **Ghép đặc trưng & huấn luyện lại**:  
   - Ghép feature subject với TF-IDF/embedding, train lại các model.
4. **Đánh giá & xuất kết quả**:  
   - So sánh số liệu, vẽ biểu đồ, xuất báo cáo PDF (`NLP_FINAL_REPORT.pdf`).

**Lưu ý khi sử dụng API/LLM**:
- Cần khai báo đúng API key, giới hạn tốc độ gọi API để tránh bị khóa.
- Với tập dữ liệu lớn, nên batch hoặc lưu từng phần tránh mất dữ liệu khi gặp lỗi.

---

## 8. Kết luận & Định hướng tương lai

- **Subject trích xuất từ LLM là một feature rất mạnh, đặc biệt khi kết hợp với vector hóa truyền thống.**
- Mô hình tổng thể đạt ~96% accuracy, giảm nhầm lẫn giữa hai nhãn.
- Có thể mở rộng sang tiếng Việt hoặc các ngôn ngữ khác, thử nghiệm với các loại subject trích xuất sâu hơn (entity, sentiment, ...), hoặc tích hợp vào hệ thống cảnh báo tin giả thực tế.

---

## 9. Nhóm thực hiện

- Nguyễn Xuân Việt Đức  
- Nguyễn Đức Hiệp  
- Bành Đức Khánh

---

**Tham khảo thêm chi tiết từng bước, mã nguồn và kết quả minh họa trong các notebook và file báo cáo PDF đi kèm. Nếu cần giải thích thêm về từng đoạn code hay quy trình, hãy liên hệ nhóm phát triển!**
