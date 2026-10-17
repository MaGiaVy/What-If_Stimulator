**TÀI LIỆU TỔNG HỢP DATASET & HƯỚNG DẪN TRIỂN KHAI ĐỒ ÁN**

**ĐỀ TÀI: HỆ THỐNG CẢNH BÁO SỚM HỌC VỤ VÀ MÔ PHỎNG KỊCH BẢN HỌC TẬP 2 TẦNG**

*(Bản tài liệu thuyết minh thuần văn bản, trình bày chi tiết ưu nhược điểm 10 bộ dữ liệu và quy trình thực hiện từng bước)*

# **PHẦN 1: TỔNG HỢP & ĐÁNH GIÁ 5 BỘ DỮ LIỆU ĐỀ TÀI 1 (DỰ ĐOÁN ĐIỂM THI TỪ GIỜ HỌC VÀ GIẤC NGỦ)**

Mục tiêu của Đề tài 1 là dự đoán kết quả học tập hoặc điểm thi cuối kỳ của sinh viên dựa trên số giờ tự học, thời lượng giấc ngủ và các thói quen sinh hoạt hàng ngày. Dưới đây là phân tích chi tiết ưu, nhược điểm của từng bộ dữ liệu:

### **1.1. Student Exam Performance Prediction Dataset (Hugging Face)**

* **Quy mô & Biến mục tiêu:** 20 sinh viên (rất nhỏ) | Biến mục tiêu: FinalExamScore (thang điểm 100 \- bài toán Hồi quy).  
* **Đặc trưng chính:** StudyHours (giờ học mỗi ngày), SleepHours (giờ ngủ mỗi ngày), SocialMediaHours (giờ dùng mạng xã hội), AttendanceRate (tỷ lệ chuyên cần).  
* **Ưu điểm:** Cấu trúc cực kỳ gọn nhẹ, dễ hiểu, nhìn thấy ngay mối quan hệ tuyến tính giữa số giờ học và điểm thi; phù hợp để tính nhẩm hoặc vẽ đồ thị minh họa nhanh.  
* **Nhược điểm:** Quy mô 20 dòng quá nhỏ để huấn luyện mô hình Học máy (Machine Learning). Khi chia tập Huấn luyện/Kiểm tra theo tỷ lệ 80/20 thì tập Kiểm tra chỉ còn 4 sinh viên, dẫn đến hiện tượng học vẹt (overfitting) và sai số đánh giá rất lớn.  
* **Đánh giá độ phù hợp:** Rất thấp (Chỉ dùng để minh họa công thức, không nên dùng làm dữ liệu chính cho đồ án).

### **1.2. Student Habits & Performance Dataset (GitHub)**

* **Quy mô & Biến mục tiêu:** 1.000 bản ghi sinh viên | Biến mục tiêu: final\_exam\_score (điểm số liên tục \- bài toán Hồi quy).  
* **Đặc trưng chính:** Hơn 15 đặc trưng bao gồm study\_hours (giờ học), sleep (giấc ngủ), diet (chế độ ăn uống), social\_media (mạng xã hội), exercise (tập thể dục), attendance (điểm danh)...  
* **Ưu điểm:** Quy mô 1.000 dòng vừa vặn cho một đồ án môn học; định dạng CSV sạch sẽ; có nhiều biến thói quen sinh hoạt phong phú để vẽ biểu đồ phân tích khám phá dữ liệu (EDA).  
* **Nhược điểm:** Là dữ liệu tổng hợp (synthetic) chia sẻ trên GitHub nên ít bài báo khoa học tham chiếu hơn so với các bộ dữ liệu trên Kaggle hoặc UCI.  
* **Đánh giá độ phù hợp:** Rất tốt (Lựa chọn phù hợp thứ 2 cho bài toán dự đoán điểm thi).

### **1.3. Student Performance Factors (Kaggle) — \[KHUYÊN DÙNG NHẤT\]**

* **Quy mô & Biến mục tiêu:** 6.607 sinh viên với 20 cột | Biến mục tiêu: Exam\_Score (Hồi quy điểm số) hoặc gắn nhãn at\_risk (Phân loại).  
* **Đặc trưng chính:** Hours\_Studied (giờ học mỗi tuần), Sleep\_Hours (giờ ngủ mỗi đêm), Attendance (tỷ lệ điểm danh), Previous\_Scores (điểm học kỳ trước), Tutoring\_Sessions (số buổi học phụ đạo), Motivation\_Level (mức độ động lực)...  
* **Ưu điểm:** Quy mô 6.607 dòng cực kỳ lý tưởng để chia tập Train/Validation/Test; dữ liệu sạch, ít nhiễu; có đầy đủ các biến quan trọng để vừa dự đoán điểm số, vừa phân tích sự kết hợp giữa Giờ học \+ Giờ ngủ \+ Chuyên cần.  
* **Nhược điểm:** Vì là dữ liệu tổng hợp có quy luật tạo sẵn nên phân phối dữ liệu khá chuẩn mực, ít các trường hợp ngoại lai (outliers) cực đoan như dữ liệu thu thập thủ công.  
* **Đánh giá độ phù hợp:** Tốt nhất (Được chọn làm bộ dữ liệu chính cho Tầng 2 của hệ thống).

### **1.4. Student Academic Performance Prediction AIML (IEEE DataPort)**

* **Quy mô & Biến mục tiêu:** 1.100 phản hồi khảo sát | Biến mục tiêu: Overall Academic Performance (Phân loại 4 mức: Excellent, Good, Average, Poor).  
* **Đặc trưng chính:** Daily Study Hours (giờ học mỗi ngày), Average Sleep Hours per Day (giờ ngủ trung bình), Average College Attendance (chuyên cần), Concentration Level (mức độ tập trung).  
* **Ưu điểm:** Dữ liệu khảo sát thực tế, định dạng Excel (.xlsx) hoàn chỉnh, không có giá trị bị khuyết thiếu (missing values).  
* **Nhược điểm:** Biến mục tiêu là nhãn phân loại học lực bằng chữ chứ không phải con số điểm thi cụ thể; ngoài ra một số tài liệu trên IEEE DataPort yêu cầu tài khoản thành viên để tải.  
* **Đánh giá độ phù hợp:** Thấp \- Trung bình (Chỉ phù hợp nếu muốn dự đoán xếp loại học lực thay vì điểm số).

### **1.5. Student Academic Performance and Lifestyle Dataset (Mendeley)**

* **Quy mô & Biến mục tiêu:** 19.222 bản ghi sinh viên ẩn danh | Biến mục tiêu: Grades (A, B, C, D, F \- Phân loại đa lớp).  
* **Đặc trưng chính:** 14 đặc trưng bao gồm study\_hours (giờ học), sleep\_duration (thời lượng ngủ), gaming\_hours (giờ chơi game), stress\_levels (mức độ căng thẳng)...  
* **Ưu điểm:** Quy mô lớn gần 20.000 dòng, rất phù hợp nếu nhóm muốn thử nghiệm các thuật toán phức tạp như XGBoost, LightGBM hoặc mạng học sâu (Deep Learning).  
* **Nhược điểm:** Biến mục tiêu là điểm chữ (A, B, C, D, F) thay vì điểm số liên tục (0 \- 100), nên bản chất là bài toán Phân loại 5 lớp chứ không phải bài toán Hồi quy.  
* **Đánh giá độ phù hợp:** Trung bình \- Khá (Rất tốt nếu đổi đề tài sang dự đoán điểm chữ A-F).

## **1.6. Bảng so sánh tổng hợp 5 bộ dữ liệu Đề tài 1**

| Tên bộ dữ liệu | Quy mô | Biến mục tiêu | Loại bài toán | Mức độ khuyên dùng |
| :---- | :---- | :---- | :---- | :---- |
| **3\. Student Performance Factors (Kaggle)** | 6.607 dòng | Exam\_Score (0-100) | Hồi quy (Regression) | Tốt nhất (Khuyên dùng) |
| **2\. Student Habits & Performance (GitHub)** | 1.000 dòng | final\_exam\_score | Hồi quy (Regression) | Rất tốt |
| **5\. Lifestyle Dataset (Mendeley)** | 19.222 dòng | Grades (A, B, C, D, F) | Phân loại 5 lớp | Trung bình |
| **4\. AIML Dataset (IEEE DataPort)** | 1.100 dòng | Học lực (4 mức) | Phân loại 4 lớp | Thấp |
| **1\. Exam Performance (Hugging Face)** | 20 dòng | FinalExamScore | Hồi quy (Regression) | Rất thấp (Quá nhỏ) |

# **PHẦN 2: TỔNG HỢP & ĐÁNH GIÁ 5 BỘ DỮ LIỆU ĐỀ TÀI 2 (DỰ ĐOÁN SINH VIÊN BỎ HỌC & GỘP BẢNG)**

Mục tiêu của Đề tài 2 là dự đoán sớm sinh viên có nguy cơ bỏ học (Dropout) dựa trên kết quả học tập, điểm danh và có yêu cầu thực hành kỹ năng gộp nhiều bảng dữ liệu (Merge Data). Dưới đây là đánh giá chi tiết 5 bộ dữ liệu:

### **2.1. OULAD \- Open University Learning Analytics Dataset (UCI / Kaggle)**

* **Quy mô & Biến mục tiêu:** Hơn 32.000 sinh viên, gồm 7 bảng rời nhau | Biến mục tiêu: final\_result (Pass, Fail, Withdrawn, Distinction).  
* **Đặc trưng & Khóa liên kết:** Liên kết 7 bảng qua khóa student\_id, code\_module và code\_presentation. Bao gồm thông tin cá nhân (studentInfo), điểm các bài kiểm tra (studentAssessment), đăng ký môn (studentRegistration) và lịch sử click chuột trên hệ thống LMS/VLE (studentVle).  
* **Ưu điểm:** Là bộ dữ liệu kinh điển nhất về phân tích học tập; đáp ứng hoàn hảo yêu cầu tự tay gom nhóm (groupby) và gộp (merge) 7 bảng dữ liệu thực tế; dữ liệu không sạch sẵn và có độ mất cân bằng thực tế.  
* **Nhược điểm:** Không có cột 'tỷ lệ điểm danh trên lớp' truyền thống mà phải quy đổi từ tổng lượt tương tác trực tuyến (sum\_click); bảng studentVle nặng hơn 10 triệu dòng nên đòi hỏi xử lý cẩn thận để tránh tràn bộ nhớ RAM.  
* **Đánh giá độ phù hợp:** Xuất sắc nếu nhóm muốn thể hiện kỹ năng xử lý dữ liệu lớn và gộp nhiều bảng phức tạp.

### **2.2. Predict Students' Dropout and Academic Success (UCI / Mendeley) — \[KHUYÊN DÙNG NHẤT\]**

* **Quy mô & Biến mục tiêu:** 4.424 sinh viên với 35 cột ở bảng chính \+ Bảng phụ trên Mendeley | Biến mục tiêu: Target (Dropout, Enrolled, Graduate).  
* **Đặc trưng & Khóa liên kết:** Chứa điểm trung bình và số môn qua ở học kỳ 1 & học kỳ 2, tình trạng đóng học phí, nợ môn, học bổng. Khi gộp (merge) với bảng phụ college\_student\_management\_data.csv sẽ bổ sung thêm cột tỷ lệ điểm danh (attendance\_rate) và chỉ số tương tác.  
* **Ưu điểm:** Quy mô 4.424 dòng vừa sức, chạy mô hình rất nhanh; các biến học thuật và tài chính cực kỳ rõ ràng; vừa thực hành được thao tác merge 2 bảng (bảng chính \+ bảng phụ), vừa đạt độ chính xác dự báo rất cao.  
* **Nhược điểm:** Phiên bản trên UCI chỉ có 1 bảng duy nhất, do đó nhóm cần tải phiên bản đầy đủ trên Mendeley Data (có kèm bảng phụ college\_student\_management\_data.csv) để thực hiện bước gộp bảng.  
* **Đánh giá độ phù hợp:** Tốt nhất (Sự lựa chọn cân bằng nhất giữa yêu cầu gộp bảng, thời gian làm bài và hiệu quả mô hình).

### **2.3. A Multi-Source Unified Dataset for Student Retention Modeling**

* **Quy mô & Biến mục tiêu:** 88.518 quan sát tổng hợp từ 3 nguồn (risk\_students, student\_dropout\_v3, enrollment\_outcomes) | Biến mục tiêu: nhãn nhị phân is\_at\_risk.  
* **Đặc trưng & Khóa liên kết:** Tích hợp sẵn tỷ lệ điểm danh (attendance\_rate), điểm trung bình (GPA), số giờ học mỗi ngày (study\_hours) và các chỉ số hành vi.  
* **Ưu điểm:** Quy mô lớn, chứa đầy đủ các biến giao thoa giữa cả Đề tài 1 (giờ học, GPA) và Đề tài 2 (điểm danh, nguy cơ bỏ học).  
* **Nhược điểm:** Nhiều kho lưu trữ chỉ cung cấp 1 file duy nhất đã được tác giả gộp sẵn (Unified). Nếu không tải được 3 bảng rời ban đầu thì nhóm sẽ không thể hiện được bước tự viết lệnh merge dữ liệu.  
* **Đánh giá độ phù hợp:** Rất tốt (Nếu tải được bản tách rời 3 nguồn ban đầu hoặc dùng làm dữ liệu đối chứng).

### **2.4. Student Dropout Prediction (Kaggle)**

* **Quy mô & Biến mục tiêu:** Quy mô nhỏ \- vừa (1 bảng CSV duy nhất) | Biến mục tiêu: Dropout (Có / Không).  
* **Đặc trưng & Khóa liên kết:** Có sẵn các cột trực diện như attendance (tỷ lệ điểm danh), marks (điểm số), tình trạng tài chính và hành vi nộp bài tập.  
* **Ưu điểm:** Cấu trúc đơn giản, dễ hiểu, tốn ít công sức tiền xử lý.  
* **Nhược điểm:** Chỉ gồm 1 bảng đơn lẻ nên không đáp ứng được yêu cầu gộp bảng (merge) của bạn.  
* **Đánh giá độ phù hợp:** Thấp (Chỉ hợp làm bài tập thực hành nhỏ trên lớp).

### **2.5. Multimodal Student Retention Dataset (Kaggle)**

* **Quy mô & Biến mục tiêu:** 2.134 bản ghi | Biến mục tiêu chi tiết: dropout\_risk, dropout\_type, dropout\_cause.  
* **Đặc trưng & Khóa liên kết:** Kết hợp dữ liệu có cấu trúc (GPA, tỷ lệ điểm danh) và dữ liệu phi cấu trúc (ghi chú bằng văn bản của cố vấn học tập).  
* **Ưu điểm:** Cung cấp góc nhìn sâu sắc về cả nguyên nhân và phân loại hình thức bỏ học.  
* **Nhược điểm:** Đòi hỏi phải kết hợp kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) để phân tích cột văn bản ghi chú; không tập trung vào bài toán gộp nhiều bảng dữ liệu quan hệ.  
* **Đánh giá độ phù hợp:** Trung bình (Chỉ nên chọn nếu nhóm muốn nghiên cứu mô hình đa phương thức Tabular \+ Text).

## **2.6. Bảng so sánh tổng hợp 5 bộ dữ liệu Đề tài 2**

| Tên bộ dữ liệu | Quy mô | Yêu cầu Merge bảng | Đặc trưng Điểm & Điểm danh | Mức độ khuyên dùng |
| :---- | :---- | :---- | :---- | :---- |
| **2\. Predict Students' Dropout (UCI/Mendeley)** | 4.424 dòng | Vừa sức (Gộp bảng chính \+ bảng phụ) | Điểm HK1, HK2, Tín chỉ, Điểm danh | Tốt nhất (Khuyên dùng) |
| **1\. OULAD (Open University)** | 32.593 SV | Phức tạp (Gộp 7 bảng quan hệ) | Điểm kiểm tra, Lượt click VLE | Xuất sắc (Nặng kỹ thuật) |
| **3\. Multi-Source Unified Dataset** | 88.518 dòng | Tốt (Nếu tải bản 3 bảng rời) | GPA, Attendance Rate, Giờ học | Rất tốt |
| **5\. Multimodal Retention Dataset** | 2.134 dòng | Không chuyên về merge bảng | GPA, Điểm danh \+ Văn bản ghi chú | Trung bình (Cần NLP) |
| **4\. Student Dropout (Kaggle)** | Nhỏ \- Vừa | Không (Chỉ có 1 bảng đơn) | Attendance, Marks | Thấp (Quá đơn giản) |

# **PHẦN 3: PHÁT TRIỂN ĐỀ TÀI THEO HƯỚNG 2 — HỆ THỐNG CẢNH BÁO SỚM HỌC VỤ VÀ MÔ PHỎNG KỊCH BẢN HỌC TẬP 2 TẦNG**

## **3.1. Đặt vấn đề và Lý do nâng cấp đề tài**

Nếu chỉ thực hiện riêng rẽ việc dự đoán điểm thi hoặc dự đoán sinh viên bỏ học theo cách truyền thống, đồ án rất dễ rơi vào lối mòn 'đại trà': đưa dữ liệu vào mô hình có sẵn và in ra độ chính xác. Trong thực tiễn giáo dục đại học, một con số dự báo đơn thuần không thể giúp sinh viên tiến bộ nếu thiếu đi giải pháp hành động cụ thể.

Vì vậy, việc kết hợp cả hai bài toán thành 'Hệ thống Cảnh báo sớm Nguy cơ Bỏ học và Mô phỏng Kịch bản Cải thiện Điểm số (What-If Simulator) 2 Tầng' giúp nâng tầm đề tài từ Phân tích Dự đoán (Predictive Analytics) lên Phân tích Đề xuất Hành động (Prescriptive Analytics).

## **3.2. Kiến trúc hoạt động của Hệ thống 2 Tầng**

| Tiêu chí | TẦNG 1: Sàng lọc & Cảnh báo Bỏ học (Cấp Vĩ mô) | TẦNG 2: Dự báo Điểm & Mô phỏng What-If (Cấp Vi mô) |
| :---- | :---- | :---- |
| **Mục tiêu nghiệp vụ** | Quét toàn bộ danh sách sinh viên đầu học kỳ/giữa kỳ để phát hiện sớm những cá nhân có nguy cơ bỏ học cao. | Đi sâu vào từng sinh viên bị cảnh báo ở Tầng 1 để tính toán lộ trình thay đổi giờ tự học, giấc ngủ và chuyên cần nhằm kéo điểm qua môn. |
| **Bản chất bài toán** | Bài toán Phân loại (Classification) kết hợp dự báo xác suất rủi ro từ 0% đến 100%. | Bài toán Hồi quy (Regression) kết hợp thuật toán tìm kiếm kịch bản giả định (What-If Simulation). |
| **Bộ dữ liệu sử dụng** | Predict Students' Dropout (UCI/Mendeley \- 4.424 sinh viên), thực hiện gộp bảng học vụ chính với bảng điểm danh phụ. | Student Performance Factors (Kaggle \- 6.607 sinh viên). |
| **Biến đầu vào cốt lõi** | Điểm trung bình HK1, Số môn đã qua, Tình trạng đóng học phí, Học bổng, Nợ môn, Tỷ lệ điểm danh (từ bảng phụ). | Số giờ tự học/tuần (Hours\_Studied), Số giờ ngủ/đêm (Sleep\_Hours), Tỷ lệ điểm danh (Attendance), Điểm kỳ trước (Previous\_Scores), Số buổi học phụ đạo. |
| **Kết quả đầu ra** | Phân loại sinh viên vào 3 nhóm: Nhóm An toàn (\<40%), Nhóm Cảnh báo Vàng (40% \- 60%) và Nhóm Nguy cơ Đỏ (\>60%). | Điểm thi cuối kỳ dự kiến (thang 100\) và Bảng 3 phương án điều chỉnh thói quen học tập tối ưu để vượt mức điểm an toàn (65 điểm). |

# **PHẦN 4: HƯỚNG DẪN QUY TRÌNH TRIỂN KHAI CHI TIẾT TỪNG BƯỚC CHO NHÓM**

## **Bước 1: Thu thập, Làm sạch và Gộp bảng Dữ liệu (Data Merging & Preprocessing)**

* **1\. Kiểm tra khóa chính trước khi gộp bảng:** Trên bộ dữ liệu Tầng 1, nhóm nạp bảng học vụ chính và bảng quản lý điểm danh phụ. Trước khi dùng lệnh gộp bảng (merge), bắt buộc phải kiểm tra và loại bỏ các dòng bị trùng lặp mã sinh viên (student\_id) ở bảng phụ để tránh làm phình số dòng sai lệch.  
* **2\. Thực hiện phép nối bảng (Left Join):** Sử dụng phép nối trái (Left Join) lấy bảng học vụ chính làm gốc, ghép cột tỷ lệ điểm danh (attendance\_rate) và tần suất đăng nhập hệ thống (lms\_login\_freq) từ bảng phụ sang bảng chính thông qua khóa student\_id.  
* **3\. Xử lý dữ liệu khuyết thiếu sau khi gộp:** Những sinh viên bị khuyết thông tin điểm danh sau khi nối bảng sẽ được điền bằng giá trị Trung vị (Median) tính theo từng ngành học (Course) tương ứng, giúp giữ nguyên phân phối tự nhiên của từng khoa.  
* **4\. Chuẩn hóa biến mục tiêu:** Chuyển đổi cột trạng thái học vụ (Target) thành biến nhị phân: gán nhãn 1 cho sinh viên Bỏ học (Dropout) và nhãn 0 cho sinh viên Tiếp tục học hoặc Tốt nghiệp (Enrolled / Graduate).

## **Bước 2: Phân tích Khám phá Dữ liệu (EDA) — 4 Biểu đồ Trọng tâm**

* **Biểu đồ 1 — Tỷ lệ mất cân bằng nhãn ở Tầng 1 (Pie Chart / Bar Chart):** Thể hiện tỷ lệ phần trăm giữa nhóm Bỏ học và Không bỏ học. Biểu đồ này là căn cứ khoa học để nhóm giải thích với hội đồng chấm thi vì sao cần áp dụng kỹ thuật cân bằng dữ liệu SMOTE ở Bước 3\.  
* **Biểu đồ 2 — So sánh Điểm học kỳ 1 và Tỷ lệ điểm danh giữa hai nhóm (Boxplot kép):** Chứng minh trực quan rằng nhóm sinh viên bỏ học có trung vị tỷ lệ điểm danh và điểm học kỳ 1 thấp hơn hẳn nhóm an toàn.  
* **Biểu đồ 3 — Ma trận tương quan các yếu tố ở Tầng 2 (Correlation Heatmap):** Thể hiện hệ số tương quan tuyến tính giữa Số giờ tự học, Số giờ ngủ, Tỷ lệ điểm danh, Điểm kỳ trước với Điểm thi cuối kỳ. Qua đó chỉ ra hai yếu tố có trọng số dương mạnh nhất là Giờ tự học và Tỷ lệ điểm danh.  
* **Biểu đồ 4 — Tác động kết hợp của Giờ học và Giấc ngủ lên Điểm thi (Scatter Plot phân nhóm):** Chia sinh viên thành 3 nhóm giấc ngủ (Thiếu ngủ \<=5h, Ngủ chuẩn 6-8h, Ngủ quá nhiều \>8h) và biểu diễn theo số giờ tự học. Biểu đồ cho thấy cùng một số giờ học nhưng nhóm ngủ đủ 6-8h luôn đạt điểm cao và ổn định hơn.

## **Bước 3: Huấn luyện Mô hình Cảnh báo Bỏ học ở Tầng 1 (Classification)**

* **1\. Chia tập dữ liệu có phân tầng (Stratified Split):** Chia dữ liệu Tầng 1 theo tỷ lệ 80% Huấn luyện (Train) và 20% Kiểm tra (Test), giữ nguyên tỷ lệ nhãn bỏ học ở cả hai tập.  
* **2\. Cân bằng dữ liệu bằng kỹ thuật SMOTE:** Áp dụng thuật toán SMOTE để sinh mẫu tổng hợp cho lớp thiểu số (Bỏ học). Lưu ý nguyên tắc quan trọng: chỉ áp dụng SMOTE trên tập Train, tuyệt đối giữ nguyên tập Test để đánh giá khách quan.  
* **3\. Huấn luyện và đối sánh đa mô hình:** Chạy thử nghiệm ít nhất 3 thuật toán từ cơ bản đến nâng cao gồm Logistic Regression (làm mô hình cơ sở), Random Forest Classifier và XGBoost Classifier.  
* **4\. Tinh chỉnh ngưỡng quyết định (Threshold Tuning) ưu tiên chỉ số Recall:** Thay vì dùng ngưỡng mặc định 50% (0.50), hệ thống hạ ngưỡng cảnh báo xuống 40% (0.40). Trong quản lý giáo dục, việc bỏ sót một sinh viên có nguy cơ bỏ học (False Negative) gây hậu quả lớn hơn nhiều so với việc nhắc nhở sớm một sinh viên đang an toàn (False Positive). Do đó, nhóm sử dụng Recall, F1-Score và ROC-AUC làm thước đo chính.

## **Bước 4: Huấn luyện Mô hình Dự báo Điểm & Xây dựng Bộ Giả lập What-If ở Tầng 2**

* **1\. Huấn luyện mô hình Hồi quy (Regression):** Sử dụng bộ dữ liệu Student Performance Factors (6.607 sinh viên), chia tập Train/Test 80/20 và huấn luyện 3 thuật toán: Linear Regression, Ridge Regression và Random Forest Regressor. Đánh giá sai số thông qua các chỉ số RMSE, MAE và hệ số xác định R-squared (R2).  
* **2\. Thiết kế thuật toán Mô phỏng Kịch bản (What-If Simulator):** Khi một sinh viên bị Tầng 1 gắn cờ Cảnh báo Vàng hoặc Nguy cơ Đỏ, hồ sơ của sinh viên đó (bao gồm Tỷ lệ điểm danh hiện tại và Điểm kỳ trước quy đổi sang thang 100\) sẽ được chuyển sang Tầng 2\.  
* **3\. Cơ chế đề xuất lộ trình tự động:** Hệ thống tự động thử nghiệm các kịch bản tăng dần số giờ tự học mỗi tuần (tăng từng nấc \+2 giờ/tuần), tăng tỷ lệ đi học chuyên cần (tăng từng nấc \+5%) và chuẩn hóa giấc ngủ về mức 7 \- 8 giờ/đêm, sau đó đưa vào mô hình Hồi quy để tính điểm dự kiến mới. Hệ thống lọc ra 3 phương án tốn ít công sức thay đổi nhất nhưng đảm bảo đưa điểm số dự kiến vượt mốc qua môn an toàn (65/100 điểm).

## **Bước 5: Thiết kế Giao diện Ứng dụng Tương tác (Web App Demo)**

* **Khung điều khiển bên trái (Dành cho Giáo vụ / Cố vấn học tập):** Cho phép nhập hoặc chọn hồ sơ sinh viên (Điểm HK1, Số môn qua, Học phí, Nợ môn, Điểm danh). Hệ thống hiển thị ngay phần trăm nguy cơ bỏ học kèm nhãn màu trực quan (Đỏ / Vàng / Xanh).  
* **Khung điều khiển bên phải (Dành cho Sinh viên tự mô phỏng):** Cung cấp các thanh trượt (Slider) để sinh viên kéo thử Số giờ tự học mỗi tuần, Số giờ ngủ mỗi đêm, Tỷ lệ điểm danh mục tiêu và Số buổi học phụ đạo. Ngay khi kéo thanh trượt, đồng hồ Điểm thi dự kiến và mức chênh lệch so với điểm qua môn sẽ cập nhật tức thì theo thời gian thực.

# **PHẦN 5: ĐỀ XUẤT CẤU TRÚC BÁO CÁO ĐỒ ÁN & PHÂN CÔNG CÔNG VIỆC NHÓM**

## **5.1. Đề cương chi tiết cuốn Báo cáo Đồ án (5 Chương)**

* **Chương 1 — Tổng quan Đề tài:** Đặt vấn đề, mục tiêu nghiên cứu, ý nghĩa thực tiễn của hệ thống cảnh báo sớm học vụ kết hợp mô phỏng lộ trình học tập, phạm vi và cấu trúc đồ án.  
* **Chương 2 — Cơ sở Lý thuyết & Khảo sát Dữ liệu:** Lý thuyết về bài toán Phân loại, Hồi quy, kỹ thuật xử lý mất cân bằng SMOTE, các độ đo đánh giá (Recall, F1, ROC-AUC, RMSE, R2); phân tích chi tiết các bộ dữ liệu và quy trình làm sạch, gộp bảng (Merge Data).  
* **Chương 3 — Phân tích Khám phá (EDA) & Thiết kế Hệ thống 2 Tầng:** Trình bày 4 biểu đồ EDA trọng tâm kèm nhận xét chuyên sâu; sơ đồ khối kiến trúc liên thông giữa Tầng 1 và Tầng 2; thiết kế thuật toán What-If Simulator.  
* **Chương 4 — Thực nghiệm, Đánh giá Kết quả & Xây dựng Ứng dụng:** Bảng so sánh kết quả thực nghiệm của các thuật toán ở cả 2 tầng; phân tích mức độ quan trọng của các đặc trưng (Feature Importance); minh họa giao diện Web App thực tế trên các ca điển hình (Case Studies).  
* **Chương 5 — Kết luận và Hướng phát triển:** Tổng kết các kết quả đạt được, hạn chế của đề tài và hướng mở rộng trong tương lai (tích hợp dữ liệu chuỗi thời gian theo từng tuần học trên bộ OULAD hoặc tích hợp AI tạo sinh tư vấn tự động).

## **5.2. Bảng phân công nhiệm vụ cho nhóm (3 \- 4 thành viên)**

| Thành viên | Mảng phụ trách chính | Công việc chi tiết & Sản phẩm bàn giao |
| :---- | :---- | :---- |
| **Thành viên 1 (Data & EDA)** | Kỹ thuật dữ liệu & Khám phá dữ liệu | Tải dữ liệu, kiểm tra trùng lặp, thực hiện gộp bảng (Merge) ở Tầng 1, xử lý dữ liệu khuyết, vẽ 4 biểu đồ EDA và viết Chương 1, Chương 2 của báo cáo. |
| **Thành viên 2 (ML Tầng 1\)** | Mô hình Phân loại Cảnh báo Bỏ học | Áp dụng SMOTE cân bằng dữ liệu, huấn luyện & so sánh Logistic Regression, Random Forest, XGBoost; tối ưu ngưỡng cảnh báo 40% và viết phần Thực nghiệm Tầng 1\. |
| **Thành viên 3 (ML Tầng 2\)** | Mô hình Hồi quy & Bộ giả lập What-If | Huấn luyện 3 mô hình Hồi quy dự đoán điểm thi trên bộ Kaggle, đánh giá RMSE/MAE/R2, xây dựng hàm gợi ý lộ trình học tập What-If và viết phần Thực nghiệm Tầng 2\. |
| **Thành viên 4 (App & Tổng hợp)** | Giao diện Web App & Hoàn thiện Báo cáo | Kết nối 2 mô hình vào giao diện Streamlit, chụp ảnh minh họa các tình huống sử dụng, viết Chương 4, Chương 5, ghép nối toàn bộ báo cáo Word và thiết kế Slide. |

