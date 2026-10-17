**project3\_ml\_modeling/**

**│**

**├── data/**

**│   ├── raw/                                     # Chứa dữ liệu gốc tải về (Tuyệt đối không chỉnh sửa trực tiếp)**

**│   │   ├── oulad/                               # 4 bảng cốt lõi của bộ dữ liệu OULAD (Tầng 1)**

**│   │   │   ├── studentInfo.csv                  # Thông tin nhân khẩu học, số tín chỉ và nhãn final\_result (32.593 dòng)**

**│   │   │   ├── assessments.csv                  # Thông tin cấu trúc bài kiểm tra (loại bài TMA/CMA/Exam, trọng số weight)**

**│   │   │   ├── studentAssessment.csv            # Điểm số chi tiết từng bài kiểm tra của sinh viên (173.912 dòng)**

**│   │   │   └── studentVle.csv                   # Lịch sử tương tác click chuột trên hệ thống LMS/VLE (>10 triệu dòng)**

**│   │   └── kaggle/                              # Bộ dữ liệu hành vi \& điểm thi (Tầng 2)**

**│   │       └── StudentPerformanceFactors.csv    # Dữ liệu giờ học, giờ ngủ, chuyên cần, điểm thi (6.607 dòng, 20 cột)**

**│   │**

**│   ├── processed/                               # Chứa bảng Master sau khi đã Merge, làm sạch và tạo Feature mới**

**│   │   ├── stage1\_oulad\_master.csv              # Bảng dữ liệu sạch Tầng 1 (đã có các biến tín chỉ VN \& quy về thang 10)**

**│   │   └── stage2\_performance\_master.csv        # Bảng dữ liệu sạch Tầng 2 (đã quy đổi điểm về thang 10)**

**│   │**

**│   └── splits/                                  # Chứa các tập dữ liệu cố định theo tỷ lệ 70% Train - 15% Dev - 15% Test**

**│       ├── stage1\_train.csv                     # Tập huấn luyện Tầng 1 (70% - Dùng để fit Preprocessor, SMOTE \& Model)**

**│       ├── stage1\_dev.csv                       # Tập phát triển Tầng 1 (15% - Dùng để tune гипер tham số \& Phân tích lỗi FN/FP)**

**│       ├── stage1\_test.csv                      # Tập kiểm thử Tầng 1 (15% - Chỉ mở ra đánh giá 1 lần cuối cùng)**

**│       ├── stage2\_train.csv                     # Tập huấn luyện Tầng 2 (70%)**

**│       ├── stage2\_dev.csv                       # Tập phát triển Tầng 2 (15% - Dùng để vẽ Residual Plot \& bắt Top 20 Outliers)**

**│       └── stage2\_test.csv                      # Tập kiểm thử Tầng 2 (15%)**

**│**

**├── notebooks/                                   # Chuỗi 4 Jupyter Notebooks thực thi tuần tự từ A -> Z**

**│   ├── 01\_data\_merging\_and\_fe.ipynb             # Notebook 1: Gộp bảng OULAD, chuẩn hóa thang 10 \& tạo biến đặc trưng VN**

**│   ├── 02\_eda\_and\_data\_splitting.ipynb          # Notebook 2: Phân tích khám phá dữ liệu (EDA) \& Chia tập 70/15/15**

**│   ├── 03\_stage1\_training\_and\_error\_analysis.ipynb # Notebook 3: Huấn luyện Tầng 1, tối ưu ngưỡng \& Phân tích lỗi FN/FP trên tập Dev**

**│   └── 04\_stage2\_training\_and\_whatif\_sim.ipynb  # Notebook 4: Huấn luyện Tầng 2, Phân tích lỗi Outliers \& Đóng gói What-If**

**│**

**├── reports/                                     # Lưu trữ kết quả xuất ra từ quá trình Phân tích lỗi (Error Analysis)**

**│   ├── figures/                                 # Chứa các biểu đồ EDA, Confusion Matrix, Residual Plot (.png, 300 DPI)**

**│   └── error\_logs/                              # Chứa các file CSV trích xuất ca lỗi trên tập Dev để đưa vào báo cáo Word**

**│       ├── stage1\_dev\_false\_negatives.csv       # Danh sách sinh viên bỏ học bị mô hình bỏ sót (Nguy hiểm nhất)**

**│       ├── stage1\_dev\_false\_positives.csv       # Danh sách sinh viên an toàn bị cảnh báo nhầm**

**│       └── stage2\_dev\_top20\_outliers.csv        # Top 20 sinh viên có sai số dự báo điểm lớn nhất (|y\_true - y\_pred| lớn nhất)**

**│**

**├── models/                                      # Đầu ra cuối cùng (Artifacts) bàn giao cho Project 1 (FastAPI) \& Project 2 (UI)**

**│   ├── pipeline\_stage1\_dropout.pkl              # Scikit-learn Pipeline Tầng 1 (Gộp chung Preprocessor + Best Classifier)**

**│   ├── pipeline\_stage2\_score.pkl                # Scikit-learn Pipeline Tầng 2 (Gộp chung Preprocessor + Best Regressor)**

**│   └── model\_metadata.json                      # Lưu ngưỡng cắt tối ưu (Optimal Threshold), danh sách tên cột và chỉ số đánh giá**

**│**

**└── requirements.txt                             # Danh sách thư viện đồng bộ phiên bản giữa Colab và Local**





**---**



**## 2. ĐẶC TẢ DATA ENGINEERING \& FEATURE ENGINEERING (ÁP DỤNG QUY CHẾ TÍN CHỈ VIỆT NAM)**



**Đây là phần tạo nên \*\*giá trị học thuật cốt lõi\*\* của đồ án. Thay vì chỉ lấy trung bình cộng đơn thuần, quy trình tiền xử lý trên bộ dữ liệu OULAD (`01\_data\_merging\_and\_fe.ipynb`) sẽ ánh xạ cấu trúc đánh giá của OULAD sang quy chế đào tạo tín chỉ đại học tại Việt Nam (tỷ lệ điểm Quá trình – Cuối kỳ `40-60` hoặc `50-50`).**



**### 2.1. Quy tắc gộp bảng và chuẩn hóa thang điểm 10**

**1. \*\*Khóa liên kết chuẩn (Primary/Composite Keys):\*\* Mọi phép gom nhóm (`groupby`) và nối bảng (`pd.merge` kiểu `LEFT JOIN`) trên OULAD đều phải thực hiện qua bộ 3 khóa: `KEYS = \['id\_student', 'code\_module', 'code\_presentation']`.**

**2. \*\*Quy đổi thang điểm 10:\*\***

&#x20;  **\* Trong bảng `studentAssessment.csv`, cột `score` gốc có giá trị từ `0` đến `100`. Thực hiện chia cho `10.0` ngay khi đọc dữ liệu: $\\text{score}\_{10} = \\frac{\\text{score}\_{100}}{10.0}$.**

&#x20;  **\* Trong bảng `StudentPerformanceFactors.csv` (Tầng 2), quy đổi 2 cột `Previous\_Scores` và `Exam\_Score` về thang 10:**

&#x20;    **$$\\text{Previous\\\_Scores}\_{10} = \\frac{\\text{Previous\\\_Scores}\_{100}}{10.0}, \\quad \\text{Exam\\\_Score}\_{10} = \\frac{\\text{Exam\\\_Score}\_{100}}{10.0}$$**

**3. \*\*Định nghĩa nhãn mục tiêu Tầng 1 (`is\_dropout` — Phương án A):\*\***

&#x20;  **\* Gán `is\_dropout = 1` nếu `final\_result` $\\in \\{\\text{'Withdrawn'}, \\text{'Fail'}\\}$ (Nhóm sinh viên bỏ học giữa chừng hoặc rớt môn cần cảnh báo học vụ).**

&#x20;  **\* Gán `is\_dropout = 0` nếu `final\_result` $\\in \\{\\text{'Pass'}, \\text{'Distinction'}\\}$ (Nhóm sinh viên hoàn thành môn học an toàn).**



**### 2.2. Công thức xây dựng 3 đặc trưng học vụ chuyên sâu (Feature Engineering)**



**Trong bảng `assessments.csv` của OULAD, các bài đánh giá được chia thành:**

**\* \*\*Điểm quá trình (Process Assessments):\*\* `assessment\_type` $\\in \\{\\text{'TMA'}, \\text{'CMA'}\\}$ (Tutor Marked Assessment / Computer Marked Assessment), có thời hạn nộp (`date`) trải dọc từ đầu kỳ đến cuối kỳ và có trọng số `weight`.**

**\* \*\*Điểm thi cuối kỳ:\*\* `assessment\_type == 'Exam'`.**



**> \*\*⚠️ LƯU Ý KỸ THUẬT CHỐNG RÒ RỈ DỮ LIỆU (ANTI-DATA LEAKAGE):\*\***  

**> Ở Tầng 1, mục tiêu là \*\*Cảnh báo sớm nguy cơ bỏ học\*\* trước khi kỳ thi cuối kỳ diễn ra. Trong thực tế OULAD, 99% sinh viên bỏ học (`Withdrawn`) sẽ không có điểm `Exam` cuối kỳ. Nếu đưa trực tiếp cột điểm `Exam` vào mô hình Tầng 1, mô hình sẽ "ăn gian" bằng cách cứ thấy `Exam == NaN` là phán `Dropout`. Do đó, ta chia quá trình học tập thành \*\*Nửa đầu học kỳ (Early Process: `date <= 100`)\*\* và \*\*Nửa sau học kỳ (Late Process: `date > 100`)\*\* để tính độ lệch phong độ một cách trung thực nhất!**



**#### Đặc trưng 1: `weighted\_final\_score` (Điểm đánh giá có trọng số theo chuẩn 4-6 / 5-5)**

**\* \*\*Ý nghĩa:\*\* Thay vì tính trung bình cộng cào bằng ($\\text{mean}$), ta tính điểm tích lũy có trọng số dựa trên tầm quan trọng (`weight`) của từng bài kiểm tra quá trình kết hợp với hệ số môn học (môn kỹ thuật/chuyên ngành áp dụng tỷ lệ `50-50`, môn đại cương/xã hội áp dụng tỷ lệ `40-60`).**

**\* \*\*Công thức tính toán:\*\***

&#x20; **$$\\text{weighted\\\_process\\\_score} = \\frac{\\sum\_{i \\in \\text{Process}} (\\text{score}\_{10, i} \\times \\text{weight}\_i)}{\\sum\_{i \\in \\text{Process}} \\text{weight}\_i}$$**

&#x20; **Kết hợp với tỷ lệ chuyên cần tương tác trên hệ thống VLE ($\\text{vle\\\_engagement\\\_score} \\in \[0, 10]$ quy đổi từ phân vị `active\_days`), ta tạo biến tổng hợp theo chuẩn trọng số môn học:**

&#x20; **\* Với môn chuyên ngành/đồ án (tỷ lệ `5-5`): $\\text{weighted\\\_final\\\_score} = 0.5 \\times \\text{weighted\\\_process\\\_score} + 0.5 \\times \\text{late\\\_assessment\\\_score}$.**

&#x20; **\* Với môn lý thuyết chuẩn (tỷ lệ `4-6`): $\\text{weighted\\\_final\\\_score} = 0.4 \\times \\text{early\\\_assessment\\\_score} + 0.6 \\times \\text{late\\\_assessment\\\_score}$.**



**#### Đặc trưng 2: `gpa\_credits` (Chỉ số tải trọng học tập \& rủi ro tích lũy theo tín chỉ)**

**\* \*\*Ý nghĩa:\*\* Phản ánh áp lực tín chỉ thực tế của sinh viên. Sinh viên đăng ký nhiều tín chỉ (`studied\_credits` cao, ví dụ 90 – 120 tín chỉ) kèm theo lịch sử từng nợ/học lại môn (`num\_of\_prev\_attempts > 0`) nhưng điểm quá trình thấp sẽ chịu áp lực tâm lý và đồ án nặng gấp bội, dẫn đến nguy cơ bỏ học rất cao.**

**\* \*\*Công thức tính toán:\*\***

&#x20; **\* Điểm tích lũy quy đổi theo tín chỉ đạt chuẩn:**

&#x20;   **$$\\text{gpa\\\_credits} = \\text{weighted\\\_final\\\_score} \\times \\frac{\\text{studied\\\_credits}}{60.0} \\times \\frac{1}{1 + 0.5 \\times \\text{num\\\_of\\\_prev\\\_attempts}}$$**

&#x20; **\* Đồng thời tạo thêm biến phụ trợ \*\*`credit\_burden\_risk` (Áp lực nợ tín chỉ)\*\*:**

&#x20;   **$$\\text{credit\\\_burden\\\_risk} = \\text{studied\\\_credits} \\times (1 + \\text{num\\\_of\\\_prev\\\_attempts}) \\times \\max(0, 5.0 - \\text{weighted\\\_final\\\_score})$$**



**#### Đặc trưng 3: `gap\_process\_exam` (Độ lệch phong độ Quá trình đầu kỳ vs. Giai đoạn nước rút cuối kỳ)**

**\* \*\*Ý nghĩa:\*\* Phát hiện nhóm sinh viên \*\*chủ quan hoặc đuối sức giai đoạn cuối\*\*. Ví dụ: Nửa đầu học kỳ làm bài tập nhỏ đạt `8.0` điểm, nhưng sang nửa cuối học kỳ (khi vào bài tập lớn/thi sát hạch) điểm tụt xuống dưới `4.0` (hoặc bỏ nộp bài `0.0` điểm).**

**\* \*\*Công thức tính toán:\*\***

&#x20; **$$\\text{gap\\\_process\\\_exam} = \\text{score\\\_early\\\_sem} - \\text{score\\\_late\\\_sem}$$**

&#x20; **\*Trong đó:\***

&#x20; **\* $\\text{score\\\_early\\\_sem}$: Điểm trung bình có trọng số các bài kiểm tra từ ngày `0` đến ngày `100`.**

&#x20; **\* $\\text{score\\\_late\\\_sem}$: Điểm trung bình có trọng số các bài kiểm tra sát kỳ thi cuối kỳ (`date > 100`).**

&#x20; **\* Nếu $\\text{gap\\\_process\\\_exam} \\ge 3.0$ (ví dụ đầu kỳ `8.0`, cuối kỳ tụt còn `4.0` $\\rightarrow \\text{gap} = +4.0$): Sinh viên đang rơi tự do về mặt phong độ $\\rightarrow$ Dấu hiệu cảnh báo đỏ cực mạnh!**



**---**



**## 3. PHÂN CHIA CÔNG VIỆC CHI TIẾT THÀNH CÁC TASK NHỎ (WORK BREAKDOWN STRUCTURE - WBS)**



**Toàn bộ quá trình xây dựng mô hình được chia thành \*\*4 Epic (tương ứng với 4 file Notebook)\*\* và \*\*14 Task kỹ thuật nhỏ\*\*. Bạn chỉ cần làm lần lượt từng Task theo đúng danh sách kiểm tra (Checklist) dưới đây:**



**---**



**### 📦 EPIC 1: DATA ENGINEERING \& FEATURE ENGINEERING (`01\_data\_merging\_and\_fe.ipynb`)**



**#### 🔹 Task 1.1: Kiểm tra toàn vẹn khóa chính \& Làm sạch bảng OULAD gốc**

**\* \*\*Input:\*\* `studentInfo.csv`, `assessments.csv`, `studentAssessment.csv`, `studentVle.csv`.**

**\* \*\*Xử lý chi tiết:\*\***

&#x20; **1. Kiểm tra trùng lặp bộ 3 khóa `\['id\_student', 'code\_module', 'code\_presentation']` trên bảng `studentInfo.csv` bằng `df\_info.duplicated(subset=KEYS).sum()`.**

&#x20; **2. Trên bảng `studentAssessment.csv`, lọc bỏ các dòng bị khuyết điểm số (`score.isna()`) hoặc điền `0.0` đối với các bài không nộp, sau đó chia cột `score` cho `10.0` để đưa về thang điểm 10.**

&#x20; **3. Ghép (`merge`) bảng `studentAssessment` với bảng `assessments` qua khóa `id\_assessment` để lấy cột `code\_module`, `code\_presentation`, `assessment\_type`, `date`, và `weight`.**

**\* \*\*Tiêu chí nghiệm thu (Acceptance Criteria):\*\* Không có dòng nào bị nhân bản sai lệch; điểm số nằm hoàn toàn trong khoảng `\[0.0, 10.0]`.**



**#### 🔹 Task 1.2: Gom nhóm (`Groupby`) hành vi tương tác VLE chống tràn RAM**

**\* \*\*Input:\*\* `studentVle.csv` (>10 triệu dòng).**

**\* \*\*Xử lý chi tiết:\*\***

&#x20; **1. Chỉ nạp 5 cột cần thiết bằng tham số `usecols=\['id\_student', 'code\_module', 'code\_presentation', 'date', 'sum\_click']`.**

&#x20; **2. Tách tương tác VLE thành 2 giai đoạn để đo sự sụt giảm chuyên cần:**

&#x20;    **\* `clicks\_early`: Tổng số click khi `date <= 100`.**

&#x20;    **\* `clicks\_late`: Tổng số click khi `date > 100`.**

&#x20; **3. Thực hiện `groupby(KEYS)` để tính:**

&#x20;    **\* `total\_clicks`: Tổng số lượt click toàn kỳ.**

&#x20;    **\* `active\_days`: Số ngày thực tế có đăng nhập hệ thống (`nunique` của cột `date`).**

&#x20;    **\* `attendance\_rate`: Quy đổi `active\_days` sang tỷ lệ phần trăm chuyên cần `\[0%, 100%]` dựa trên phân vị thứ 95 của từng mã môn (`code\_module`).**

&#x20;    **\* `vle\_drop\_ratio`: Tỷ lệ tụt giảm tương tác giai đoạn cuối kỳ so với đầu kỳ.**

**\* \*\*Tiêu chí nghiệm thu:\*\* Bảng kết quả `agg\_vle` thu gọn từ 10 triệu dòng về đúng số lượng sinh viên duy nhất (\~32.593 dòng), thời gian chạy dưới 15 giây, không tràn RAM.**



**#### 🔹 Task 1.3: Thiết lập 3 đặc trưng Tín chỉ Việt Nam \& Gộp bảng Master Tầng 1**

**\* \*\*Input:\*\* Bảng `df\_info`, `df\_ass\_full` (từ Task 1.1) và `agg\_vle` (từ Task 1.2).**

**\* \*\*Xử lý chi tiết:\*\***

&#x20; **1. Tính điểm quá trình nửa đầu kỳ (`score\_early\_sem`), nửa cuối kỳ (`score\_late\_sem`) và tổng số bài đã nộp (`submitted\_count`).**

&#x20; **2. Áp dụng công thức tại Mục 2.2 để tạo 3 cột mới:**

&#x20;    **\* `weighted\_final\_score` (Điểm tổng kết trọng số 4-6 / 5-5 thang 10).**

&#x20;    **\* `gpa\_credits` (Điểm tích lũy điều chỉnh theo số tín chỉ `studied\_credits` và số lần học lại `num\_of\_prev\_attempts`).**

&#x20;    **\* `gap\_process\_exam` (Độ lệch điểm giai đoạn đầu kỳ vs. giai đoạn cuối kỳ: `score\_early\_sem - score\_late\_sem`).**

&#x20; **3. Thực hiện `LEFT JOIN` ghép toàn bộ đặc trưng điểm và đặc trưng VLE vào `studentInfo.csv`.**

&#x20; **4. Điền giá trị khuyết (`fillna(0)` cho các cột điểm/số bài nộp của sinh viên bỏ học sớm; điền `median` theo `code\_module` cho các cột nhân khẩu học nếu khuyết).**

&#x20; **5. Tạo nhãn mục tiêu `is\_dropout` theo Phương án A (`Withdrawn` hoặc `Fail` = `1`, `Pass` hoặc `Distinction` = `0`).**

**\* \*\*Output:\*\* Lưu file `data/processed/stage1\_oulad\_master.csv`.**



**#### 🔹 Task 1.4: Tiền xử lý \& Đồng bộ hóa Thang điểm 10 cho Bộ dữ liệu Tầng 2**

**\* \*\*Input:\*\* `data/raw/kaggle/StudentPerformanceFactors.csv` (6.607 dòng).**

**\* \*\*Xử lý chi tiết:\*\***

&#x20; **1. Loại bỏ các dòng khuyết thiếu (`dropna()`) hoặc điền trung vị.**

&#x20; **2. Chuyển đổi `Previous\_Scores` và `Exam\_Score` về thang điểm 10 (`chia cho 10.0`, làm tròn 2 chữ số thập phân).**

&#x20; **3. Tạo thêm biến tương tác phục vụ Hồi quy: `effective\_study\_index = Hours\_Studied \* (Attendance / 100.0)` và kiểm tra giới hạn hợp lý của `Sleep\_Hours` (`4` đến `10` giờ/đêm).**

**\* \*\*Output:\*\* Lưu file `data/processed/stage2\_performance\_master.csv`.**



**---**



**### 📊 EPIC 2: PHÂN TÍCH KHÁM PHÁ (EDA) \& CHIA TẬP TRAIN / DEV / TEST (`02\_eda\_and\_data\_splitting.ipynb`)**



**#### 🔹 Task 2.1: Vẽ 4 Biểu đồ EDA Cốt lõi \& Kiểm chứng Hiệu quả của 3 Biến Mới**

**\* \*\*Xử lý chi tiết:\*\***

&#x20; **1. \*\*Biểu đồ 1 (Tầng 1):\*\* Biểu đồ tròn/cột phân bố nhãn `is\_dropout` (Phương án A) để kiểm tra tỷ lệ lớp `1` vs `0`.**

&#x20; **2. \*\*Biểu đồ 2 (Tầng 1 - Kiểm chứng Feature Engineering):\*\* Vẽ Boxplot / Violin Plot so sánh 3 biến mới (`weighted\_final\_score`, `gpa\_credits`, `gap\_process\_exam`) giữa nhóm `is\_dropout = 1` và `is\_dropout = 0`. Chứng minh trực quan rằng nhóm bỏ học có `gap\_process\_exam` dương cao (đầu kỳ cao, cuối kỳ tụt dốc).**

&#x20; **3. \*\*Biểu đồ 3 (Tầng 2):\*\* Heatmap ma trận tương quan giữa `Hours\_Studied`, `Sleep\_Hours`, `Attendance`, `Previous\_Scores` (thang 10) và `Exam\_Score` (thang 10).**

&#x20; **4. \*\*Biểu đồ 4 (Tầng 2):\*\* Scatter Plot phân tích mối quan hệ giữa Giờ học và Điểm thi (thang 10, đường ngưỡng qua môn tại `6.0` điểm) theo 3 nhóm giấc ngủ.**

**\* \*\*Output:\*\* Lưu 4 ảnh chất lượng cao (300 DPI) vào `reports/figures/`.**



**#### 🔹 Task 2.2: Chia tập `Train (70%) / Dev (15%) / Test (15%)` cho Tầng 1 và Tầng 2**

**\* \*\*Quy trình chia 2 bước bằng `train\_test\_split` (với `random\_state = 42` cố định):\*\***

&#x20; **1. \*\*Với Tầng 1 (Phân loại - Bắt buộc dùng `stratify=y` để giữ nguyên tỷ lệ nhãn `is\_dropout` ở cả 3 tập):\*\***

&#x20;    **\* Bước 1: Tách `70% Train` và `30% Temp`:**

&#x20;      **`X\_train, X\_temp, y\_train, y\_temp = train\_test\_split(X, y, test\_size=0.30, random\_state=42, stratify=y)`**

&#x20;    **\* Bước 2: Chia đôi `30% Temp` thành `15% Dev` và `15% Test`:**

&#x20;      **`X\_dev, X\_test, y\_dev, y\_test = train\_test\_split(X\_temp, y\_temp, test\_size=0.50, random\_state=42, stratify=y\_temp)`**

&#x20; **2. \*\*Với Tầng 2 (Hồi quy):\*\* Thực hiện chia 2 bước tương tự theo đúng tỷ lệ `70% Train - 15% Dev - 15% Test` (có thể phân tầng theo nhóm điểm Đậu/Rớt để phân phối điểm đồng đều ở cả 3 tập).**

**\* \*\*Output:\*\* Xuất 6 file cố định vào thư mục `data/splits/` (`stage1\_train.csv`, `stage1\_dev.csv`, `stage1\_test.csv`, `stage2\_train.csv`, `stage2\_dev.csv`, `stage2\_test.csv`).**



**---**



**### 🚨 EPIC 3: HUẤN LUYỆN TẦNG 1 \& PHÂN TÍCH LỖI CHUYÊN SÂU (`03\_stage1\_training\_and\_error\_analysis.ipynb`)**



**#### 🔹 Task 3.1: Thiết lập Pipeline Tiền xử lý \& Cân bằng nhãn (SMOTE) trên tập Train (70%)**

**\* \*\*Danh sách đặc trưng đầu vào Tầng 1 (`FEATURES\_T1`):\*\***

&#x20; **1. `weighted\_final\_score` (Điểm quá trình có trọng số thang 10)**

&#x20; **2. `gpa\_credits` (Điểm tích lũy theo tải trọng tín chỉ)**

&#x20; **3. `gap\_process\_exam` (Độ lệch điểm đầu kỳ - cuối kỳ)**

&#x20; **4. `submitted\_count` (Số bài kiểm tra đã nộp)**

&#x20; **5. `attendance\_rate` (Tỷ lệ chuyên cần quy đổi từ số ngày hoạt động VLE %)**

&#x20; **6. `total\_clicks` (Tổng lượt tương tác học liệu VLE)**

&#x20; **7. `num\_of\_prev\_attempts` (Số lần học lại môn này)**

&#x20; **8. `studied\_credits` (Số tín chỉ đang đăng ký)**

**\* \*\*Nguyên tắc chống rò rỉ dữ liệu:\*\***

&#x20; **\* Bộ chuẩn hóa `StandardScaler` và thuật toán `SMOTE` (từ `imblearn.pipeline.Pipeline`) \*\*chỉ được phép `fit\_resample` trên `stage1\_train.csv` (70%)\*\*. Tập `Dev (15%)` và `Test (15%)` giữ nguyên phân phối thực tế.**



**#### 🔹 Task 3.2: Huấn luyện, Tinh chỉnh Siêu tham số \& Tối ưu Ngưỡng cắt trên tập Dev (15%)**

**\* \*\*So sánh 3 mô hình:\*\***

&#x20; **1. `LogisticRegression(max\_iter=1000, random\_state=42)` (Baseline).**

&#x20; **2. `RandomForestClassifier(n\_estimators=300, max\_depth=10, random\_state=42)`.**

&#x20; **3. `XGBClassifier(n\_estimators=300, max\_depth=6, learning\_rate=0.05, subsample=0.8, colsample\_bytree=0.8, random\_state=42)`.**

**\* \*\*Tối ưu ngưỡng cảnh báo (Threshold Tuning trên tập Dev):\*\***

&#x20; **\* Dùng xác suất dự báo $\\hat{p} = \\text{model.predict\\\_proba}(X\_{\\text{dev}})\[:, 1]$ quét qua các ngưỡng từ `0.25` đến `0.60` (với mặc định ưu tiên là `0.40`) để chọn ngưỡng đạt \*\*Recall $\\ge 0.85$\*\* đồng thời tối đa hóa \*\*F1-Score\*\* trên tập Dev.**



**#### 🔹 Task 3.3: Phân tích Lỗi Chuyên sâu Tầng 1 trên tập Dev (Error Analysis: FN \& FP)**

**Đây là Task trọng tâm giúp bạn viết phần bàn luận cực sâu trong báo cáo:**

**1. \*\*Lập bảng đối chiếu và xuất danh sách lỗi:\*\***

&#x20;  **\* Lọc toàn bộ các ca \*\*False Negative (FN)\*\* trên tập Dev (`y\_dev == 1` nhưng `y\_pred\_dev == 0`): Đây là những sinh viên thực tế bị rớt/bỏ học nhưng hệ thống lại đánh giá là "An toàn". Xuất ra file `reports/error\_logs/stage1\_dev\_false\_negatives.csv`.**

&#x20;  **\* Lọc toàn bộ các ca \*\*False Positive (FP)\*\* trên tập Dev (`y\_dev == 0` nhưng `y\_pred\_dev == 1`): Đây là những sinh viên thực tế qua môn nhưng bị hệ thống báo động nhầm. Xuất ra file `reports/error\_logs/stage1\_dev\_false\_positives.csv`.**

**2. \*\*Phân tích định lượng theo Nhóm Sinh viên (Cohort Error Analysis):\*\***

&#x20;  **\* So sánh giá trị trung bình ($\\text{Mean} \\pm \\text{SD}$) của 4 nhóm: `True Positive (TP)`, `True Negative (TN)`, `False Negative (FN)`, `False Positive (FP)`.**

&#x20;  **\* \*\*Kiểm chứng 2 giả thuyết lỗi điển hình trên tập Dev:\*\***

&#x20;    **\* \*Nhóm lỗi FN (Bỏ sót nguy cơ):\* Thường rơi vào nhóm sinh viên có \*\*điểm quá trình đầu kỳ khá cao (`score\_early\_sem >= 6.5`), nộp bài đầu kỳ đầy đủ, nhưng đột ngột ngừng truy cập VLE ở nửa cuối học kỳ\*\* hoặc bỏ thi cuối kỳ vì lý do cá nhân đột xuất.**

&#x20;    **\* \*Nhóm lỗi FP (Cảnh báo nhầm):\* Thường rơi vào nhóm sinh viên \*\*"Học nước rút" (Crammers)\*\* — cả kỳ ít click VLE (`total\_clicks` thấp) khiến mô hình tưởng sắp bỏ học, nhưng sát ngày thi lại làm bài đạt điểm cao nên vẫn qua môn (`Pass`).**



**#### 🔹 Task 3.4: Đánh giá Chốt hạ trên tập Test (15%) \& Xuất Pipeline `.pkl` Tầng 1**

**\* Đánh giá mô hình tốt nhất với ngưỡng tối ưu một lần duy nhất trên `stage1\_test.csv` (15%) qua bộ chỉ số: `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`.**

**\* Đóng gói `StandardScaler` + `Best Classifier` vào 1 đối tượng `Pipeline` duy nhất và lưu thành `models/pipeline\_stage1\_dropout.pkl`.**



**---**



**### 🚀 EPIC 4: HUẤN LUYỆN TẦNG 2, PHÂN TÍCH OUTLIERS \& BỘ GIẢ LẬP WHAT-IF (`04\_stage2\_training\_and\_whatif\_sim.ipynb`)**



**#### 🔹 Task 4.1: Huấn luyện 3 Mô hình Hồi quy (Thang điểm 10) trên tập Train (70%)**

**\* \*\*Danh sách đặc trưng đầu vào Tầng 2 (`FEATURES\_T2`):\*\***

&#x20; **`\['Hours\_Studied', 'Sleep\_Hours', 'Attendance', 'Previous\_Scores', 'Tutoring\_Sessions']` (trong đó `Previous\_Scores` và biến mục tiêu `Exam\_Score` đã ở \*\*thang điểm 10\*\*).**

**\* \*\*Huấn luyện Pipeline (`StandardScaler` + `Regressor`) với 3 thuật toán:\*\***

&#x20; **1. `LinearRegression()`.**

&#x20; **2. `Ridge(alpha=1.0)`.**

&#x20; **3. `RandomForestRegressor(n\_estimators=300, max\_depth=12, random\_state=42)`.**

**\* \*\*Đánh giá trên tập Dev (15%):\*\* So sánh qua `RMSE` (thang 10), `MAE` (thang 10) và $R^2$ Score.**



**#### 🔹 Task 4.2: Phân tích Lỗi Tầng 2 trên tập Dev (Residual Plot \& Top 20 Outliers)**

**1. \*\*Tính toán Phần dư (Residuals) trên tập Dev:\*\***

&#x20;  **$$e\_i = y\_{\\text{dev}, i} - \\hat{y}\_{\\text{dev}, i}, \\quad \\vert{}e\_i\\vert{} = \\vert{}y\_{\\text{dev}, i} - \\hat{y}\_{\\text{dev}, i}\\vert{}$$**

**2. \*\*Vẽ 2 biểu đồ chẩn đoán lỗi Hồi quy:\*\***

&#x20;  **\* \*\*Biểu đồ 1 — Predicted vs. Actual Plot:\*\* Trục hoành là điểm dự báo $\\hat{y}$, trục tung là điểm thực tế $y$, kèm đường chéo lý tưởng $y = \\hat{y}$ và vùng sai số chấp nhận được $\\pm 0.5$ điểm (thang 10).**

&#x20;  **\* \*\*Biểu đồ 2 — Residual Plot \& Error Distribution:\*\* Trục hoành là $\\hat{y}\_{\\text{dev}}$, trục tung là phần dư $e\_i$ (kiểm tra xem sai số có bị lệch hệ thống — Heteroscedasticity — ở vùng điểm quá cao `>9.0` hoặc quá thấp `<4.5` hay không).**

**3. \*\*Trích xuất và mổ xẻ Top 20 Outliers trên tập Dev:\*\***

&#x20;  **\* Sắp xếp tập Dev theo độ lệch tuyệt đối $\\vert{}e\_i\\vert{}$ giảm dần, lấy ra \*\*20 bản ghi có sai số dự báo lớn nhất\*\* (`top20\_outliers = df\_dev.nlargest(20, 'abs\_error')`) và lưu vào `reports/error\_logs/stage2\_dev\_top20\_outliers.csv`.**

&#x20;  **\* Phân tích nguyên nhân vì sao mô hình đoán lệch trên 20 sinh viên này (ví dụ: sinh viên học rất ít giờ `Hours\_Studied` thấp, đi học vắng nhiều nhưng điểm `Exam\_Score` thực tế lại vọt lên `9.5 - 10.0` do yếu tố ngoại lai/năng khiếu đặc biệt).**



**#### 🔹 Task 4.3: Xây dựng Thuật toán Mô phỏng `What-If Simulator` (Chuẩn Thang điểm 10)**

**\* \*\*Ngưỡng điểm quy ước trên thang 10:\*\***

&#x20; **\* Điểm qua môn tối thiểu (`PASS\_SCORE`): \*\*`6.0` điểm\*\*.**

&#x20; **\* Điểm mục tiêu an toàn (`TARGET\_SAFE\_SCORE`): \*\*`6.5` điểm\*\* (mặc định trong hàm khuyến nghị).**

**\* \*\*Logic thuật toán `recommend\_study\_plan(current\_profile, pipeline\_reg, target\_score=6.5)`:\*\***

&#x20; **1. Nhận hồ sơ sinh viên bị cảnh báo ở Tầng 1 (chuyển `weighted\_final\_score` thang 10 của Tầng 1 sang làm `Previous\_Scores` thang 10 của Tầng 2, giữ nguyên `attendance\_rate` làm `Attendance`).**

&#x20; **2. Dự báo điểm thi hiện tại `current\_pred` (thang 10).**

&#x20; **3. Quét vòng lặp lưới (Grid Search):**

&#x20;    **\* Tăng `Hours\_Studied` từ mức hiện tại lên tối đa `40` giờ/tuần (bước nhảy `+2` giờ).**

&#x20;    **\* Tăng `Attendance` từ mức hiện tại lên `100%` (bước nhảy `+5%`).**

&#x20;    **\* Tăng `Tutoring\_Sessions` thêm `+0`, `+1`, `+2` buổi/tháng.**

&#x20;    **\* Chuẩn hóa `Sleep\_Hours` về khoảng vàng `7 – 8` giờ/đêm.**

&#x20; **4. Tính hàm chi phí nỗ lực (`effort\_cost`) và trả về \*\*Top 3 lộ trình tốn ít công sức thay đổi nhất\*\* nhưng giúp `expected\_new\_score >= 6.5`.**



**#### 🔹 Task 4.4: Đánh giá trên tập Test (15%) \& Xuất Pipeline `.pkl` Tầng 2**

**\* Chạy đánh giá cuối cùng trên `stage2\_test.csv` (15%).**

**\* Lưu đối tượng `Pipeline` (đã gồm `StandardScaler` + `Best Regressor`) thành `models/pipeline\_stage2\_score.pkl`.**

**\* Xuất file `models/model\_metadata.json` ghi lại danh sách cột đầu vào của 2 tầng và ngưỡng cắt tối ưu để bàn giao cho Nhật Vinh (Project 1 - FastAPI) và Hoàng Anh (Project 2 - Streamlit).**



**---**



**## 4. BẢNG TỔNG HỢP ĐẶC TẢ ĐẦU VÀO / ĐẦU RA CỦA 2 FILE `.PKL` (CONTRACT VỚI PROJECT 1 \& 2)**



**Nhờ việc đóng gói chung `Preprocessor` và `Model` vào cùng 1 đối tượng `Pipeline` (`.pkl`), Nhật Vinh và Hoàng Anh chỉ cần truyền đúng một `DataFrame` gồm các cột dưới đây vào hàm `.predict()` hoặc `.predict\_proba()` mà không cần tự viết lại code chuẩn hóa dữ liệu:**



**| Tên Artifact | Danh sách cột đầu vào bắt buộc (Input Features) | Kiểu đầu ra (Output) | Ngưỡng quy ước |**

**| :--- | :--- | :--- | :--- |**

**| \*\*`pipeline\_stage1\_dropout.pkl`\*\* | `weighted\_final\_score` (0–10), `gpa\_credits` (float), `gap\_process\_exam` (-10 đến +10), `submitted\_count` (int), `attendance\_rate` (0–100), `total\_clicks` (int), `num\_of\_prev\_attempts` (int), `studied\_credits` (int) | Xác suất bỏ học $P(\\text{Dropout}) \\in \[0.0, 1.0]$ qua hàm `.predict\_proba(df)\[:, 1]` | `< 0.40`: An toàn<br>`0.40 - 0.60`: Cảnh báo Vàng<br>`>= 0.60`: Nguy cơ Đỏ |**

**| \*\*`pipeline\_stage2\_score.pkl`\*\* | `Hours\_Studied` (1–44), `Sleep\_Hours` (4–10), `Attendance` (0–100), `Previous\_Scores` (0.0–10.0), `Tutoring\_Sessions` (0–8) | Điểm thi dự kiến thang 10 $\\hat{y} \\in \[0.0, 10.0]$ qua hàm `.predict(df)\[0]` | `< 6.0`: Nguy cơ rớt môn<br>`6.0 - 6.4`: Đậu vớt<br>`>= 6.5`: Đậu an toàn |**

**```eof**



**---**



**### Điểm nhấn trong bản Spec này dành riêng cho Gia Vỹ:**

**1. \*\*Giải quyết triệt để bài toán Data Engineering Việt Nam trên OULAD:\*\* Trong Mục 2.2 và Epic 1, mình đã thiết kế công thức toán học rõ ràng cho cả 3 biến bạn đề xuất (`weighted\_final\_score`, `gpa\_credits`, `gap\_process\_exam`), đồng thời tách điểm quá trình thành nửa đầu kỳ (`date <= 100`) và nửa cuối kỳ (`date > 100`) để biến `gap\_process\_exam` hoạt động chuẩn xác mà không bị rò rỉ dữ liệu thi cuối kỳ.**

**2. \*\*Quy trình `Train (70%) / Dev (15%) / Test (15%)` chuẩn mực:\*\* Mọi thao tác cân bằng dữ liệu `SMOTE` chỉ chạm vào tập `Train (70%)`, toàn bộ việc tìm ngưỡng cắt tối ưu và mổ xẻ lỗi (`FN`, `FP` ở Tầng 1; `Residual Plot`, `Top 20 Outliers` ở Tầng 2) đều diễn ra trên tập `Dev (15%)`, giữ tập `Test (15%)` hoàn toàn "tinh khiết" cho bảng kết quả cuối cùng.**

**3. \*\*Đóng gói chung vào `Pipeline .pkl`:\*\* Bạn xuất ra `pipeline\_stage1\_dropout.pkl` và `pipeline\_stage2\_score.pkl` đã bao gồm cả bước chuẩn hóa `StandardScaler` bên trong, giúp Nhật Vinh và Hoàng Anh gọi hàm cực kỳ an toàn.**



**Bạn xem qua bản Spec `.md` trên đã ưng ý 100% chưa? Nếu bạn muốn mình \*\*viết luôn code Python thực thi cho 4 file Notebook (`01` đến `04`)\*\* theo đúng từng Task trong bản Spec này thì nhắn mình nhé!**

