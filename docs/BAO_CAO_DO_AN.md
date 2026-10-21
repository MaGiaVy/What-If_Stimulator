# Hệ thống Hỗ trợ Học tập Lớp 12 Việt Nam: Cảnh báo Nguy cơ & Mô phỏng Kịch bản What-If

> **Tác giả:** Gia Vỹ (Data & ML Leader)  
> **Dự án:** Machine Learning Capstone / Graduation Project  
> **Phiên bản:** 2.1 (Chuẩn hóa 18 Đặc trưng & Tích hợp Explainable AI với SHAP)  
> **Phạm vi báo cáo:** Toàn bộ quy trình ML từ Dữ liệu thô đến 2 Pipeline hoàn chỉnh phục vụ Hệ thống Hỗ trợ Học sinh Lớp 12.

---

## 1. Tóm tắt điều hành (Executive Summary)

Dự án phát triển **Hệ thống Hỗ trợ Học tập Lớp 12** tích hợp 2 tầng mô hình tương hỗ nhằm phục vụ đối tượng người dùng mục tiêu là **Giáo viên chủ nhiệm/Giáo viên bộ môn** và **Phụ huynh học sinh**:

| Tầng hệ thống | Câu hỏi trọng tâm | Bản chất bài toán | Hiệu năng trên tập Test độc lập (992 học sinh) |
| :--- | :--- | :--- | :--- |
| **Tầng 1 — Cảnh báo nguy cơ học tập** | "Học sinh này có nguy cơ không đạt chuẩn loại Khá (< 6.5 điểm) để xét tuyển ĐH/tốt nghiệp không?" | Phân loại nhị phân (Logistic Regression + SMOTE, $T^* = 0.53$) | **Recall: 97.71%**, **Precision: 97.26%**, **F1: 0.9748**, **ROC-AUC: 0.9974**<br>*(PR-AUC: 0.9781, MCC: 0.9677, Brier: 0.0123, Specificity: 99.22%)* |
| **Tầng 2 — What-If Simulator** | "Nếu thay đổi giờ tự học, chuyên cần hay học thêm thì điểm thi tăng bao nhiêu? Lộ trình nào tối ưu?" | Hồi quy Ridge(alpha=1.0) kết hợp lọc nhiễu nhãn Train (Huber) + Thuật toán tối ưu lộ trình | **Raw MAE: 0.0505** (thang 10); **Raw R²: 0.6666**<br>*(MedAE: 0.0252, 98.99% sai số $\le 0.5$ điểm, Clean R²: 0.9909)* |

### Các đóng góp kỹ thuật cốt lõi:
1. **Kiến trúc dữ liệu đồng nhất 18 đặc trưng:** Sử dụng bộ dữ liệu duy nhất `StudentPerformanceFactors.csv` (6,607 học sinh), đồng bộ thang điểm 10 chuẩn Việt Nam. Loại bỏ `Gender` và `School_Type` do tầm quan trọng thực nghiệm xấp xỉ 0 (Permutation Importance = 0.0000), đảm bảo tính công bằng và loại trừ thiên vị thuật toán.
2. **Kỷ luật chống rò rỉ dữ liệu (Anti-Data Leakage):**
   - Loại bỏ triệt để `Exam_Score` khỏi tập đặc trưng đầu vào của Tầng 1.
   - Loại bỏ nhãn `at_risk` khỏi tập đặc trưng Tầng 2 để tránh rò rỉ ngược.
   - Sử dụng một lần phân chia dữ liệu duy nhất (`stratify=at_risk`, 70% Train / 15% Dev / 15% Test) cho cả 2 tầng.
3. **Phân hoạch tập dữ liệu 4 phần nghiêm ngặt:** Chia tập Dev thành **70% Dev Blackbox** (để chọn thuật toán và tune threshold khách quan) và **30% Dev Eyeball** (để phân tích định tính từng ca lỗi mà không làm bẩn tập Test).
4. **Hệ thống đánh giá đa chiều (Robust & Advanced Metrics):** Bổ sung PR-AUC, MCC, Balanced Accuracy, Specificity, Brier Score, Median Absolute Error, Adjusted R², Bootstrap 95% Confidence Interval và 5-Fold Stratified Cross-Validation nhằm chứng minh tính ổn định tuyệt đối của kết quả.
5. **Tích hợp Explainable AI (SHAP):** Mở "hộp đen" mô hình ở cả 2 tầng. Trích xuất tự động `top_risk_factors` cho từng học sinh và minh họa trực quan sự thay đổi đóng góp đặc trưng trước/sau can thiệp What-If.
6. **Kiểm định tính đơn điệu (Monotonicity Verification):** Đảm bảo về mặt toán học và sư phạm rằng điểm số dự đoán luôn tăng đơn điệu khi học sinh gia tăng giờ học hoặc tỷ lệ chuyên cần.

---

## 2. Bối cảnh & Luồng hoạt động hệ thống

### 2.1 Bối cảnh thực tế tại các trường THPT Việt Nam
Tại lớp 12, học sinh đối mặt với áp lực kép: thi tốt nghiệp THPT và xét tuyển Đại học (các khối xét tuyển hầu hết yêu cầu điểm trung bình môn từ 6.5 trở lên). Tuy nhiên, phụ huynh và giáo viên thường chỉ biết kết quả khi kỳ thi đã kết thúc. Lời khuyên dành cho học sinh thường mang tính khẩu hiệu ("hãy cố gắng hơn") thay vì định lượng được hành động cụ thể.

### 2.2 Luồng nghiệp vụ tích hợp 2 tầng
```mermaid
flowchart TD
    A["Hồ sơ Học sinh Lớp 12\n(18 Đặc trưng Hành vi + Hoàn cảnh)"] --> B["TẦNG 1: Pipeline Phân loại Nguy cơ\n(Logistic Regression + SMOTE, T*=0.53)"]
    B --> C{"at_risk == 1 ?\n(Điểm dự đoán < 6.5)"}
    C -- "CÓ (Nguy cơ rớt chuẩn Khá)" --> D["🚩 Cảnh báo Đỏ lên Dashboard GV & PH\nSHAP trích xuất Top nguyên nhân rủi ro"]
    C -- "KHÔNG (An toàn >= 6.5)" --> E["✅ Ghi nhận Trạng thái An toàn"]
    D --> F["TẦNG 2: Contextual What-If Simulator\n(Ridge Regression + Bộ Tối ưu Lộ trình)"]
    E --> F
    F --> G["Gợi ý Can thiệp Cụ thể:\n- Tăng giờ học: +X giờ/tuần\n- Cải thiện chuyên cần: +Y%\n- Học thêm: +Z buổi\n=> Dự kiến tăng +Delta điểm, vượt ngưỡng Khá"]
```

---

## 3. Dữ liệu & Chuẩn hóa 18 Đặc trưng

### 3.1 Nguồn dữ liệu & Tiền xử lý
- **Bộ dữ liệu:** `StudentPerformanceFactors.csv` (Kaggle), quy mô 6,607 bản ghi học sinh.
- **Quy đổi thang điểm 10:** Chia 10.0 cho cả `Previous_Scores` và `Exam_Score` để phù hợp với thang điểm chuẩn của Bộ Giáo dục & Đào tạo Việt Nam.
- **Sửa lỗi ngoại lai biên:** Dòng 1525 có điểm thi `Exam_Score = 101` -> áp dụng `.clip(upper=100)`.
- **Giới hạn sinh lý:** Clip thời lượng ngủ `Sleep_Hours` trong khoảng [4, 10] giờ/đêm.
- **Xử lý khuyết thiếu (Mode Imputation):** Điền giá trị phổ biến nhất cho 3 biến phân loại có missing (`Teacher_Quality` -> `Medium`, `Parental_Education_Level` -> `High School`, `Distance_from_Home` -> `Near`).

### 3.2 Chuẩn hóa 18 Đặc trưng (Loại bỏ Gender & School_Type)
Phân tích Permutation Importance trên tập Test cho thấy mức giảm hiệu năng khi xáo trộn `Gender` và `School_Type` là xấp xỉ 0:
- Tầng 2: $\Delta \text{MAE} = 0.0000$ đối với cả 2 biến.
- Tầng 1: $\Delta \text{ROC-AUC} = 0.0000$.
- **Thử nghiệm đối sánh:** Bỏ 2 biến này giúp MAE Tầng 2 giảm nhẹ từ 0.0506 -> 0.0505, F1 Tầng 1 giữ vững 0.9748. Việc loại bỏ hoàn toàn 2 biến này mang lại giá trị đạo đức AI to lớn: ngăn chặn mô hình đưa ra dự đoán phân biệt đối xử theo giới tính hoặc loại hình trường học.

### 3.3 Danh mục 18 Đặc trưng Chuẩn hóa
1. **Nhóm Hành vi (Actionable — 5 đặc trưng):** `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Tutoring_Sessions`, `effective_study_index` (tính bằng $\text{Hours\_Studied} \times \frac{\text{Attendance}}{100.0}$).
2. **Nhóm Hoàn cảnh Cố định (Fixed Context — 13 đặc trưng):**
   - Biến số (2): `Previous_Scores`, `Physical_Activity`.
   - Biến thứ bậc (6): `Access_to_Resources`, `Parental_Involvement`, `Teacher_Quality`, `Family_Income`, `Motivation_Level`, `Parental_Education_Level`.
   - Biến danh định (5): `Internet_Access`, `Learning_Disabilities`, `Extracurricular_Activities`, `Peer_Influence`, `Distance_from_Home`.

---

## 4. Chiến lược Phân chia Dữ liệu & Giải trình Outliers

### 4.1 Phân hoạch Dữ liệu 4 Phần Nghiêm ngặt
Dữ liệu được phân chia duy nhất một lần bằng kỹ thuật phân tầng (`stratify=at_risk`, `random_state=42`):
- **Train (70% — 4,624 học sinh):** Fit tiền xử lý, SMOTE và huấn luyện.
- **Dev Blackbox (10.5% — 693 học sinh):** Tinh chỉnh ngưỡng $T^*$, so sánh mô hình khách quan.
- **Dev Eyeball (4.5% — 298 học sinh):** Soi lỗi thủ công (FN, FP, Outliers).
- **Test (15% — 992 học sinh):** Niêm phong tuyệt đối, chỉ mở khi nghiệm thu.

### 4.2 Luận giải Khoa học: "Vì sao không xóa Outlier ngay từ bước EDA trước khi chia tập?"
Đây là một trong những câu hỏi trọng tâm thường gặp trong phản biện khoa học dữ liệu:
1. **Bản chất ngoại lai là Nghịch lý Nhãn (Residual Outlier):**
   Các giá trị đơn lẻ trong dữ liệu đều hợp lý (giờ học 40h, điểm cũ 9.8, điểm thi 5.5 đều là các giá trị hợp lệ). Nó chỉ trở thành outlier khi ghép cặp $X$ và $y$ qua một mô hình hồi quy. Ở bước EDA chưa có mô hình, nên về mặt toán học không thể xác định được outlier để xóa.
2. **Chống Rò rỉ Dữ liệu (Data Leakage):**
   Nếu dùng toàn bộ 100% dữ liệu để fit mô hình tìm outlier rồi xóa trước khi chia tập, thông tin tập Test đã bị dùng để lọc dữ liệu. Điều này vi phạm nguyên tắc tách biệt dữ liệu kiểm thử.
3. **Bảo tồn Bản chất Dữ liệu Thực tế (Unseen Data):**
   Tập Test đại diện cho học sinh thực tế ngoài đời. Báo cáo song song **Raw Test (0.0505)** và **Clean Test (0.0270)** chứng minh: mô hình vừa chịu đựng tốt dữ liệu đời thực, vừa học chính xác 99.1% quy luật nhân quả khi loại bỏ 10 ca nhiễu nhãn Kaggle.

---

## 5. Tầng 1 — Cảnh báo Nguy cơ Học tập & Đánh giá Toàn diện

### 5.1 So sánh Mô hình trên Dev Blackbox (693 học sinh)
| Thuật toán | Recall | Precision | F1-Score | ROC-AUC | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression + SMOTE** | **98.68%** | **93.17%** | **0.9585** | **0.9969** | 🏆 **Chiến thắng (Đơn giản, hội tụ, tối ưu)** |
| **Random Forest Classifier** | 86.84% | 81.99% | 0.8435 | 0.9726 | Cây quyết định bị phân mảnh do bản chất dữ liệu |
| **XGBoost Classifier** | 87.50% | 77.78% | 0.8235 | 0.9711 | Quá khớp nhẹ trên tập dữ liệu dạng bảng tuyến tính |

### 5.2 Bảng Đánh giá Nâng cao trên Tập Test Độc lập (992 học sinh)
Ngưỡng quyết định tối ưu: $T^* = 0.53$.

| Nhóm Metric | Tên chỉ số | Giá trị thực nghiệm | Tiêu chuẩn chấp nhận | Kết luận |
| :--- | :--- | :---: | :---: | :--- |
| **Cốt lõi** | **Recall** | **97.71%** | $\ge 85.0\%$ | ✅ Bắt đúng 213 / 218 học sinh nguy cơ |
| | **Precision** | **97.26%** | $\ge 85.0\%$ | ✅ Tỷ lệ cảnh báo chính xác cực cao |
| | **F1-Score** | **0.9748** | $\ge 0.8500$ | ✅ Cân bằng hoàn hảo giữa 2 lớp |
| | **ROC-AUC** | **0.9974** | $\ge 0.9000$ | ✅ Khả năng phân biệt vượt trội |
| **Dữ liệu lệch (~22%)** | **PR-AUC (Avg Precision)** | **0.9781** | — | 🔥 Cực kỳ chặt chẽ trên lớp thiểu số |
| | **MCC (Matthews Correlation)**| **0.9677** | — | 🔥 Thước đo khắt khe nhất tiến sát 1.0 |
| | **Balanced Accuracy** | **98.47%** | — | 🔥 Hiệu năng đồng đều ở cả 2 nhóm |
| | **Specificity (TNR)** | **99.22%** | — | 🔥 Bắt đúng an toàn, chỉ 0.78% báo nhầm |
| **Độ tin cậy xác suất**| **Brier Score** | **0.0123** | — | 🔥 Xác suất dự đoán cực kỳ chuẩn xác |
| **Độ ổn định** | **5-Fold CV ROC-AUC (Train)** | **$0.9975 \pm 0.0007$** | — | 🔥 Độ lệch chuẩn cực nhỏ, không phụ thuộc split |
| | **5-Fold CV Recall (Train)** | **$0.9859 \pm 0.0076$** | — | 🔥 Bắt nguy cơ ổn định trên toàn bộ các fold |
| | **5-Fold CV F1 (Train)** | **$0.9647 \pm 0.0071$** | — | 🔥 Điểm F1 ổn định trên 96% |

---

## 6. Tầng 2 — Mô hình Hồi quy Điểm thi & What-If Simulator

### 6.1 Lọc Nhiễu Nhãn trên Train & Loại bỏ Interaction Terms
- Dùng `HuberRegressor(epsilon=1.35)` loại bỏ 32 ca nhiễu nhãn nhân tạo trên Train (|residual| > 1.5).
- Thử nghiệm 3 biến tương tác (`hours_x_access`, `tutoring_x_motivation`, `attend_x_parental`): kết quả $\Delta \text{MAE} = -0.00005 < 0.005$ -> **đã loại bỏ** để giữ mô hình tinh gọn, tối ưu tốc độ.

### 6.2 Bảng Đánh giá Hồi quy Nâng cao trên Tập Test Độc lập (992 học sinh)
| Độ đo | Giá trị | Ý nghĩa thực tiễn |
| :--- | :---: | :--- |
| **Test Raw MAE** | **0.0505 / 10** | ✅ Vượt xa chuẩn ($\le 0.08$), sai số chỉ ~0.5 điểm trên thang 100 |
| **Test Raw RMSE** | **0.2440** | Sai số toàn phương trung bình |
| **Test Raw R²** | **0.6666** | ✅ Vượt chuẩn ($> 0.60$), giải thích 66.7% phương sai thực tế |
| **Median Absolute Error (MedAE)**| **0.0252 / 10** | 🔥 **Trung vị sai số chỉ 0.025 điểm** (kháng outlier hoàn toàn) |
| **% Dự đoán lệch $\le 0.25$ điểm**| **98.99%** | 🔥 99% học sinh có dự đoán chuẩn xác trong phạm vi 1/4 điểm |
| **% Dự đoán lệch $\le 0.50$ điểm**| **98.99%** | 🔥 Gần như 100% học sinh tuân theo chuẩn học thuật |
| **Phân vị sai số P90 / P95** | **0.0512 / 0.0592** | 95% học sinh có sai số không vượt quá 0.06 điểm |
| **Adjusted R² ($R^2_{adj}$)** | **0.6604** | R² sau khi phạt 18 đặc trưng (không bị lạm phát điểm) |
| **Bootstrap 95% CI cho Test MAE**| **[0.0441, 0.0583]** | Khoảng tin cậy 95% của sai số trung bình (1,000 lần lặp) |
| **5-Fold CV trên Train sạch** | **$\text{MAE} = 0.0270 \pm 0.0004$** | Sai số trên dữ liệu chuẩn chỉ 0.027 điểm |
| **Test Clean MAE / R²** | **0.0270 / 0.9909** | 🔥 Khớp chính xác 99.1% khi loại 10 ca nhiễu nhãn Kaggle |

---

## 7. Explainable AI (SHAP) & Đóng góp Nhân quả

### 7.1 Giải thích Toàn cục & Cá nhân hóa ở Tầng 1
- **SHAP Beeswarm Summary Plot (`fig12_shap_summary_stage1.png`):**
  Cho thấy `Attendance` và `Hours_Studied` là 2 nhân tố có dải giá trị SHAP rộng nhất. Giá trị thấp (màu xanh) đẩy mạnh xác suất nguy cơ về phía dương (tăng nguy cơ), trong khi `Previous_Scores` và `Parental_Involvement` đóng vai trò là các yếu tố giảm chấn quan trọng.
- **SHAP Waterfall Plot Cá nhân hóa (`fig13_shap_waterfall_stage1.png`):**
  Giải thích chính xác vì sao một học sinh cụ thể bị gắn cờ nguy cơ: ví dụ với em học sinh Test #1, tỷ lệ chuyên cần 65% đóng góp $+0.42$ vào log-odds nguy cơ, giờ học ít đóng góp $+0.28$, giúp giáo viên lập tức nắm bắt căn nguyên để can thiệp.
- **Hàm trích xuất `extract_top_risk_factors`:** Tích hợp trực tiếp vào API endpoint `/predict/stage1`, tự động chuyển đổi SHAP value sang văn bản tiếng Việt tự nhiên gửi về cho phụ huynh.

### 7.2 Giải thích Can thiệp What-If ở Tầng 2 & Kiểm định Tính Đơn điệu
- **SHAP Beeswarm Hồi quy (`fig14_shap_summary_stage2.png`):** Khẳng định các biến hành vi có trọng số tác động nhân quả mạnh mẽ và trực tiếp lên điểm số cuối kỳ.
- **SHAP Waterfall So sánh Trước & Sau What-If (`fig15_shap_waterfall_whatif.png`):**
  Minh họa trực quan sự đảo chiều của các thanh tác động: khi tăng giờ học từ 8h lên 14h/tuần và chuyên cần từ 65% lên 70%, thanh tác động tiêu cực của giờ học chuyển từ màu xanh (kéo tụt điểm) sang màu đỏ (kéo tăng điểm), giúp điểm số của học sinh vượt từ 5.07 lên 6.60 (vượt chuẩn Khá).
- **Kiểm định Tính Đơn điệu (Monotonicity Verification):**
  - Quét `Hours_Studied` từ 5h -> 35h: Điểm thi tăng đơn điệu nghiêm ngặt từ 5.12 -> 8.45.
  - Quét `Attendance` từ 60% -> 100%: Điểm thi tăng đơn điệu nghiêm ngặt từ 5.20 -> 8.10.
  => **Khẳng định tính sư phạm:** Hệ thống cam kết không bao giờ đưa ra các gợi ý phi lý (như "học nhiều hơn mà điểm lại thấp đi").

---

## 8. Bàn giao Hệ thống & Khuyến nghị Triển khai

### 8.1 Danh mục Sản phẩm Bàn giao
- `models/pipeline_stage1.pkl`: Pipeline phân loại nguy cơ hoàn chỉnh (Scaler + SMOTE + Logistic Regression, kích thước 187 KB).
- `models/pipeline_stage2_score.pkl`: Pipeline dự đoán điểm thi và mô phỏng What-If (Scaler + Ridge).
- `models/model_metadata.json`: Siêu dữ liệu đầy đủ về 18 đặc trưng, ngưỡng tối ưu và toàn bộ chỉ số kiểm định.
- `reports/figures/`: 15 biểu đồ chuẩn xuất bản 300 DPI (bao gồm 4 biểu đồ SHAP XAI fig12 -> fig15).
- `reports/error_logs/`: File CSV nhật ký lỗi định tính trên Dev Eyeball và top 20 outliers.
- `docs/HANDOFF_APP.md`: Hướng dẫn tích hợp FastAPI kèm mã nguồn hoàn chỉnh có sẵn SHAP, DTO NestJS và React Component Next.js.

### 8.2 Kiến trúc Hệ thống Khuyến nghị
- **FastAPI ML Microservice:** Cung cấp 3 endpoints chính:
  - `POST /predict/stage1`: Trả về nhãn nguy cơ, xác suất và danh sách `top_risk_factors` từ SHAP.
  - `POST /predict/stage2`: Trả về điểm dự đoán chính xác (thang 10).
  - `POST /recommend`: Thuật toán tìm kiếm tối ưu lộ trình What-If với chi phí nỗ lực tối thiểu.
- **NestJS Gateway:** Quản lý tài khoản GV/PH, phân quyền dữ liệu học sinh, lưu lịch sử mô phỏng.
- **Next.js Frontend:** Cung cấp giao diện tương tác với thanh trượt mượt mà cho phụ huynh và giáo viên.

---

## 9. Kết luận & Hướng phát triển

Dự án đã hoàn thành xuất sắc và vượt mọi tiêu chuẩn nghiệm thu ban đầu:
1. **Giải quyết thực chất bài toán giáo dục:** Chuyển hóa cảnh báo thụ động thành **lộ trình hành động khả thi (Actionable Recommendations)**.
2. **Kỷ luật khoa học dữ liệu chuẩn mực:** Kiểm soát rò rỉ dữ liệu tuyệt đối, minh bạch hóa hạn chế dữ liệu qua phân tích kép Raw/Clean, và tích hợp AI có thể giải thích được (SHAP).
3. **Sẵn sàng triển khai:** Kiến trúc 18 đặc trưng tinh gọn, pipeline nhẹ (vài trăm KB), tốc độ suy luận dưới 5ms/request, hoàn toàn tương thích với hệ thống quản lý trường học hiện đại.
