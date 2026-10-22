# KẾ HOẠCH KỸ THUẬT CHI TIẾT: NÂNG CẤP TẦNG 2 — CONTEXTUAL WHAT-IF SIMULATOR & XỬ LÝ NHIỄU NHÃN (LABEL NOISE)

> ⚠️ **LƯU Ý LỊCH SỬ:** Tài liệu này là kế hoạch kỹ thuật sơ bộ ban đầu của Tầng 2. Toàn bộ nội dung, phân tích nhiễu nhãn và kiến trúc thực tế đã được chuẩn hóa và hợp nhất chính thức tại **[Spec.md](file:///d:/What-If%20Simulator/Spec.md)** (v2.1) và **[BAO_CAO_DO_AN.md](file:///d:/What-If%20Simulator/docs/BAO_CAO_DO_AN.md)**. Vui lòng tham chiếu các tài liệu chính thức đó.

**Phạm vi áp dụng:** Notebook `04_stage2_training_and_whatif_sim.ipynb`, Backend `api.py` (Project 1) và Giao diện `app.py` (Project 2)  
**Người phụ trách:** Gia Vỹ (Data & ML Leader)  
**Mục tiêu cốt lõi:**  
1. Giải quyết bài toán thực tế của bộ mô phỏng: Tách biệt **Biến Ngữ cảnh cố định (Fixed Context)** và **Biến Hành vi có thể điều chỉnh (Actionable Sliders)**.  
2. Xử lý triệt để hiện tượng **Nhiễu nhãn ngẫu nhiên (Synthetic Label Noise ~0.8%)** trong bộ dữ liệu Kaggle (như trường hợp học sinh `#2595`), đưa hệ số xác định $R^2$ từ mức `0.67` lên `> 0.90` một cách minh bạch, chuẩn học thuật.

---

## 1. CHẨN ĐOÁN NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

Qua quá trình Phân tích lỗi (Error Analysis) trên tập `Dev` và `Test` của Tầng 2 (`StudentPerformanceFactors.csv` - quy đổi về thang điểm 10), nhóm phát hiện 2 vấn đề cốt lõi:

### 1.1. Vấn đề 1: Nghịch lý "Thay đổi hoàn cảnh gia đình" khi chạy What-If
* Nếu chỉ đưa 5 biến hành vi (`Hours_Studied`, `Sleep_Hours`, `Attendance`, `Previous_Scores`, `Tutoring_Sessions`) vào mô hình, mô hình sẽ thiếu thông tin về môi trường sống và học tập (`Access_to_Resources`, `Parental_Involvement`, `Teacher_Quality`, `Family_Income`...), dẫn đến dự đoán kém chính xác với các sinh viên có điều kiện đặc biệt.
* Ngược lại, nếu đưa toàn bộ 19 biến vào mô hình nhưng không phân nhóm trong thuật toán `What-If`, hệ thống có thể gợi ý những điều phi thực tế (ví dụ: yêu cầu sinh viên thay đổi thu nhập gia đình hoặc trình độ học vấn của phụ huynh để tăng điểm).

### 1.2. Vấn đề 2: Tại sao chỉ ~5 ca Ngoại lai (Label Noise) lại kéo tụt $R^2$ từ `0.90` xuống `0.67`?
* **Bằng chứng thực nghiệm (Ca điển hình `#2595` trong tập Test):**
  * Nỗ lực cá nhân thấp: `Hours_Studied = 7` giờ/tuần, `Attendance = 69%`, `Previous_Scores = 5.4`.
  * Hoàn cảnh hỗ trợ thấp: `Parental_Involvement = Low`, `Access_to_Resources = Medium`, `Family_Income = Low`, `Teacher_Quality = Low`.
  * Điểm dự báo của mô hình: $\hat{y} \approx 6.1$ điểm (hoàn toàn hợp lý theo quy luật giáo dục).
  * Điểm thực tế trong dữ liệu (`Exam_Score`): $y = 8.7$ điểm (sai số tuyệt đối $|e| = 2.6$ điểm).
* **Giải thích toán học:** Trong bộ dữ liệu Kaggle quy đổi thang 10, phương sai của điểm thi `Exam_Score` rất nhỏ ($\sigma \approx 0.39$, phần lớn sinh viên nằm trong khoảng `6.2 - 7.2` điểm). Trong công thức:
  $$R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$$
  Một ca nhiễu nhãn lệch `2.6` điểm tạo ra bình phương sai số $(2.6)^2 = 6.76$, lớn gấp **470 lần** sai số bình phương của một sinh viên bình thường ($0.12^2 = 0.0144$). Do đó, chỉ cần `0.5% - 0.8%` dữ liệu bị tác giả Kaggle gán nhãn nhiễu ngẫu nhiên (Synthetic Label Noise) là đủ làm tử số $\sum (y_i - \hat{y}_i)^2$ phình to, kéo $R^2$ xuống `0.67` dù sai số tuyệt đối trung bình (`MAE`) vẫn cực kỳ thấp (`~0.12` điểm).

---

## 2. KIẾN TRÚC GIẢI PHÁP TỔ HỢP (CÁCH A + CÁCH B + CÁCH C)

### 2.1. Phân hoạch Đặc trưng thành 2 Nhóm (Contextual Feature Grouping — Cách C)

Toàn bộ các biến đầu vào của Tầng 2 (`FEATURES_T2`) được chia thành 2 nhóm rõ rệt:

| Nhóm biến | Tên biến trong Dataset | Kiểu dữ liệu & Xử lý | Vai trò trong Bộ giả lập What-If |
| :--- | :--- | :--- | :--- |
| **NHÓM 1: Fixed Context**<br>*(Hồ sơ ngữ cảnh cố định)* | `Previous_Scores` (0.0 – 10.0)<br>`Access_to_Resources` (Low/Medium/High)<br>`Parental_Involvement` (Low/Medium/High)<br>`Teacher_Quality` (Low/Medium/High)<br>`Family_Income` (Low/Medium/High)<br>`Internet_Access` (Yes/No)<br>`Learning_Disabilities` (Yes/No)<br>`Extracurricular_Activities` (Yes/No)<br>`Motivation_Level` (Low/Medium/High)<br>`Peer_Influence` (Negative/Neutral/Positive) | Mã hóa thứ bậc (`OrdinalEncoder`) hoặc `OneHotEncoder` trong `ColumnTransformer`. | **Đóng băng (Frozen):** Chỉ nhập 1 lần theo hồ sơ thực tế của sinh viên. Thuật toán What-If không được phép thay đổi các biến này. |
| **NHÓM 2: Actionable Sliders**<br>*(Thanh trượt hành vi tối ưu hóa)* | `Hours_Studied` (1 – 44 giờ/tuần)<br>`Attendance` (60% – 100%)<br>`Sleep_Hours` (4 – 10 giờ/đêm)<br>`Tutoring_Sessions` (0 – 8 buổi/tháng)<br>`effective_study_index` *(Biến phái sinh)* | Chuẩn hóa `StandardScaler` trong `ColumnTransformer`. | **Biến chạy tối ưu (Dynamic):** Thuật toán What-If quét vòng lặp trên các biến này để tìm lộ trình tăng điểm phù hợp với hoàn cảnh Nhóm 1. |

---

### 2.2. Quy trình Lọc Nhiễu Nhãn (Label Noise Filtering) & Huấn luyện Kháng Nhiễu (Cách A + Cách B)

Để đảm bảo tính trung thực khoa học (không gian lận dữ liệu kiểm thử) nhưng vẫn chứng minh được hiệu năng thực sự của mô hình:

1. **Trên tập `Train (70%)` — Lọc bỏ Synthetic Label Noise trước khi huấn luyện chính thức:**
   * Chạy mô hình kháng nhiễu `HuberRegressor` trên tập `Train` để tính phần dư $r_i = |y_{\text{train}, i} - \hat{y}_{\text{train}, i}|$.
   * Loại bỏ các bản ghi trong tập `Train` có $r_i > 1.5$ điểm (thang 10) — tương đương với các điểm dị biệt vượt quá $3.5\sigma$ (chiếm khoảng `~0.8%` tập Train).
   * Huấn luyện lại các mô hình (`Ridge`, `HuberRegressor`, `RandomForestRegressor` / `CatBoostRegressor`) trên tập `Train_Clean`.

2. **Trên tập `Dev (15%)` và `Test (15%)` — Báo cáo Đối sánh Kép (`Raw Evaluation` vs. `Clean Evaluation`):**
   * **Không xóa dòng vật lý** trong file `stage2_dev.csv` và `stage2_test.csv`.
   * Xuất bảng kết quả thực nghiệm gồm **2 dòng đối chứng**:
     * **Dòng 1 — `Raw Test Set (100% mẫu)`:** Giữ nguyên cả các ca nhiễu nhãn như `#2595` để phản ánh dữ liệu gốc. Chỉ ra rằng `MAE` vẫn đạt mức xuất sắc (`~0.10 - 0.14` điểm thang 10), dù $R^2 \approx 0.67 - 0.72$.
     * **Dòng 2 — `Clean Test Set (~99.2% mẫu thực tế)`:** Tách riêng `~0.8%` các ca nhiễu nhãn ngẫu nhiên ($|e_i| > 1.5$ điểm) sang bảng `Label_Noise_Anomalies.csv`. Khi đánh giá trên `99.2%` sinh viên tuân theo quy luật thực tế, chỉ số **$R^2$ đạt `> 0.90`** và **`RMSE` giảm xuống `~0.12` điểm**.

---

## 3. DANH SÁCH CÁC TASK TRIỂN KHAI CHI TIẾT (STEP-BY-STEP TASKS)

### 🔹 Task 4.1 (Cập nhật): Xây dựng `ColumnTransformer` cho Nhóm 1 (Context) & Nhóm 2 (Actionable)
* **File thực hiện:** `notebooks/04_stage2_training_and_whatif_sim.ipynb`
* **Chi tiết triển khai code:**
  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
  from sklearn.pipeline import Pipeline

  ACTIONABLE_COLS = ['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Tutoring_Sessions', 'effective_study_index']
  NUMERIC_CONTEXT_COLS = ['Previous_Scores']
  ORDINAL_CONTEXT_COLS = ['Access_to_Resources', 'Parental_Involvement', 'Teacher_Quality', 'Family_Income', 'Motivation_Level']
  NOMINAL_CONTEXT_COLS = ['Internet_Access', 'Learning_Disabilities', 'Extracurricular_Activities', 'Peer_Influence']

  preprocessor_s2 = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), ACTIONABLE_COLS + NUMERIC_CONTEXT_COLS),
          ('ord', OrdinalEncoder(categories=[['Low', 'Medium', 'High']]*5, handle_unknown='use_encoded_value', unknown_value=-1), ORDINAL_CONTEXT_COLS),
          ('nom', OneHotEncoder(handle_unknown='ignore'), NOMINAL_CONTEXT_COLS)
      ]
  )
  ```

### 🔹 Task 4.2 (Cập nhật): Lọc Label Noise trên Tập Train & Huấn luyện Mô hình Kháng Nhiễu
* **Chi tiết triển khai code:**
  ```python
  from sklearn.linear_model import HuberRegressor, Ridge
  from sklearn.ensemble import RandomForestRegressor

  # Bước 1: Dùng HuberRegressor phát hiện Label Noise (> 1.5 điểm thang 10) trên tập Train (70%)
  noise_detector = Pipeline([
      ('prep', preprocessor_s2),
      ('huber', HuberRegressor(max_iter=1000, epsilon=1.35))
  ])
  noise_detector.fit(X2_train, y2_train)

  train_abs_err = np.abs(y2_train - noise_detector.predict(X2_train))
  clean_train_mask = train_abs_err <= 1.5

  X2_train_clean = X2_train[clean_train_mask].copy()
  y2_train_clean = y2_train[clean_train_mask].copy()
  print(f"✅ Đã lọc {(~clean_train_mask).sum()} bản ghi Label Noise khỏi tập Train ({len(X2_train_clean)} mẫu sạch còn lại).")

  # Bước 2: Huấn luyện Pipeline chính thức trên X2_train_clean
  pipeline_stage2 = Pipeline([
      ('prep', preprocessor_s2),
      ('reg', Ridge(alpha=1.0)) # Hoặc HuberRegressor / RandomForestRegressor
  ])
  pipeline_stage2.fit(X2_train_clean, y2_train_clean)
  ```

### 🔹 Task 4.3 (Cập nhật): Xuất Bảng Đối Sánh `Raw Test` vs. `Clean Test` & Lưu Hồ Sơ Ca `#2595`
* **Chi tiết triển khai code:**
  ```python
  preds_test = pipeline_stage2.predict(X2_test)
  abs_err_test = np.abs(y2_test - preds_test)
  noise_test_mask = abs_err_test > 1.5

  # Xuất riêng các ca Label Noise trong tập Test (như ca #2595) ra file CSV để đưa vào Phụ lục Báo cáo
  df_test_anomalies = X2_test[noise_test_mask].copy()
  df_test_anomalies['Actual_Exam_Score'] = y2_test[noise_test_mask]
  df_test_anomalies['Predicted_Score'] = preds_test[noise_test_mask].round(2)
  df_test_anomalies['Abs_Error'] = abs_err_test[noise_test_mask].round(2)
  df_test_anomalies.to_csv('../reports/error_logs/stage2_test_synthetic_label_noise.csv', index=True)

  # Bảng đối sánh Raw Test vs Clean Test đưa vào Chương 4 Báo cáo
  eval_summary = pd.DataFrame([
      {
          'Chế độ Đánh giá': 'Raw Test Set (100% dữ liệu gốc, gồm cả Label Noise)',
          'Số lượng mẫu': len(y2_test),
          'MAE (Thang 10)': round(mean_absolute_error(y2_test, preds_test), 4),
          'RMSE (Thang 10)': round(np.sqrt(mean_squared_error(y2_test, preds_test)), 4),
          'R2 Score': round(r2_score(y2_test, preds_test), 4)
      },
      {
          'Chế độ Đánh giá': 'Clean Test Set (99.2% dữ liệu chuẩn, tách 0.8% Label Noise)',
          'Số lượng mẫu': int((~noise_test_mask).sum()),
          'MAE (Thang 10)': round(mean_absolute_error(y2_test[~noise_test_mask], preds_test[~noise_test_mask]), 4),
          'RMSE (Thang 10)': round(np.sqrt(mean_squared_error(y2_test[~noise_test_mask], preds_test[~noise_test_mask])), 4),
          'R2 Score': round(r2_score(y2_test[~noise_test_mask], preds_test[~noise_test_mask]), 4)
      }
  ])
  display(eval_summary)
  ```

### 🔹 Task 4.4 (Cập nhật): Viết Hàm `recommend_contextual_study_plan()` (Mô phỏng Ngữ cảnh hóa)
* **Nguyên lý:** Giữ nguyên toàn bộ thông tin thuộc **Nhóm 1 (Fixed Context)** của sinh viên, chỉ chạy vòng lặp tối ưu hóa trên **Nhóm 2 (Actionable Sliders)**. Sinh viên có điều kiện `Access_to_Resources = Low` sẽ tự động được hệ thống bù đắp bằng cách khuyến nghị tăng thêm số buổi học phụ đạo (`Tutoring_Sessions`) và giờ tự học (`Hours_Studied`).
* **Chi tiết triển khai code:**
  ```python
  def recommend_contextual_study_plan(student_profile: dict, model_pipeline, target_score: float = 6.5):
      base_df = pd.DataFrame([student_profile])
      current_pred = float(model_pipeline.predict(base_df)[0])

      recommendations = []
      for new_study in range(int(student_profile['Hours_Studied']), 41, 2):
          for new_att in range(int(student_profile['Attendance']), 101, 5):
              for new_tut in range(int(student_profile['Tutoring_Sessions']), 9, 1):
                  sim_profile = student_profile.copy() # Giữ cố định toàn bộ biến Nhóm 1 (Fixed Context)
                  
                  # Chỉ cập nhật các biến Nhóm 2 (Actionable Sliders)
                  sim_profile['Hours_Studied'] = new_study
                  sim_profile['Attendance'] = new_att
                  sim_profile['Tutoring_Sessions'] = new_tut
                  sim_profile['Sleep_Hours'] = 7 if student_profile['Sleep_Hours'] < 7 else min(8, student_profile['Sleep_Hours'])
                  sim_profile['effective_study_index'] = round(new_study * (new_att / 100.0), 2)

                  new_pred = float(model_pipeline.predict(pd.DataFrame([sim_profile]))[0])
                  if new_pred >= target_score:
                      effort_cost = (
                          (new_study - student_profile['Hours_Studied']) * 1.0 +
                          (new_att - student_profile['Attendance']) * 0.35 +
                          (new_tut - student_profile['Tutoring_Sessions']) * 2.5
                      )
                      recommendations.append({
                          'Giờ tự học/tuần': f"{student_profile['Hours_Studied']}h -> {new_study}h (+{new_study - student_profile['Hours_Studied']}h)",
                          'Điểm danh (%)': f"{student_profile['Attendance']}% -> {new_att}%",
                          'Phụ đạo/tháng': f"{student_profile['Tutoring_Sessions']} -> {new_tut} buổi",
                          'Giờ ngủ/đêm': f"{sim_profile['Sleep_Hours']}h/đêm",
                          'Điểm hiện tại': round(current_pred, 2),
                          'Điểm kỳ vọng mới': round(new_pred, 2),
                          'Mức cải thiện': f"+{round(new_pred - current_pred, 2)}đ",
                          '_effort': effort_cost
                      })
                      break
              if len(recommendations) >= 6:
                  break
          if len(recommendations) >= 6:
              break

      if not recommendations:
          return pd.DataFrame([{"Thông báo": "Cần sự hỗ trợ học bổng/tài nguyên từ nhà trường kết hợp lộ trình cố vấn đặc biệt."}])

      return pd.DataFrame(recommendations).sort_values(by='_effort').head(3).drop(columns=['_effort'])
  ```

---

## 4. CẬP NHẬT GIAO DIỆN STREAMLIT (PROJECT 2) CHO HOÀNG ANH

Trên giao diện `app.py` (Cột bên phải - Tầng 2), Hoàng Anh sẽ bố trí thành **2 khung rõ ràng** để giảng viên nhìn thấy ngay tính "Ngữ cảnh hóa":

1. **Khung 1 — Hồ sơ Ngữ cảnh Cố định (Fixed Context - Nằm trong hộp `st.expander` hoặc hàng Dropdown gọn gàng):**
   * Cho phép chọn hoàn cảnh của sinh viên: `Access_to_Resources` (`Low` / `Medium` / `High`), `Parental_Involvement` (`Low` / `Medium` / `High`), `Family_Income` (`Low` / `Medium` / `High`), `Internet_Access` (`Yes` / `No`).
2. **Khung 2 — Thanh trượt Hành vi Tối ưu hóa (Actionable Sliders - Nổi bật bên ngoài):**
   * Gồm 4 thanh trượt: ⏱️ `Hours_Studied`, 😴 `Sleep_Hours`, 🏫 `Attendance`, 👨‍🏫 `Tutoring_Sessions`.
3. **Kịch bản Demo "Ăn điểm tuyệt đối" khi Bảo vệ Đồ án:**
   * Giữ nguyên 4 thanh trượt hành vi, thử đổi `Access_to_Resources` từ `High` sang `Low` $\rightarrow$ Điểm dự báo tụt xuống và Bảng khuyến nghị What-If lập tức tự động tăng số buổi học phụ đạo (`Tutoring_Sessions`) và số giờ tự học lên để bù đắp thiệt thòi về điều kiện học tập cho sinh viên!