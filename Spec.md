# Spec.md — Hệ thống Hỗ trợ Học tập Lớp 12

> **Phiên bản:** 2.1  
> **Cập nhật lần cuối:** 2026-10-21  
> **Thay đổi lớn so với v1:** Bỏ OULAD + Stage 1 OULAD. Dùng duy nhất `StudentPerformanceFactors.csv` cho cả 2 tầng. Scope đổi sang **học sinh lớp 12 Việt Nam**.  
> **Thay đổi so với v2.0:** Bỏ `Gender` + `School_Type` (còn 18 feature); bổ sung metric đánh giá; ghi nhận thực tế `at_risk` ~22%, Interaction Terms đã loại; **quyết định không dùng dataset thứ 2** (xem mục 12).

---

## 1. Tổng quan hệ thống

**Tên project:** Hệ thống Hỗ trợ Học tập Lớp 12  
**Mục tiêu:** Vừa cảnh báo nguy cơ rớt môn, vừa gợi ý cải thiện kết quả học tập cho học sinh lớp 12.  
**Người dùng chính:** Giáo viên chủ nhiệm / Giáo viên bộ môn + Phụ huynh học sinh.

### Luồng hoạt động

```
Nhập thông tin học sinh
        │
        ├── TẦNG 1 (Classification)
        │     Phân loại: at_risk = 1 nếu điểm dự đoán < 6.5 (Không đạt loại Khá)
        │     → Dashboard GV/PH: "Danh sách học sinh nguy hiểm 🚩"
        │
        └── TẦNG 2 (Regression + What-If)
              Dự đoán điểm thi (thang 10)
              → What-If Simulator: "Nếu tăng giờ học → điểm tăng bao nhiêu?"
              → Gợi ý hành động cụ thể
```

---

## 2. Dataset

| Thuộc tính | Chi tiết |
|---|---|
| **File** | `data/raw/StudentPerformanceFactors.csv` |
| **Nguồn** | Kaggle — StudentPerformanceFactors |
| **Số bản ghi** | ~6,607 học sinh |
| **Thang điểm gốc** | `Exam_Score` 0–100 → **convert sang 0–10** (`Exam_Score / 10`) |
| **Scope** | Áp dụng cho **học sinh lớp 12 Việt Nam** (1 kỳ học) |

### Phân loại cột

**Cột hành vi — Học sinh tự thay đổi được (dùng làm slider What-If):**
```
Hours_Studied          # Giờ học/tuần
Attendance             # % điểm danh
Sleep_Hours            # Giờ ngủ/ngày
Tutoring_Sessions      # Số buổi học thêm/tháng
Extracurricular_Activities  # Có hoạt động ngoại khóa không (Yes/No)
```

**Cột hoàn cảnh — GV/PH/Nhà trường can thiệp:**
```
Previous_Scores        # Điểm kỳ trước (thang 10 sau convert)
Access_to_Resources    # Mức tiếp cận tài liệu (High/Medium/Low)
Parental_Involvement   # Mức tham gia của PH (High/Medium/Low)
Teacher_Quality        # Chất lượng GV (High/Medium/Low)
Family_Income          # Thu nhập gia đình (High/Medium/Low)
Motivation_Level       # Mức độ động lực (High/Medium/Low)
Internet_Access        # Có Internet không (Yes/No)
Learning_Disabilities  # Có khó khăn học tập không (Yes/No)
Peer_Influence         # Ảnh hưởng bạn bè (Positive/Neutral/Negative)
Distance_from_Home     # Khoảng cách từ nhà (Near/Moderate/Far)
Parental_Education_Level  # Trình độ học vấn của PH
Physical_Activity      # Giờ vận động/tuần
```

**Cột bị loại (v2.1):** `Gender`, `School_Type` — permutation importance ≈ 0 ở cả 2 tầng; bỏ thêm giúp model không dùng giới tính để dự đoán. Thử nghiệm: bỏ 2 cột → MAE Tầng 2 0.0506→0.0505, F1 Tầng 1 0.9771→0.9748 (nằm trong nhiễu).

**Cột dẫn xuất:** `effective_study_index = Hours_Studied × Attendance / 100` (tính trước khi đưa vào pipeline).

**Số feature đầu vào: 18.**

**Cột mục tiêu:**
```
Exam_Score             # Điểm thi cuối kỳ (0–10 sau convert)
at_risk                # NHÃN TẦNG 1: 1 nếu Exam_Score < 6.5 (Không đạt loại Khá: ~21.98%), else 0
```

> **Lưu ý quan trọng:** `at_risk` được tạo ra từ `Exam_Score` — **không dùng `Exam_Score` làm feature của Tầng 1**.

---

## 3. Splitting dữ liệu

- **Train:** 70% (4,624) — dùng cho cả Tầng 1 và Tầng 2
- **Dev:** 15% — tách thành Dev Blackbox (693, tuning/chọn model) + Dev Eyeball (298, soi lỗi thủ công)
- **Test:** 15% (992) — đánh giá cuối cùng, **không động vào trong quá trình phát triển**
- `stratify = at_risk`
- Outlier nhiễu nhãn: chỉ lọc trên **Train** (HuberRegressor, |residual| > 1.5, 32 ca); **Test giữ nguyên**, báo cáo cả Raw lẫn Clean (loại 10 ca) làm sensitivity analysis
- `random_state = 42` cho tất cả các bước split
- **Cùng 1 lần split duy nhất** → cả 2 tầng dùng chung index → không bị leakage

---

## 4. Tầng 1 — Phân loại học sinh nguy cơ

### Mục tiêu
Phân loại học sinh có nguy cơ rớt môn (`at_risk = 1`) trước khi thi.

### Ngưỡng nguy hiểm
`at_risk = 1` khi `Exam_Score (thang 10) < 6.5` (Không đạt loại Khá chuẩn đầu ra THPT / xét tuyển Đại học; chiếm ~22% mẫu). Điểm dưới 6.5 cần can thiệp sớm. (Dataset không có mẫu nào < 5.5 nên ngưỡng 5.0 không khả thi.)

### Features
18 feature (xem mục 2), **trừ `Exam_Score` và `at_risk`**.

### Thuật toán
Thử cả 3, chọn cái tốt nhất theo F1-score trên Dev Blackbox:
1. **Logistic Regression + SMOTE** (baseline, kết quả thực tế: thắng)
2. **Random Forest**
3. **XGBoost + SMOTE**

Tối ưu ngưỡng quyết định T* trên Dev (hiện tại 0.53).

### Metric đánh giá
| Metric | Ưu tiên | Target | Lý do |
|---|---|---|---|
| **Recall** | ⭐ Cao nhất | ≥ 0.85 | Không bỏ sót học sinh nguy hiểm |
| F1-score | | ≥ 0.85 | Cân bằng Recall và Precision |
| ROC-AUC | | ≥ 0.90 | Đánh giá tổng quát |
| **PR-AUC** (mới) | Bổ sung | báo cáo | Chặt hơn ROC-AUC khi nhãn lệch ~22% |
| **MCC** (mới) | Bổ sung | báo cáo | Cân bằng khi nhãn lệch |
| **Balanced Accuracy, Specificity** (mới) | Bổ sung | báo cáo | Thấy rõ mức báo động nhầm |
| **Confusion matrix** (mới) | Bổ sung | báo cáo | Số ca tuyệt đối TP/FP/FN/TN |
| **Brier score + Calibration curve** (mới) | Bổ sung | báo cáo | Xác suất hiển thị cho PH phải đáng tin |
| **5-fold CV (mean ± std)** (mới) | Bổ sung | báo cáo | Kết quả không phụ thuộc một lần split |

### Output
- Xác suất nguy cơ (0.0–1.0) cho từng học sinh
- Nhãn phân loại: `at_risk` / `safe`
- Feature importance: giải thích yếu tố nào ảnh hưởng nhất

---

## 5. Tầng 2 — Dự đoán điểm và What-If Simulator

### Mục tiêu
Dự đoán điểm thi (thang 10) và mô phỏng "Nếu học sinh thay đổi hành vi X thì điểm thay đổi bao nhiêu?"

### Features
18 feature (xem mục 2), **trừ `Exam_Score` và `at_risk`**.

### Thuật toán
- **Ridge Regression** (alpha = 1.0)
- Preprocessing: `ColumnTransformer` gồm `StandardScaler` (số) + `OrdinalEncoder` + `OneHotEncoder` (categorical)
- Xử lý nhiễu label: `HuberRegressor` filter trên tập Train (loại ca bất thường)

### Interaction Terms — ĐÃ THỬ, ĐÃ LOẠI
```python
hours_x_access        = Hours_Studied × access_ordinal
tutoring_x_motivation = Tutoring_Sessions × motivation_ordinal
attend_x_parental     = Attendance × parental_ordinal
```
Quy tắc: chỉ giữ nếu Dev MAE giảm > 0.005. Kết quả thực tế: ΔMAE = -0.00005 → loại để giữ model tinh gọn.

### What-If Simulator
- GV/PH điều chỉnh **các cột hành vi** (slider)
- **Các cột hoàn cảnh** giữ cố định theo hồ sơ học sinh
- Output: Điểm dự đoán mới + chênh lệch so với hiện tại
- Gợi ý ưu tiên: Sắp xếp hành động theo marginal effect (hành động nào tăng điểm nhiều nhất)
- `Sleep_Hours` **giữ nguyên** là slider What-If như cũ (dù permutation importance ≈ 0; GV/PH vẫn thấy tác động nhỏ)
- Actionable chính (importance cao nhất): `Attendance`, `Hours_Studied`, `Tutoring_Sessions`

### Metric đánh giá
| Metric | Target |
|---|---|
| **MAE (Test Raw)** | ≤ 0.08 (thang 10) |
| R² (Test Raw) | > 0.60 |
| MAE / R² (Test Clean, loại ca nhiễu) | báo cáo (sensitivity) |
| **Median AE** (mới) | báo cáo (không bị outlier kéo lệch) |
| **% dự đoán lệch ≤ 0.5 điểm, P95 \|error\|** (mới) | báo cáo (dễ hiểu với GV/PH) |
| **Adjusted R²** (mới) | báo cáo (n=992, p=18) |
| **5-fold CV MAE (mean ± std), Bootstrap 95% CI** (mới) | báo cáo |
| **Permutation Importance** (mới) | xếp hạng feature, kiểm tra chiều tác động What-If đơn điệu |

---

## 6. Kiến trúc kỹ thuật

```
┌─────────────────────────────────────────────┐
│              Next.js Frontend               │
│   Dashboard GV/PH + What-If Simulator UI   │
└───────────────────┬─────────────────────────┘
                    │ HTTP
┌───────────────────▼─────────────────────────┐
│              NestJS Gateway                 │
│   Auth, routing, business logic             │
└───────────────────┬─────────────────────────┘
                    │ HTTP
┌───────────────────▼─────────────────────────┐
│              FastAPI ML Service             │
│   /predict/stage1  →  at_risk + probability │
│   /predict/stage2  →  score + what-if       │
│   /recommend       →  gợi ý cải thiện       │
└───────────────────┬─────────────────────────┘
                    │ joblib.load()
┌───────────────────▼─────────────────────────┐
│            Model Files (.pkl)               │
│   models/pipeline_stage1.pkl               │
│   models/pipeline_stage2_score.pkl         │
└─────────────────────────────────────────────┘
```

---

## 7. Cấu trúc Notebooks

| File | Nội dung |
|---|---|
| `01_data_prep.ipynb` | Load StudentPerformanceFactors, convert thang 10, EDA cơ bản, tạo nhãn `at_risk` |
| `02_data_splitting.ipynb` | Split Train/Dev/Test (70/15/15), lưu splits, kiểm tra phân phối |
| `03_stage1_baseline.ipynb` | Logistic Regression baseline cho Tầng 1 |
| `04_stage1_advanced.ipynb` | Random Forest + XGBoost + SMOTE, threshold tuning, chọn model tốt nhất |
| `05_stage2_regression_whatif.ipynb` | Ridge (đã thử & loại interaction terms), What-If Simulator, export model |

---

## 8. Cấu trúc thư mục

```
What-If Simulator/
├── data/
│   ├── raw/
│   │   └── StudentPerformanceFactors.csv
│   └── processed/
│       ├── train.csv
│       ├── dev.csv
│       └── test.csv
├── models/
│   ├── pipeline_stage1.pkl
│   ├── pipeline_stage2_score.pkl
│   └── model_metadata.json
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_data_splitting.ipynb
│   ├── 03_stage1_baseline.ipynb
│   ├── 04_stage1_advanced.ipynb
│   └── 05_stage2_regression_whatif.ipynb
├── reports/
│   └── figures/
├── docs/
│   ├── BAO_CAO_DO_AN.md
│   └── HANDOFF_APP.md
├── Spec.md                ← file này
├── requirements.txt
└── .gitignore
```

---

## 9. Ràng buộc & Nguyên tắc

| Nguyên tắc | Chi tiết |
|---|---|
| **Không leakage** | `Exam_Score` không được là feature của Tầng 1. `at_risk` không được là feature của Tầng 2. |
| **Cùng split** | Cả 2 tầng dùng chung index từ 1 lần `train_test_split` duy nhất |
| **Không touch Test** | Test set chỉ dùng để đánh giá lần cuối, không dùng để tuning |
| **random_state = 42** | Tất cả nơi cần randomness |
| **Thang điểm** | Toàn bộ hệ thống dùng thang 10 (`Exam_Score / 10`) |
| **Ngưỡng nguy hiểm** | `at_risk` khi điểm dự đoán < 6.5 (Không đạt loại Khá) |

---

## 10. Những gì BỎ so với v1

| Bỏ | Lý do |
|---|---|
| OULAD dataset | Không phù hợp với context lớp 12 VN (hệ từ xa UK, 9 tháng/module) |
| `03a_stage1_baseline.ipynb` (OULAD) | Thay bằng `03_stage1_baseline.ipynb` (StudentPerf) |
| `03b_stage1_advanced_models.ipynb` (OULAD) | Thay bằng `04_stage1_advanced.ipynb` (StudentPerf) |
| Stage 1 OULAD models (XGBoost trên VLE data) | Không còn phù hợp |
| `stage2_plan.md` | Được tích hợp vào Spec.md này |

---

## 11. API Endpoints (FastAPI ML Service)

### POST `/predict/stage1`
```json
Request:
{
  "Hours_Studied": 5,
  "Attendance": 85,
  "Sleep_Hours": 7,
  "Tutoring_Sessions": 2,
  "Previous_Scores": 6.5,
  "Motivation_Level": "Medium",
  "Family_Income": "Low",
  "Access_to_Resources": "Medium",
  ...
}

Response:
{
  "at_risk": 1,
  "probability": 0.73,
  "top_risk_factors": ["Low Family_Income", "Low Motivation"]
}
```

### POST `/predict/stage2`
```json
Request:
{
  "Hours_Studied": 5,
  "Attendance": 85,
  ...
}

Response:
{
  "predicted_score": 4.8,
  "is_at_risk": true
}
```

### POST `/recommend`
```json
Request: { ...thông tin học sinh... }

Response:
{
  "recommendations": [
    {"action": "Tăng giờ học từ 5h lên 7h/tuần", "score_gain": +0.6},
    {"action": "Tăng điểm danh từ 85% lên 95%", "score_gain": +0.4}
  ]
}
```



---

## 12. Quyết định v2.1 & Kế hoạch thực hiện

### Quyết định
| Quyết định | Lý do |
|---|---|
| **Giữ 1 dataset** (`StudentPerformanceFactors.csv`) cho cả 2 tầng | Cùng một học sinh phải có cùng bộ feature ở cả 2 tầng |
| **Không dùng "Student Performance" 10k dòng (Performance Index) cho Tầng 2** | Chỉ 5 feature, mất `Attendance` (feature quan trọng nhất) và `Tutoring_Sessions`; target không phải điểm thi nên không nối được ngưỡng 6.5 của Tầng 1; hai tầng lệch đầu vào; phải viết lại Tầng 2 |
| **Bỏ `Gender`, `School_Type`** | Importance ≈ 0 ở cả 2 tầng; thử nghiệm cho thấy không mất điểm; giảm rủi ro thiên vị giới tính |
| **Giữ `Sleep_Hours`** trong model và What-If | Quyết định của nhóm |
| **Thêm metric đánh giá** (mục 4, 5) | Bổ sung bằng chứng, trả lời câu hỏi outlier mà không xóa dữ liệu |
| **Tích hợp Explainable AI (SHAP) cho cả 2 tầng** | Tầng 1: Giải thích xác suất nguy cơ + trích xuất `top_risk_factors` cho API/Dashboard. Tầng 2: Phân tích đóng góp điểm cho từng học sinh và trước/sau What-If. Xuất cả Beeswarm plot toàn cục và Waterfall plot cá nhân hóa. |

### Kết quả thử nghiệm bỏ 2 feature (Test, 992 HS)
| | 20 feature | 18 feature |
|---|---|---|
| Tầng 1 F1 / Precision / Recall / AUC | 0.9771 / 0.9771 / 0.9771 / 0.9974 | 0.9748 / 0.9726 / 0.9771 / 0.9974 |
| Tầng 2 MAE / R² (Raw) | 0.0506 / 0.6666 | 0.0505 / 0.6666 |

### Các task thực hiện (mỗi task = 1 commit riêng, user tự push)
| Task | Nội dung chi tiết | File liên quan |
|---|---|---|
| **5.1** | **Chuẩn hóa 18 features**: Bỏ `Gender`, `School_Type` khỏi master dataset, splits (train, dev, test) và định nghĩa pipeline Tầng 1 & Tầng 2. Huấn luyện lại, lưu model `.pkl` + `model_metadata.json`. | `notebooks/01..05`, `data/processed/`, `data/splits/`, `models/` |
| **5.2** | **Tầng 1 Advanced Metrics & SHAP XAI**: Thêm PR-AUC, MCC, Balanced Acc, Specificity, Brier score, Calibration curve, 5-fold CV vào `04_stage1_advanced.ipynb`. Tích hợp SHAP (LinearExplainer): vẽ Beeswarm plot (`fig12_shap_summary_stage1.png`), Waterfall plot cá nhân (`fig13_shap_waterfall_stage1.png`), viết hàm `extract_top_risk_factors()` cho API. | `notebooks/04_stage1_advanced.ipynb`, `reports/figures/` |
| **5.3** | **Tầng 2 Robust Metrics & SHAP What-If**: Thêm Median AE, % lệch ≤0.5, P95, Adjusted R², 5-fold CV, Bootstrap 95% CI vào `05_stage2_regression_whatif.ipynb`. Tích hợp SHAP: vẽ Beeswarm plot (`fig14_shap_summary_stage2.png`), Waterfall plot What-If (`fig15_shap_waterfall_whatif.png`), kiểm tra tính đơn điệu. | `notebooks/05_stage2_regression_whatif.ipynb`, `reports/figures/` |
| **5.4** | **Tài liệu Bàn giao & FastAPI Service với SHAP**: Cập nhật `BAO_CAO_DO_AN.md`, `HANDOFF_APP.md` (chuẩn 18 features, code FastAPI hoàn chỉnh có endpoint `/explain` và field `top_risk_factors` từ SHAP, NestJS / Next.js UI component), cập nhật `requirements.txt` (`shap>=0.42.0`). | `docs/`, `requirements.txt`, `Spec.md` |

### Acceptance Criteria
- Tầng 1 Test: Recall ≥ 0.85, F1 ≥ 0.85, ROC-AUC ≥ 0.90
- Tầng 2 Test Raw: MAE ≤ 0.08, R² > 0.60
- Không còn tham chiếu `Gender` / `School_Type` trong code, dữ liệu, docs, API
- Đầy đủ 4 hình SHAP trong `reports/figures/` (fig12 -> fig15)
- Code FastAPI có hàm trích xuất `top_risk_factors` hoạt động thực tế
