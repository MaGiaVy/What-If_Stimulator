# Spec.md — Hệ thống Hỗ trợ Học tập Lớp 12

> **Phiên bản:** 2.0  
> **Cập nhật lần cuối:** 2026-10-19  
> **Thay đổi lớn so với v1:** Bỏ OULAD + Stage 1 OULAD. Dùng duy nhất `StudentPerformanceFactors.csv` cho cả 2 tầng. Scope đổi sang **học sinh lớp 12 Việt Nam**.

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
Gender                 # Giới tính
```

**Cột mục tiêu:**
```
Exam_Score             # Điểm thi cuối kỳ (0–10 sau convert)
at_risk                # NHÃN TẦNG 1: 1 nếu Exam_Score < 6.5 (Không đạt loại Khá: 32.3%), else 0
```

> **Lưu ý quan trọng:** `at_risk` được tạo ra từ `Exam_Score` — **không dùng `Exam_Score` làm feature của Tầng 1**.

---

## 3. Splitting dữ liệu

- **Train:** 70% — dùng cho cả Tầng 1 và Tầng 2
- **Dev (Blackbox):** 15% — tuning, chọn model
- **Test:** 15% — đánh giá cuối cùng, **không động vào trong quá trình phát triển**
- `random_state = 42` cho tất cả các bước split
- **Cùng 1 lần split duy nhất** → cả 2 tầng dùng chung index → không bị leakage

---

## 4. Tầng 1 — Phân loại học sinh nguy cơ

### Mục tiêu
Phân loại học sinh có nguy cơ rớt môn (`at_risk = 1`) trước khi thi.

### Ngưỡng nguy hiểm
`at_risk = 1` khi `Exam_Score (thang 10) < 6.5` (Không đạt loại Khá chuẩn đầu ra THPT / xét tuyển Đại học; chiếm ~32.3% mẫu). Điểm dưới 6.5 cần can thiệp sớm.

### Features
Tất cả các cột (hành vi + hoàn cảnh), **trừ `Exam_Score`**.

### Thuật toán
Thử cả 3, chọn cái tốt nhất theo F1-score trên Dev:
1. **Logistic Regression** (baseline đơn giản)
2. **Random Forest** (cân bằng giữa hiệu suất và giải thích được)
3. **XGBoost + SMOTE** (xử lý mất cân bằng nhãn)

### Metric đánh giá
| Metric | Ưu tiên | Lý do |
|---|---|---|
| **Recall** | ⭐ Cao nhất | Không bỏ sót học sinh nguy hiểm |
| F1-score | Cân bằng Recall và Precision | |
| ROC-AUC | Đánh giá tổng quát | |

### Output
- Xác suất nguy cơ (0.0–1.0) cho từng học sinh
- Nhãn phân loại: `at_risk` / `safe`
- Feature importance: giải thích yếu tố nào ảnh hưởng nhất

---

## 5. Tầng 2 — Dự đoán điểm và What-If Simulator

### Mục tiêu
Dự đoán điểm thi (thang 10) và mô phỏng "Nếu học sinh thay đổi hành vi X thì điểm thay đổi bao nhiêu?"

### Features
Tất cả các cột (hành vi + hoàn cảnh), **trừ `at_risk`**.

### Thuật toán
- **Ridge Regression** + **Interaction Terms** (như kế hoạch cũ)
- Preprocessing: `ColumnTransformer` gồm `StandardScaler` (số) + `OrdinalEncoder` + `OneHotEncoder` (categorical)
- Xử lý nhiễu label: `HuberRegressor` filter trên tập Train (loại ca bất thường)

### Interaction Terms (thêm mới)
```python
hours_x_access        = Hours_Studied × access_ordinal
tutoring_x_motivation = Tutoring_Sessions × motivation_ordinal
attend_x_parental     = Attendance × parental_ordinal
```
Chỉ giữ nếu Dev MAE giảm > 0.005 so với baseline.

### What-If Simulator
- GV/PH điều chỉnh **các cột hành vi** (slider)
- **Các cột hoàn cảnh** giữ cố định theo hồ sơ học sinh
- Output: Điểm dự đoán mới + chênh lệch so với hiện tại
- Gợi ý ưu tiên: Sắp xếp hành động theo marginal effect (hành động nào tăng điểm nhiều nhất)

### Metric đánh giá
| Metric | Target |
|---|---|
| **MAE (Test Raw)** | ≤ 0.08 (thang 10) |
| R² (Test Raw) | > 0.60 |

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
| `05_stage2_regression_whatif.ipynb` | Ridge + interaction terms, What-If Simulator, export model |

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
