# HANDOFF — Hướng dẫn Tích hợp Mô hình vào Ứng dụng (App Integration)

> **Dành cho:** Kỹ sư Backend (**NestJS & FastAPI**) và Kỹ sư Frontend (**Next.js / React**).  
> **Dự án:** Hệ thống Hỗ trợ Học tập Lớp 12 Việt Nam (Phiên bản 2.1 — 18 Đặc trưng & Explainable AI với SHAP)  
> **Đầu ra Machine Learning:** 2 file pipeline `.pkl` đã tích hợp đầy đủ quy trình tiền xử lý dữ liệu và thuật toán mô phỏng What-If.

---

## 1. Danh mục Bàn giao & Tài nguyên

| File / Thư mục | Định dạng | Trách nhiệm sử dụng | Mô tả chức năng |
| :--- | :---: | :--- | :--- |
| `models/pipeline_stage1.pkl` | Binary (Joblib) | FastAPI Service | Pipeline Tầng 1: Phân loại nguy cơ học tập (`at_risk` < 6.5) |
| `models/pipeline_stage2_score.pkl` | Binary (Joblib) | FastAPI Service | Pipeline Tầng 2: Dự đoán điểm thi (0–10) & What-If Simulator |
| `models/model_metadata.json` | JSON | Backend & Frontend | Cấu hình siêu dữ liệu: 18 đặc trưng, ngưỡng $T^* = 0.53$, chỉ số kiểm định |
| `docs/BAO_CAO_DO_AN.md` | Markdown | Cả nhóm | Báo cáo học thuật chi tiết để hiểu rõ bối cảnh và logic mô hình |
| `reports/figures/` | PNG (300 DPI) | Frontend / Docs | 15 biểu đồ phân tích, chẩn đoán, đường cong ROC và SHAP Plots |

### Môi trường thực thi khuyến nghị (Python Backend ML Service)
Các pipeline được xuất xưởng từ Python 3.12. Yêu cầu cài đặt môi trường độc lập:
```bash
pip install scikit-learn>=1.8.0 imbalanced-learn>=0.14.0 pandas>=3.0.0 numpy>=2.0.0 joblib>=1.5.0 shap>=0.42.0 fastapi uvicorn[standard] pydantic
```

---

## 2. Kiến trúc Tích hợp Hệ thống (System Architecture)

```
┌────────────────────────────────────────────────────────┐
│                   Next.js Frontend                     │
│    Dashboard GV/PH + Thanh trượt What-If Simulator     │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (JSON)
┌───────────────────────────▼────────────────────────────┐
│                    NestJS Gateway                      │
│     Xác thực người dùng, Phân quyền GV/PH, Lưu DB      │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP (Internal Microservice)
┌───────────────────────────▼────────────────────────────┐
│                  FastAPI ML Service                    │
│      Nạp models/pipeline_stage1.pkl & stage2.pkl       │
│      - POST /predict/stage1 (kèm SHAP risk factors)    │
│      - POST /predict/stage2 (dự đoán điểm thi)         │
│      - POST /recommend      (lộ trình can thiệp tối ưu)│
└────────────────────────────────────────────────────────┘
```

---

## 3. Hợp đồng Dữ liệu Chi tiết (Data Contract)

Cả hai mô hình đều tiếp nhận **cùng một bộ thông tin học sinh gồm 18 đặc trưng đầu vào** (đã loại bỏ `Gender` và `School_Type` để tránh thiên vị và tối ưu độ tinh gọn).

### 3.1 Bảng Đặc tả Thuộc tính Học sinh (18 trường)

| Tên trường (Field Name) | Kiểu | Phân loại | Miền giá trị hợp lệ | Ghi chú cho UI / Database |
| :--- | :---: | :---: | :---: | :--- |
| `Hours_Studied` | int | Hành vi | $1 \le x \le 40$ | Số giờ tự học mỗi tuần (Thanh trượt What-If) |
| `Attendance` | int | Hành vi | $60 \le x \le 100$ | Tỷ lệ chuyên cần / điểm danh (%) (Thanh trượt) |
| `Sleep_Hours` | float | Hành vi | $4.0 \le x \le 10.0$ | Thời lượng giấc ngủ trung bình (giờ/đêm) |
| `Tutoring_Sessions` | int | Hành vi | $0 \le x \le 8$ | Số buổi đi học thêm mỗi tháng (Thanh trượt) |
| `Previous_Scores` | float | Hoàn cảnh | $0.0 \le x \le 10.0$ | Điểm thi kỳ trước (Thang 10, cố định) |
| `Physical_Activity` | int | Hoàn cảnh | $0 \le x \le 6$ | Số giờ hoạt động thể chất mỗi tuần |
| `Parental_Involvement` | string | Hoàn cảnh | `"Low"`, `"Medium"`, `"High"` | Sự quan tâm, kèm cặp của phụ huynh |
| `Access_to_Resources` | string | Hoàn cảnh | `"Low"`, `"Medium"`, `"High"` | Khả năng tiếp cận tài liệu, sách vở |
| `Motivation_Level` | string | Hoàn cảnh | `"Low"`, `"Medium"`, `"High"` | Động lực học tập của học sinh |
| `Family_Income` | string | Hoàn cảnh | `"Low"`, `"Medium"`, `"High"` | Điều kiện kinh tế gia đình |
| `Teacher_Quality` | string | Hoàn cảnh | `"Low"`, `"Medium"`, `"High"` | Đánh giá chất lượng giáo viên bộ môn |
| `Parental_Education_Level`| string | Hoàn cảnh | `"High School"`, `"College"`, `"Postgraduate"` | Trình độ học vấn của phụ huynh |
| `Extracurricular_Activities`| string | Hoàn cảnh | `"Yes"`, `"No"` | Có tham gia hoạt động ngoại khóa không |
| `Internet_Access` | string | Hoàn cảnh | `"Yes"`, `"No"` | Có kết nối Internet tại nhà không |
| `Peer_Influence` | string | Hoàn cảnh | `"Positive"`, `"Neutral"`, `"Negative"` | Môi trường bạn bè xung quanh |
| `Learning_Disabilities` | string | Hoàn cảnh | `"Yes"`, `"No"` | Có khó khăn / rào cản nhận thức không |
| `Distance_from_Home` | string | Hoàn cảnh | `"Near"`, `"Moderate"`, `"Far"` | Khoảng cách từ nhà đến trường |

> **Quy tắc kỹ thuật sống còn:** Backend hoặc FastAPI **tự động tính toán** trường tương tác:
> `effective_study_index = round(Hours_Studied * (Attendance / 100.0), 2)` trước khi đưa vào mô hình. Frontend không cần gửi trường này.

---

## 4. Mã nguồn Dịch vụ FastAPI ML Hoàn chỉnh (`api.py`)

Backend team có thể lưu file này vào `service/api.py` và chạy ngay (`uvicorn api:app --reload --port 8000`):

```python
# service/api.py
import os
import joblib
import pandas as pd
import numpy as np
import shap
from typing import Literal, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="ML Microservice — He thong Ho tro Hoc tap Lop 12",
    version="2.1",
    description="API phan loai nguy co hoc tap, giai thich SHAP va mo phong What-If cho GV va PH"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Đường dẫn mô hình
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_S1_PATH = os.path.join(BASE_DIR, 'models', 'pipeline_stage1.pkl')
MODEL_S2_PATH = os.path.join(BASE_DIR, 'models', 'pipeline_stage2_score.pkl')

print("⏳ Dang nap cac pipeline mo hinh...")
pipe_stage1 = joblib.load(MODEL_S1_PATH)
pipe_stage2 = joblib.load(MODEL_S2_PATH)
print("✅ Nap mo hinh thanh cong!")

# Ngưỡng quyết định tối ưu đã được tune trên Dev Blackbox
OPTIMAL_THRESHOLD_STAGE1 = 0.53

# Khởi tạo SHAP Explainer sẵn trong bộ nhớ cho Tầng 1
prep_s1 = pipe_stage1.named_steps['prep']
clf_s1 = pipe_stage1.named_steps['clf']
feature_names_s1 = [
    c.replace('num__', '').replace('ord__', '').replace('nom__', '')
    for c in prep_s1.get_feature_names_out()
]
shap_explainer_s1 = shap.LinearExplainer(
    clf_s1, np.zeros((1, len(feature_names_s1))), feature_names=feature_names_s1
)

# Từ điển Việt hóa giải thích SHAP cho GV/PH
VN_RISK_MAP = {
    'Attendance': 'Tỷ lệ chuyên cần thấp (< 80%)',
    'Hours_Studied': 'Số giờ tự học hàng tuần còn ít',
    'Previous_Scores': 'Điểm kiểm tra kỳ trước thấp (< 6.5)',
    'effective_study_index': 'Chỉ số tích lũy học tập thực tế chưa cao',
    'Parental_Involvement': 'Mức độ tham gia hỗ trợ của gia đình thấp',
    'Access_to_Resources': 'Thiếu hụt tài liệu và tài nguyên học tập',
    'Tutoring_Sessions': 'Chưa tham gia các buổi học thêm bổ trợ',
    'Motivation_Level': 'Động lực học tập cá nhân chưa cao',
    'Family_Income': 'Điều kiện kinh tế gia đình khó khăn',
    'Teacher_Quality': 'Chất lượng giảng dạy của giáo viên chưa tối ưu',
    'Learning_Disabilities_Yes': 'Có khó khăn hoặc rào cản trong học tập',
    'Peer_Influence_Negative': 'Ảnh hưởng tiêu cực từ nhóm bạn bè',
    'Distance_from_Home_Far': 'Khoảng cách nhà tới trường xa',
    'Internet_Access_No': 'Không có kết nối Internet tại nhà',
    'Extracurricular_Activities_No': 'Không tham gia hoạt động ngoại khóa',
    'Physical_Activity': 'Ít vận động thể chất'
}

# 2. Pydantic Schemas
class StudentProfile(BaseModel):
    Hours_Studied: int = Field(..., ge=1, le=40, example=15)
    Attendance: int = Field(..., ge=60, le=100, example=80)
    Sleep_Hours: float = Field(..., ge=4.0, le=10.0, example=7.0)
    Tutoring_Sessions: int = Field(..., ge=0, le=8, example=1)
    Previous_Scores: float = Field(..., ge=0.0, le=10.0, example=6.8)
    Physical_Activity: int = Field(..., ge=0, le=6, example=2)
    Parental_Involvement: Literal['Low', 'Medium', 'High'] = Field(..., example='Medium')
    Access_to_Resources: Literal['Low', 'Medium', 'High'] = Field(..., example='Medium')
    Motivation_Level: Literal['Low', 'Medium', 'High'] = Field(..., example='Medium')
    Family_Income: Literal['Low', 'Medium', 'High'] = Field(..., example='Medium')
    Teacher_Quality: Literal['Low', 'Medium', 'High'] = Field(..., example='Medium')
    Parental_Education_Level: Literal['High School', 'College', 'Postgraduate'] = Field(..., example='High School')
    Extracurricular_Activities: Literal['Yes', 'No'] = Field(..., example='No')
    Internet_Access: Literal['Yes', 'No'] = Field(..., example='Yes')
    Peer_Influence: Literal['Positive', 'Neutral', 'Negative'] = Field(..., example='Neutral')
    Learning_Disabilities: Literal['Yes', 'No'] = Field(..., example='No')
    Distance_from_Home: Literal['Near', 'Moderate', 'Far'] = Field(..., example='Near')

class WhatIfRequest(BaseModel):
    student: StudentProfile
    target_score: float = Field(..., ge=5.0, le=10.0, example=7.5)
    max_extra_hours: Optional[int] = Field(10, ge=0, le=20)
    max_extra_tutoring: Optional[int] = Field(4, ge=0, le=6)

def _prepare_df(profile: StudentProfile) -> pd.DataFrame:
    d = profile.model_dump()
    # Tự động tính trường tương tác bắt buộc
    d['effective_study_index'] = round(d['Hours_Studied'] * (d['Attendance'] / 100.0), 2)
    return pd.DataFrame([d])

def _extract_shap_risk_factors(df_in: pd.DataFrame, top_k: int = 3) -> List[str]:
    """Trích xuất top nguyên nhân rủi ro dựa trên giá trị đóng góp SHAP thực tế"""
    Z = prep_s1.transform(df_in)
    sv = shap_explainer_s1(Z)
    contrib = pd.Series(sv.values[0], index=feature_names_s1).sort_values(ascending=False)
    
    factors = []
    for feat, val in contrib.items():
        if val > 0:
            factors.append(VN_RISK_MAP.get(feat, feat))
        if len(factors) >= top_k:
            break
    return factors if factors else ["Không ghi nhận yếu tố rủi ro nổi bật"]

# 3. Endpoints API
@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "version": "2.1", "stage1_loaded": True, "stage2_loaded": True, "features_count": 18}

@app.post("/predict/stage1")
def predict_stage1(profile: StudentProfile):
    """Phân loại học sinh có nguy cơ không đạt loại Khá (< 6.5 điểm) kèm giải thích SHAP"""
    df_in = _prepare_df(profile)
    prob = float(pipe_stage1.predict_proba(df_in)[:, 1][0])
    is_at_risk = int(prob >= OPTIMAL_THRESHOLD_STAGE1)
    
    # Trích xuất yếu tố rủi ro bằng SHAP
    risk_factors = _extract_shap_risk_factors(df_in, top_k=3) if is_at_risk else []
    
    return {
        "at_risk": is_at_risk,
        "risk_probability": round(prob, 4),
        "threshold_used": OPTIMAL_THRESHOLD_STAGE1,
        "status_label": "NGUY CƠ (Cần can thiệp)" if is_at_risk else "AN TOÀN",
        "top_risk_factors": risk_factors
    }

@app.post("/predict/stage2")
def predict_stage2(profile: StudentProfile):
    """Dự đoán điểm thi cuối kỳ (thang 10)"""
    df_in = _prepare_df(profile)
    pred_score = float(pipe_stage2.predict(df_in)[0])
    pred_score = max(0.0, min(10.0, pred_score))
    
    return {
        "predicted_score": round(pred_score, 2),
        "is_at_risk": bool(pred_score < 6.5),
        "scale": "0-10"
    }

@app.post("/recommend")
def recommend_plan(req: WhatIfRequest):
    """Mô phỏng kịch bản What-If: tìm lộ trình can thiệp hành vi tối ưu để đạt điểm mục tiêu"""
    profile = req.student
    target = req.target_score
    df_in = _prepare_df(profile)
    
    curr_score = float(pipe_stage2.predict(df_in)[0])
    curr_score = round(max(0.0, min(10.0, curr_score)), 2)
    
    if curr_score >= target:
        return {
            "status": "ALREADY_ACHIEVED",
            "current_pred": curr_score,
            "target_score": target,
            "message": "Học sinh đã đạt hoặc vượt điểm số mục tiêu kỳ vọng!"
        }
    
    base_dict = profile.model_dump()
    curr_h = base_dict['Hours_Studied']
    curr_att = base_dict['Attendance']
    curr_tut = base_dict['Tutoring_Sessions']
    
    candidates = []
    for delta_h in range(0, req.max_extra_hours + 1, 2):
        for delta_att in [0, 5, 10, 15]:
            for delta_tut in range(0, req.max_extra_tutoring + 1):
                new_h = min(40, curr_h + delta_h)
                new_att = min(100, curr_att + delta_att)
                new_tut = min(8, curr_tut + delta_tut)
                
                sim_row = base_dict.copy()
                sim_row['Hours_Studied'] = new_h
                sim_row['Attendance'] = new_att
                sim_row['Tutoring_Sessions'] = new_tut
                sim_row['effective_study_index'] = round(new_h * (new_att / 100.0), 2)
                
                sim_df = pd.DataFrame([sim_row])
                sim_pred = float(pipe_stage2.predict(sim_df)[0])
                
                if sim_pred >= target:
                    # Hàm chi phí nỗ lực
                    cost = delta_h * 1.0 + delta_att * 0.5 + delta_tut * 2.0
                    candidates.append({
                        "hours_added": delta_h,
                        "attendance_added": delta_att,
                        "tutoring_added": delta_tut,
                        "new_hours": new_h,
                        "new_attendance": new_att,
                        "new_tutoring": new_tut,
                        "predicted_score": round(sim_pred, 2),
                        "score_gain": round(sim_pred - curr_score, 2),
                        "effort_cost": cost
                    })
    
    if not candidates:
        return {
            "status": "UNREACHABLE",
            "current_pred": curr_score,
            "target_score": target,
            "message": "Mục tiêu quá cao so với giới hạn can thiệp an toàn. Khuyến nghị đặt mục tiêu từng bước."
        }
    
    candidates.sort(key=lambda x: (x['effort_cost'], -x['predicted_score']))
    best = candidates[0]
    
    return {
        "status": "SUCCESS",
        "current_pred": curr_score,
        "target_score": target,
        "recommended_plan": best
    }
```

---

## 5. Mẫu Tích hợp Backend NestJS (Gateway Service)

```typescript
// student-ml.dto.ts
import { IsInt, IsNumber, IsIn, Min, Max } from 'class-validator';

export class StudentProfileDto {
  @IsInt() @Min(1) @Max(40) Hours_Studied: number;
  @IsInt() @Min(60) @Max(100) Attendance: number;
  @IsNumber() @Min(4.0) @Max(10.0) Sleep_Hours: number;
  @IsInt() @Min(0) @Max(8) Tutoring_Sessions: number;
  @IsNumber() @Min(0.0) @Max(10.0) Previous_Scores: number;
  @IsInt() @Min(0) @Max(6) Physical_Activity: number;
  @IsIn(['Low', 'Medium', 'High']) Parental_Involvement: string;
  @IsIn(['Low', 'Medium', 'High']) Access_to_Resources: string;
  @IsIn(['Low', 'Medium', 'High']) Motivation_Level: string;
  @IsIn(['Low', 'Medium', 'High']) Family_Income: string;
  @IsIn(['Low', 'Medium', 'High']) Teacher_Quality: string;
  @IsIn(['High School', 'College', 'Postgraduate']) Parental_Education_Level: string;
  @IsIn(['Yes', 'No']) Extracurricular_Activities: string;
  @IsIn(['Yes', 'No']) Internet_Access: string;
  @IsIn(['Positive', 'Neutral', 'Negative']) Peer_Influence: string;
  @IsIn(['Yes', 'No']) Learning_Disabilities: string;
  @IsIn(['Near', 'Moderate', 'Far']) Distance_from_Home: string;
}
```

---

## 6. Mẫu Giao diện Frontend Next.js (React Component)

```tsx
// components/WhatIfSimulator.tsx
import React, { useState } from 'react';

export default function WhatIfSimulator({ studentProfile }: { studentProfile: any }) {
  const [hours, setHours] = useState(studentProfile.Hours_Studied);
  const [attendance, setAttendance] = useState(studentProfile.Attendance);
  const [tutoring, setTutoring] = useState(studentProfile.Tutoring_Sessions);
  const [prediction, setPrediction] = useState<number | null>(null);

  const simulate = async () => {
    const payload = {
      ...studentProfile,
      Hours_Studied: hours,
      Attendance: attendance,
      Tutoring_Sessions: tutoring
    };
    const res = await fetch('http://localhost:8000/predict/stage2', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    setPrediction(data.predicted_score);
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-lg max-w-xl">
      <h2 className="text-xl font-bold mb-4 text-indigo-700">Mô phỏng Kịch bản What-If</h2>
      
      <div className="mb-4">
        <label className="block text-sm font-semibold">Giờ tự học mỗi tuần: {hours}h</label>
        <input type="range" min="1" max="40" value={hours} onChange={e => setHours(+e.target.value)} className="w-full" />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-semibold">Tỷ lệ chuyên cần: {attendance}%</label>
        <input type="range" min="60" max="100" value={attendance} onChange={e => setAttendance(+e.target.value)} className="w-full" />
      </div>

      <div className="mb-4">
        <label className="block text-sm font-semibold">Số buổi học thêm: {tutoring} buổi/tháng</label>
        <input type="range" min="0" max="8" value={tutoring} onChange={e => setTutoring(+e.target.value)} className="w-full" />
      </div>

      <button onClick={simulate} className="w-full py-2 bg-indigo-600 text-white font-bold rounded-lg hover:bg-indigo-700">
        Mô phỏng Điểm thi Mới
      </button>

      {prediction !== null && (
        <div className={`mt-4 p-4 rounded-lg font-bold text-center ${prediction >= 6.5 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
          Điểm dự kiến: {prediction} / 10 — {prediction >= 6.5 ? 'ĐẠT CHUẨN LOẠI KHÁ 🎉' : 'NGUY CƠ CHƯA ĐẠT KHÁ ⚠️'}
        </div>
      )}
    </div>
  );
}
```
