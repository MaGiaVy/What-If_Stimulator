# HANDOFF — Hướng dẫn Tích hợp Mô hình vào Ứng dụng (App Integration)

> **Dành cho:** Kỹ sư Backend (**NestJS & FastAPI**) và Kỹ sư Frontend (**Next.js / React**).  
> **Dự án:** Hệ thống Hỗ trợ Học tập Lớp 12 Việt Nam  
> **Đầu ra Machine Learning:** 2 file pipeline `.pkl` đã tích hợp đầy đủ quy trình tiền xử lý dữ liệu và thuật toán mô phỏng.

---

## 1. Danh mục Bàn giao & Tài nguyên

| File / Thư mục | Định dạng | Trách nhiệm sử dụng | Mô tả chức năng |
| :--- | :---: | :--- | :--- |
| `models/pipeline_stage1.pkl` | Binary (Joblib) | FastAPI Service | Pipeline Tầng 1: Phân loại nguy cơ học tập (`at_risk` < 6.5) |
| `models/pipeline_stage2_score.pkl` | Binary (Joblib) | FastAPI Service | Pipeline Tầng 2: Dự đoán điểm thi (0–10) & What-If Simulator |
| `models/model_metadata.json` | JSON | Backend & Frontend | Cấu hình siêu dữ liệu: danh sách đặc trưng, ngưỡng $T^* = 0.53$, chỉ số đánh giá |
| `docs/BAO_CAO_DO_AN.md` | Markdown | Cả nhóm | Báo cáo học thuật chi tiết để hiểu rõ bối cảnh và logic mô hình |

### Môi trường thực thi khuyến nghị (Python Backend ML Service)
Các pipeline được xuất xưởng từ Python 3.12. Yêu cầu cài đặt môi trường độc lập:
```bash
pip install scikit-learn==1.8.0 imbalanced-learn==0.14.2 pandas==3.0.2 numpy==1.26.4 joblib==1.5.3 fastapi uvicorn[standard] pydantic
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
│      - POST /predict/stage1                            │
│      - POST /predict/stage2                            │
│      - POST /recommend                                 │
└────────────────────────────────────────────────────────┘
```

---

## 3. Hợp đồng Dữ liệu Chi tiết (Data Contract)

Cả hai mô hình đều tiếp nhận **cùng một bộ thông tin học sinh** (20 đặc trưng đầu vào).

### 3.1 Bảng Đặc tả Thuộc tính Học sinh (20 trường)

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
| `School_Type` | string | Hoàn cảnh | `"Public"`, `"Private"` | Trường công lập hay tư thục |
| `Peer_Influence` | string | Hoàn cảnh | `"Positive"`, `"Neutral"`, `"Negative"` | Môi trường bạn bè xung quanh |
| `Learning_Disabilities` | string | Hoàn cảnh | `"Yes"`, `"No"` | Có khó khăn / rào cản nhận thức không |
| `Distance_from_Home` | string | Hoàn cảnh | `"Near"`, `"Moderate"`, `"Far"` | Khoảng cách từ nhà đến trường |
| `Gender` | string | Hoàn cảnh | `"Male"`, `"Female"` | Giới tính |

> **Quy tắc kỹ thuật sống còn:** Backend hoặc FastAPI **tự động tính toán** trường tương tác:
> `effective_study_index = round(Hours_Studied * (Attendance / 100.0), 2)` trước khi đưa vào mô hình. Frontend không cần gửi trường này.

---

## 4. Mã nguồn Dịch vụ FastAPI ML Hoàn chỉnh (`api.py`)

Backend team có thể lưu file này vào `service/api.py` và chạy ngay:

```python
# service/api.py
import os
import joblib
import pandas as pd
from typing import Literal, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="ML Microservice — He thong Ho tro Hoc tap Lop 12",
    version="2.0",
    description="API phan loai nguy co hoc tap va mo phong kich ban What-If cho GV va PH"
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
    School_Type: Literal['Public', 'Private'] = Field(..., example='Public')
    Peer_Influence: Literal['Positive', 'Neutral', 'Negative'] = Field(..., example='Neutral')
    Learning_Disabilities: Literal['Yes', 'No'] = Field(..., example='No')
    Distance_from_Home: Literal['Near', 'Moderate', 'Far'] = Field(..., example='Near')
    Gender: Literal['Male', 'Female'] = Field(..., example='Male')

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

# 3. Endpoints API
@app.get("/health")
def health_check():
    return {"status": "HEALTHY", "version": "2.0", "stage1_loaded": True, "stage2_loaded": True}

@app.post("/predict/stage1")
def predict_stage1(profile: StudentProfile):
    """Phân loại học sinh có nguy cơ không đạt loại Khá (< 6.5 điểm)"""
    df_in = _prepare_df(profile)
    prob = float(pipe_stage1.predict_proba(df_in)[:, 1][0])
    is_at_risk = int(prob >= OPTIMAL_THRESHOLD_STAGE1)
    
    # Xác định các yếu tố cảnh báo chính
    risk_factors = []
    if profile.Hours_Studied < 15: risk_factors.append("Thời gian tự học ít (< 15h/tuần)")
    if profile.Attendance < 80: risk_factors.append("Tỷ lệ chuyên cần thấp (< 80%)")
    if profile.Motivation_Level == 'Low': risk_factors.append("Động lực học tập thấp")
    if profile.Previous_Scores < 6.5: risk_factors.append("Điểm kỳ trước dưới chuẩn Khá")
    
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
                    # Ưu tiên tự học và đi học đủ trước khi tăng học thêm
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
            "status": "TARGET_TOO_HIGH",
            "current_pred": curr_score,
            "target_score": target,
            "message": "Mục tiêu quá cao so với trần nỗ lực tối đa có thể điều chỉnh trong 1 kỳ."
        }
    
    best = sorted(candidates, key=lambda x: (x['effort_cost'], -x['predicted_score']))[0]
    return {
        "status": "SUCCESS",
        "current_pred": curr_score,
        "target_score": target,
        "optimal_plan": best
    }
```

---

## 5. Hướng dẫn Tích hợp Phía NestJS (API Gateway)

NestJS đóng vai trò Gateway tiếp nhận request từ Frontend, kiểm tra Token xác thực, gọi sang FastAPI Microservice và lưu kết quả:

```typescript
// src/ml-service/ml.service.ts
import { Injectable, HttpService } from '@nestjs/common';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class MlService {
  private readonly mlBaseUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';

  constructor(private readonly httpService: HttpService) {}

  async checkRisk(studentProfile: any) {
    const response = await firstValueFrom(
      this.httpService.post(`${this.mlBaseUrl}/predict/stage1`, studentProfile)
    );
    return response.data;
  }

  async predictScore(studentProfile: any) {
    const response = await firstValueFrom(
      this.httpService.post(`${this.mlBaseUrl}/predict/stage2`, studentProfile)
    );
    return response.data;
  }

  async simulateWhatIf(studentProfile: any, targetScore: number) {
    const response = await firstValueFrom(
      this.httpService.post(`${this.mlBaseUrl}/recommend`, {
        student: studentProfile,
        target_score: targetScore,
      })
    );
    return response.data;
  }
}
```

---

## 6. Hướng dẫn Tích hợp Phía Next.js / React (Frontend UI)

Gợi ý cấu trúc Component giao diện What-If Simulator dành cho Giáo viên / Phụ huynh:

```tsx
// components/WhatIfSimulator.tsx
import React, { useState } from 'react';

export default function WhatIfSimulator({ studentProfile }: { studentProfile: any }) {
  const [targetScore, setTargetScore] = useState<number>(7.5);
  const [recommendation, setRecommendation] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleSimulate = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/gateway/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ student: studentProfile, target_score: targetScore })
      });
      const data = await res.json();
      setRecommendation(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-xl shadow-md border border-gray-100">
      <h3 className="text-xl font-bold text-gray-800 mb-2">Bộ Giả Lập Kịch Bản Học Tập (What-If)</h3>
      <p className="text-sm text-gray-500 mb-4">
        Tìm kiếm lộ trình nỗ lực tối ưu để cải thiện điểm số cho học sinh.
      </p>

      <div className="mb-4">
        <label className="block text-sm font-semibold text-gray-700 mb-1">
          Điểm mục tiêu kỳ vọng: <span className="text-blue-600 font-bold">{targetScore}/10</span>
        </label>
        <input
          type="range"
          min="5.0"
          max="10.0"
          step="0.1"
          value={targetScore}
          onChange={(e) => setTargetScore(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg cursor-pointer"
        />
      </div>

      <button
        onClick={handleSimulate}
        disabled={loading}
        className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition"
      >
        {loading ? 'Đang mô phỏng kịch bản...' : 'Mô Phỏng Lộ Trình Can Thiệp'}
      </button>

      {recommendation?.status === 'SUCCESS' && (
        <div className="mt-5 p-4 bg-emerald-50 border border-emerald-200 rounded-lg">
          <h4 className="font-bold text-emerald-800">Lộ Trình Đề Xuất Tối Ưu:</h4>
          <ul className="mt-2 text-sm text-emerald-700 space-y-1">
            <li>• Tăng tự học: <strong>+{recommendation.optimal_plan.hours_added} giờ/tuần</strong> (Đạt {recommendation.optimal_plan.new_hours}h)</li>
            <li>• Cải thiện chuyên cần: <strong>+{recommendation.optimal_plan.attendance_added}%</strong> (Đạt {recommendation.optimal_plan.new_attendance}%)</li>
            <li>• Học thêm: <strong>+{recommendation.optimal_plan.tutoring_added} buổi/tháng</strong></li>
            <li className="pt-2 text-base font-bold text-emerald-900">
              => Dự kiến đạt: {recommendation.optimal_plan.predicted_score}/10 (Tăng +{recommendation.optimal_plan.score_gain} điểm)
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

## 7. Các Lưu ý Kỹ thuật Quan trọng Khi Triển khai (Checklist)

1. **Không cho phép người dùng kéo slider các biến Hoàn cảnh:** Thu nhập gia đình, mức độ học vấn phụ huynh, điểm kỳ trước là các biến cố định phản ánh thực tế học sinh. Chỉ cho phép điều chỉnh `Hours_Studied`, `Attendance`, `Tutoring_Sessions`.
2. **Luôn dùng ngưỡng $T^* = 0.53$ ở Tầng 1:** Không dùng ngưỡng mặc định 0.50 của `predict()` để đảm bảo đạt Recall tối đa 97.71% trong việc phát hiện học sinh rủi ro.
3. **Độ trễ suy luận:** Cả hai pipeline Ridge và Logistic Regression đều có thời gian phản hồi cực nhanh ($< 5\text{ms}$/request), sẵn sàng chịu tải hàng chục nghìn lượt request/giây mà không cần GPU.
