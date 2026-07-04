import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier

# Инициализация приложения FastAPI
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Production ML Microservice for predicting bank customer churn using CatBoost & Pydantic.",
    version="1.0.0"
)

# Загрузка обученной скомпилированной модели CatBoost
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'catboost_model.cbm')

model = CatBoostClassifier()
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)
else:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run src/train.py first!")

# Pydantic схема валидации входных данных
class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, example=650)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Female")
    Age: int = Field(..., ge=18, le=100, example=42)
    Tenure: int = Field(..., ge=0, le=10, example=5)
    Balance: float = Field(..., ge=0.0, example=75000.0)
    NumOfProducts: int = Field(..., ge=1, le=4, example=2)
    HasCrCard: int = Field(..., ge=0, le=1, example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, example=1)
    EstimatedSalary: float = Field(..., ge=0.0, example=50000.0)

@app.get("/")
def root():
    return {"message": "Bank Churn Prediction API is running. Go to /docs for interactive Swagger UI."}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # Превращаем данные Pydantic в DataFrame из 1 строки
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Предсказание класса и вероятностей
        churn_prediction = int(model.predict(input_df)[0])
        churn_probability = float(model.predict_proba(input_df)[0][1])
        
        return {
            "churn_prediction": churn_prediction,
            "churn_probability": round(churn_probability, 4),
            "status": "High Churn Risk" if churn_prediction == 1 else "Loyal Customer"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
