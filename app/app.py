from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List


# Load your model and vectorizer
lr_model = joblib.load('../models/lr_model.joblib')
tfidf_vectorizer = joblib.load('../models/tfidf_vectorizer.joblib')

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Define the request model
class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Prediction!"}

@app.post("/predict")
async def predict(input: TextInput):
    preprocessed_text = input.text.lower().strip()
    new_text_vectorized = tfidf_vectorizer.transform([preprocessed_text])
    prediction = lr_model.predict(new_text_vectorized)
    mental_health_status = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar',
                            'Personality disorder'][prediction[0]]
    return {"prediction": mental_health_status}

# Run the app with: uvicorn app:app --reload
