from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Initialize the FastAPI app
app = FastAPI(title="Player Valuation API")

# Load the trained XGBoost model
try:
    model = joblib.load('valuation_model.joblib')
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    model = None
    print("❌ Model file not found. Please run the training notebook first.")

# Define the structure of the input data using Pydantic
class PlayerStats(BaseModel):
    age: float
    height: float
    appearance: int
    goals: float
    assists: float
    yellow_cards: float
    red_cards: float
    goals_conceded: float
    clean_sheets: float
    minutes_played: int
    days_injured: int
    games_injured: int

# Define the prediction endpoint
@app.post("/predict")
def predict_value(stats: PlayerStats):
    if model is None:
        return {"error": "Model not loaded."}

    # Convert the input data to a pandas DataFrame
    data = pd.DataFrame([stats.dict()])
    
    # Reorder columns to match the model's training order
    # Note: We need to handle the column names with spaces
    data.rename(columns={
        'yellow_cards': 'yellow cards',
        'red_cards': 'red cards',
        'goals_conceded': 'goals conceded',
        'clean_sheets': 'clean sheets',
        'minutes_played': 'minutes played',
        'days_injured': 'days_injured',
        'games_injured': 'games_injured'
    }, inplace=True)
    
    # The list of features the model was trained on
    features_ordered = [
        'age', 'height', 'appearance', 'goals', 'assists', 'yellow cards',
        'red cards', 'goals conceded', 'clean sheets', 'minutes played',
        'days_injured', 'games_injured'
    ]
    
    data = data[features_ordered]

    # Make the prediction
    prediction = model.predict(data)[0]

    return {"predicted_market_value": round(prediction)}

# A simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"status": "Player Valuation API is running."}