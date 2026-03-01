# ⚽ PlayerValuator Pro v2.0

Advanced AI-Powered Football Player Valuation Platform using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

- 🤖 **AI-Powered Predictions** - XGBoost & LSTM hybrid models
- 📊 **Rich Analytics** - Interactive visualizations with Plotly
- ⚡ **Real-Time Performance** - FastAPI backend for instant predictions
- 🎯 **86% Model Accuracy** - Trained on 10,000+ player records
- 🔮 **Ensemble System** - Combines multiple ML techniques

## 📸 Screenshots

### Home Dashboard
<img width="1919" height="712" alt="Screenshot 2025-11-28 174922" src="https://github.com/user-attachments/assets/78f3b53a-1a1b-4125-94fc-fca5f5dfaaa7" />

### AI Predictor
<img width="1919" height="724" alt="Screenshot 2025-11-28 174957" src="https://github.com/user-attachments/assets/4562c830-bd75-4843-b5c7-29e51a36c1e8" />

## 🛠️ Technology Stack

- **Frontend:** Streamlit, Plotly, HTML/CSS
- **Backend:** FastAPI
- **ML Models:** XGBoost, LSTM (TensorFlow/Keras)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly, Seaborn, Matplotlib

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone [https://github.com/Subrahmanyeswar/PlayerValuatorPro.git](https://github.com/Subrahmanyeswar/PlayerValuatorPro.git)
cd PlayerValuatorPro



```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the training notebooks**
```bash
jupyter notebook
# Run notebooks in order: 1, 2, 3, 4
```

5. **Start the Streamlit app**
```bash
streamlit run app.py
```

6. **Start the FastAPI backend (optional)**
```bash
uvicorn api:app --reload
```

## 📊 Dataset

The system analyzes **10,754 player records** with features including:
- Age, Height, Position
- Goals, Assists, Appearances
- Minutes Played, Injury History
- Disciplinary Records
- Market Value

## 🧠 Model Architecture

### XGBoost Model
- **Type:** Gradient Boosting Regressor
- **Trees:** 1,000 estimators
- **Learning Rate:** 0.05
- **RMSE:** €7.4M
- **Training Samples:** 8,469

### LSTM Model (Optional)
- **Architecture:** 3-layer LSTM neural network
- **Sequence Length:** 5 time steps
- **Features:** 19 engineered features
- **Activation:** ReLU + Linear output

### Ensemble System
- **Method:** Weighted average
- **Optimization:** Minimized validation loss
- **Weights:** Dynamic based on performance

## 📁 Project Structure
```
PlayerValuatorPro/
├── app.py                          # Main Streamlit application
├── api.py                          # FastAPI backend
├── final_data.csv                  # Player dataset
├── valuation_model.joblib          # Trained XGBoost model
├── lstm_model.h5                   # Trained LSTM model (optional)
├── lstm_scaler_X.joblib           # LSTM input scaler
├── lstm_scaler_y.joblib           # LSTM output scaler
├── lstm_metadata.joblib           # LSTM configuration
├── ensemble_weights.joblib        # Ensemble weights
├── 1_Data_Exploration.ipynb       # Data analysis notebook
├── 2_Model_Training.ipynb         # XGBoost training
├── 3_LSTM_Training.ipynb          # LSTM training
├── 4_Ensemble_Model_Comparison.ipynb  # Model comparison
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎯 Usage

### Web Interface
1. Navigate to the **Home** page for system overview
2. Use **AI Predictor** to get player valuations
3. Compare models in **Model Comparison** dashboard
4. Explore data in **Visualization Gallery**
5. Review metrics in **Model Performance**

### API Endpoint
```python
import requests

payload = {
    "age": 25,
    "height": 180,
    "appearance": 30,
    "goals": 15,
    "assists": 8,
    # ... other features
}

response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
```

## 📈 Performance Metrics

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| XGBoost | €7.4M | €5.2M | 0.89 |
| LSTM | €8.5M | €6.0M | 0.85 |
| Ensemble | €6.8M | €4.8M | 0.92 |

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Player comparison feature
- [ ] Team analysis dashboard
- [ ] Transfer value predictions
- [ ] Mobile app development
- [ ] Multi-league support

## 👨‍💻 Author

**Your Name**
- GitHub: https://github.com/Subrahmanyeswar
- LinkedIn: www.linkedin.com/in/subrahmanyeswar-kolluru-914694293

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from football analytics databases
- Built with Streamlit and FastAPI
- ML models powered by XGBoost and TensorFlow

## ⭐ Star this repo if you found it helpful!

---

**Made with ❤️ and ⚽ by Subrahmanyeswar Kolluru**
```
