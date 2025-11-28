⚽ PlayerValuator Pro

Advanced AI-Powered Football Player Valuation System

PlayerValuator Pro is a state-of-the-art machine learning application designed to predict the market value of football players with high precision. By leveraging a hybrid ensemble approach combining Gradient Boosting (XGBoost) and Deep Learning (LSTM), the system analyzes complex player metrics to generate accurate valuations.

🚀 Key Features

🧠 Hybrid AI Engine: Combines the structured data power of XGBoost with the sequence modeling capabilities of LSTM neural networks.

⚖️ Ensemble Logic: Uses a weighted averaging system to balance predictions and minimize error rates (RMSE).

📊 Interactive Dashboard: Built with Streamlit to visualize player stats, feature importance, and valuation ranges dynamically.

🔌 API First: Includes a FastAPI backend (api.py) for serving predictions to external applications.

📈 Rich Visualizations: Generates HTML reports for error distribution, residual analysis, and model agreement.

🛠️ Tech Stack

Core: Python 3.9+

Data Processing: Pandas, NumPy, Scikit-Learn

Machine Learning: XGBoost, TensorFlow (Keras/LSTM)

Visualization: Plotly, Matplotlib, Seaborn

Web Framework: Streamlit (Frontend), FastAPI (Backend)

📂 Project Structure

PlayerValuatorPro/
├── app.py                          # 📱 Main Streamlit Dashboard
├── api.py                          # 🔌 FastAPI Backend Server
├── 1_Data_Exploration.ipynb        # 🔍 Data Analysis & Cleaning
├── 2_Model_Training.ipynb          # 🤖 XGBoost Model Training
├── 3_LSTM_Training.ipynb           # 🧠 LSTM Neural Network Training
├── 4_Ensemble_Model_Comparison.ipynb # ⚖️ Ensemble Logic & Evaluation
├── final_data.csv                  # 💾 Processed Dataset
├── valuation_model.joblib          # 📦 Saved XGBoost Model
├── lstm_model.h5                   # 📦 Saved LSTM Model
├── ensemble_weights.joblib         # ⚖️ Optimized Weights
└── requirements.txt                # 📜 Dependencies


⚡ Installation & Usage

Clone the Repository

git clone [https://github.com/Subrahmanyeswar/PlayerValuatorPro.git](https://github.com/Subrahmanyeswar/PlayerValuatorPro.git)
cd PlayerValuatorPro


Install Dependencies

pip install -r requirements.txt


Run the Dashboard (UI)

streamlit run app.py


Run the API (Backend)

uvicorn api:app --reload


📊 Model Performance

The system was trained on over 10,000 player records.

Model

RMSE (Root Mean Squared Error)

R² Score

XGBoost

€7.4M

0.89

LSTM

€8.5M

0.85

Ensemble (Hybrid)

€6.8M

0.92

🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Author

Subrahmanyeswar

GitHub Profile
