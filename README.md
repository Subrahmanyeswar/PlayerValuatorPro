<div align="center">

⚽ PlayerValuator Pro

AI-Powered Football Player Valuation System

<p align="center">
<a href="#-project-overview">Overview</a> •
<a href="#-key-features">Features</a> •
<a href="#-tech-stack">Tech Stack</a> •
<a href="#-installation-guide">Installation</a> •
<a href="#-model-performance">Performance</a>
</p>

</div>

📖 Project Overview

PlayerValuator Pro is an advanced machine learning platform designed to estimate the market value of football players with high precision. By moving beyond simple linear regression, this system utilizes a hybrid ensemble approach combining the structured data handling of XGBoost with the deep learning sequence capabilities of LSTMs (Long Short-Term Memory) networks.

Whether you are a scout, analyst, or football enthusiast, PlayerValuator Pro provides data-driven insights into player valuations based on performance metrics like goals, assists, minutes played, and disciplinary records.

🚀 Key Features

Feature

Description

🤖 Hybrid AI Engine

Combines Gradient Boosting and Deep Learning to capture both linear and non-linear patterns in player data.

📊 Interactive Dashboard

A fully responsive Streamlit web app that allows users to input stats and get instant valuations.

🔌 API-First Design

Includes a robust FastAPI backend (api.py) for serving predictions to external mobile or web apps.

⚖️ Smart Ensemble

Uses a weighted averaging system to balance predictions, achieving higher accuracy than individual models.

📈 Rich Analytics

Generates detailed visual reports including Error Distribution Heatmaps, Residual Plots, and Feature Importance charts.

🛠 Tech Stack

The project is built using a modern data science stack:

Component

Technologies Used

Core Language

Python 3.9+

Machine Learning

XGBoost, Scikit-Learn, Joblib

Deep Learning

TensorFlow (Keras), LSTM Neural Networks

Web Framework

Streamlit (Frontend), FastAPI (Backend)

Data Processing

Pandas, NumPy

Visualization

Plotly Interactive Graphs, Matplotlib, Seaborn

📦 Installation Guide

Follow these steps to set up the project locally.

1. Clone the Repository

git clone [https://github.com/Subrahmanyeswar/PlayerValuatorPro.git](https://github.com/Subrahmanyeswar/PlayerValuatorPro.git)
cd PlayerValuatorPro


2. Set Up Environment

It is recommended to use a virtual environment.

# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


💻 Usage

Option 1: Run the Web Dashboard

This launches the interactive UI where you can explore data and test the predictor.

streamlit run app.py


Option 2: Run the API Server

This starts the backend server for API requests.

uvicorn api:app --reload


API Documentation will be available at: http://127.0.0.1:8000/docs

📊 Model Performance

We trained our models on a dataset of 10,000+ player records. The ensemble approach proved to be the most effective.

Model Architecture

RMSE (Root Mean Square Error)

R² Score (Accuracy)

XGBoost Regressor

€7.4M

0.89

LSTM Neural Network

€8.5M

0.85

🏆 Ensemble (Hybrid)

€6.8M

0.92

The Ensemble model reduces the error margin by combining the strengths of both tree-based and neural network architectures.

📂 Project Structure

PlayerValuatorPro/
├── 📂 notebooks/               # Jupyter Notebooks for training
│   ├── 1_Data_Exploration.ipynb
│   ├── 2_Model_Training.ipynb
│   └── ...
├── 📂 models/                  # Saved Model Files
│   ├── valuation_model.joblib
│   └── lstm_model.h5
├── app.py                      # Main Streamlit Application
├── api.py                      # FastAPI Backend
├── final_data.csv              # Processed Dataset
├── requirements.txt            # Project Dependencies
└── README.md                   # Project Documentation


🤝 Contributing

Contributions are always welcome!

Fork the Project

Create your Feature Branch (git checkout -b feature/NewFeature)

Commit your Changes (git commit -m 'Add some NewFeature')

Push to the Branch (git push origin feature/NewFeature)

Open a Pull Request

<div align="center">

Developed by Subrahmanyeswar

</div>
