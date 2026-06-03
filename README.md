# Airline Flight Status Prediction & ML Models

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter" alt="Jupyter">
</p>

## 📋 Overview

This repository features a comprehensive Data Science and Machine Learning workflow for predicting airline flight statuses (On Time, Delayed, Cancelled). It includes extensive Exploratory Data Analysis (EDA) in Jupyter Notebooks and an interactive web application built with Streamlit.

## 🎯 Key Features

- **Interactive Web Interface:** A Streamlit-based UI (`app.py`) allowing users to input passenger and flight details to get real-time status predictions.
- **Machine Learning Pipeline:** Utilizes **Naive Bayes (GaussianNB)** algorithm for classification tasks.
- **Feature Engineering & Selection:** Applies `MinMaxScaler` for normalization and **Chi-Square Test (`SelectKBest`)** to automatically select the most impactful features for the model.
- **Data Visualization:** Built-in dynamic charts displaying probability distributions of prediction outcomes.

## 🏗️ Technical Stack

* **Language:** Python
* **Web Framework:** Streamlit
* **ML Library:** Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Environment:** Jupyter Notebook

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. It is recommended to use a virtual environment.

```bash
pip install streamlit pandas numpy scikit-learn
```

### Running the Application

1. Clone the repository:
   ```bash
   git clone https://github.com/tahaemree/machine_learning_project.git
   cd machine_learning_project
   ```
2. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

## 📈 Model Information

The core model uses probability theory (Naive Bayes) to classify flights based on complex passenger and flight metrics. Categorical features like Nationality, Airport Continent, and Travel Class are processed using One-Hot Encoding to maximize model accuracy.

## 📄 License

This software is open-sourced under the **MIT License**.

You are free to use, modify, and distribute this software, provided that the original copyright and permission notice are included. Please see the [LICENSE](LICENSE) file for complete details.
