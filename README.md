# 🏎️ F1-Predict: Machine Learning for Formula 1

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-lightgrey)

## 📌 Overview
Formula 1 is one of the most data-rich sports in the world, yet predicting race outcomes remains incredibly difficult due to the high variance of crashes, mechanical failures, and weather. 

This project aims to predict the exact integer **finishing position (1-20)** of F1 drivers for the 2023 and 2024 seasons using a dataset spanning 74 years of F1 history (1950–2022). The model uses an **XGBoost Regressor** heavily optimized for Mean Absolute Error (MAE), combined with advanced feature engineering to quantify the "unpredictable" chaos of motorsport.

## 📊 The Dataset
The dataset contains 25,840 historical rows and 26 raw features, including:
* **Track Characteristics:** Track length, number of turns, altitude.
* **Qualifying Data:** Q1, Q2, Q3 times, and starting grid position.
* **Championship Context:** Previous constructor points, rolling average finishing positions, and driver ages.

## 🛠️ Advanced Feature Engineering
To push the model's accuracy and capture the chaotic nature of racing, several custom metrics were engineered from the raw data:
* **Circuit Volatility Score (`circuit_predictability_score`):** Calculates the historical correlation between grid starting position and final finishing position for every specific circuit. (e.g., Monaco is highly predictable; Baku is highly chaotic).
* **Constructor Reliability (`constructor_dnf_rate`):** A rolling historical percentage of how often a specific team's car breaks down or finishes 15th or worse.
* **Driver Reliability (`driver_dnf_rate`):** Punishes crash-prone drivers mathematically by tracking their historical rate of poor finishes.
* **Sandbagging Proxy (`q1_best_gap`):** The time gap between Q1 and the best overall qualifying time to capture track evolution and driver pace management.

## 🚀 Model Performance
The model utilizes **XGBoost** (`objective='reg:absoluteerror'`) to directly optimize for the evaluation metric. 

* **Validation MAE:** `~4.56`
* **Validation R-squared ($R^2$):** `~0.40`

*Note on F1 Analytics: An $R^2$ of 0.40 is highly competitive in Formula 1 position forecasting. It indicates the model successfully captures ~40% of the variance using strictly pre-race data, with the remaining variance attributed to in-race unpredictability (collisions, safety cars, mid-race rain, and mechanical DNF events).*

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/F1-Predict.git](https://github.com/yourusername/F1-Predict.git)
   cd F1-Predict
   Install dependencies:

Bash
pip install pandas numpy xgboost scikit-learn matplotlib
Run the prediction pipeline:
Ensure train.csv and test.csv are in the root directory, then execute:

Bash
python f1_predictor.py
Outputs:

Generates a submission_xgboost_final.csv file formatted for hackathon scoring.

Outputs two Matplotlib visualizations: xgb_actual_vs_predicted.png and xgb_feature_importance.png.

📈 Visualizations
When you run the script, the model generates feature importance charts. The top predictive features are heavily dominated by:

grid (Starting Position)

prev_constructor_points (Vehicle Pace / Dominance)

Custom reliability and volatility metrics.

🤝 Future Improvements
Era Filtering: Truncate data from the 1950s-1980s where mechanical breakdown rates were drastically higher than the modern hybrid era.

Ensemble Modeling: Combine XGBoost with LightGBM to smooth out edge-case predictions.
