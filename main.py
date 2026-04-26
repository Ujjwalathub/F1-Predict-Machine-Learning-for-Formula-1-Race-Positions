import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("1. Loading datasets...")
train = pd.read_csv(r"E:\Kaggle\train.csv")
test = pd.read_csv(r"E:\Kaggle\test.csv")

# ==========================================
# Advanced Feature Engineering
# ==========================================
print("2. Engineering advanced features...")

# 1. First, we calculate historical statistics from the TRAIN dataset only.
# We do this before applying to the test set to prevent "target leakage" (peeking into the future).

# Define what we consider a "DNF or Poor Finish" proxy (Finishing 15th or worse)
train['is_dnf_proxy'] = (train['finishing_position'] >= 15).astype(int)

# --- FEATURE A: Circuit Volatility ---
# How much does the starting grid matter here? 
# We calculate the correlation between starting position and finishing position at each track.
# High correlation (~0.8) = Predictable track (like Monaco). 
# Low correlation (~0.3) = Chaotic track (high volatility).
circuit_volatility = train.groupby('circuitId').apply(
    lambda x: x['grid'].corr(x['finishing_position'])
).reset_index(name='circuit_predictability_score')

# Fill any weird NaN values with the overall average
avg_correlation = circuit_volatility['circuit_predictability_score'].mean()
circuit_volatility['circuit_predictability_score'] = circuit_volatility['circuit_predictability_score'].fillna(avg_correlation)

# --- FEATURE B: Constructor Reliability ---
# What percentage of the time does this team's car finish 15th or worse?
constructor_reliability = train.groupby('constructorId')['is_dnf_proxy'].mean().reset_index(name='constructor_dnf_rate')

# --- FEATURE C: Driver Reliability ---
# What percentage of the time does this driver finish 15th or worse?
driver_reliability = train.groupby('driverId')['is_dnf_proxy'].mean().reset_index(name='driver_dnf_rate')

# 2. Now, we create our updated feature engineering function
def engineer_advanced_features(df, circuit_stats, constructor_stats, driver_stats):
    df = df.copy()
    
    # Time gap between Q1 and Best Qualifying Time (captures sandbagging/track evolution)
    df['q1_best_gap'] = df['q1_ms'] - df['best_qual_ms']
    # Did the driver make it to Q3? (1 if yes, 0 if no)
    df['made_it_to_q3'] = df['q3_ms'].notnull().astype(int)
    
    # Merge our new historical reliability and volatility scores
    df = df.merge(circuit_stats, on='circuitId', how='left')
    df = df.merge(constructor_stats, on='constructorId', how='left')
    df = df.merge(driver_stats, on='driverId', how='left')
    
    # If a driver/team/circuit is brand new in 2023/2024 and has no history, 
    # fill their missing score with the average to prevent NaN errors in XGBoost
    df['circuit_predictability_score'] = df['circuit_predictability_score'].fillna(circuit_stats['circuit_predictability_score'].mean())
    df['constructor_dnf_rate'] = df['constructor_dnf_rate'].fillna(constructor_stats['constructor_dnf_rate'].mean())
    df['driver_dnf_rate'] = df['driver_dnf_rate'].fillna(driver_stats['driver_dnf_rate'].mean())
    
    return df

# 3. Apply the function to both Train and Test
print("Applying advanced historical features...")
train = engineer_advanced_features(train, circuit_volatility, constructor_reliability, driver_reliability)
test = engineer_advanced_features(test, circuit_volatility, constructor_reliability, driver_reliability)

# Define target and the features to use
target = 'finishing_position'
# Drop identifying columns, the target column, and the proxy column to prevent data leakage
drop_cols = ['id', 'raceId', 'driverId', 'constructorId', target, 'is_dnf_proxy']
features = [col for col in train.columns if col not in drop_cols]

X = train[features]
y = train[target]

# ==========================================
# Local Validation Setup
# ==========================================
print("3. Splitting data for local validation...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Model Training
# ==========================================
print("4. Training XGBoost Model...")
xgb_model = XGBRegressor(
    n_estimators=150, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42,
    objective='reg:absoluteerror', # Optimize specifically for MAE (the hackathon metric)
    tree_method='hist'             # Speeds up training significantly
)
xgb_model.fit(X_train, y_train)

# ==========================================
# Evaluation Metrics
# ==========================================
val_preds = xgb_model.predict(X_val)
val_mae = mean_absolute_error(y_val, val_preds)
val_r2 = r2_score(y_val, val_preds)

print("\n" + "-" * 40)
print(f"Validation MAE (Lower is better): {val_mae:.4f}")
print(f"Validation R-squared (Higher is better): {val_r2:.4f}")
print("-" * 40 + "\n")

# ==========================================
# Matplotlib Visualizations
# ==========================================
print("5. Generating performance visualizations...")

# Plot A: Actual vs Predicted Finishing Positions
plt.figure(figsize=(10, 6))
plt.scatter(y_val, val_preds, alpha=0.3, color='#1f77b4', edgecolors='w', linewidth=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title(f'XGBoost: Actual vs Predicted Finishing Position\nMAE: {val_mae:.2f} | $R^2$: {val_r2:.3f}', fontsize=14)
plt.xlabel('Actual Position', fontsize=12)
plt.ylabel('Predicted Position', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show() 


# Plot B: Feature Importances
plt.figure(figsize=(10, 8))
importances = pd.Series(xgb_model.feature_importances_, index=features).sort_values(ascending=True)
importances.tail(15).plot(kind='barh', color='#ff7f0e')
plt.title('XGBoost Top 15 Feature Importances', fontsize=14)
plt.xlabel('Relative Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()


# ==========================================
# Final Submission Generation
# ==========================================
print("6. Retraining on the FULL dataset for maximum accuracy...")
# We retrain on X and y (100% of train.csv) to give the final model all possible data
xgb_model.fit(X, y)

print("7. Predicting 2023-2024 test data...")
test_X = test[features]
test_preds = xgb_model.predict(test_X)

print("8. Formatting and saving final submission file...")
# Format MUST match sample_submission.csv (id and finishing_position only)
submission = pd.DataFrame({
    'id': test['id'],
    'finishing_position': test_preds
})

# Save to CSV
submission.to_csv('submission_xgboost_final.csv', index=False)
print("\nSUCCESS! Saved hackathon entry to 'submission_xgboost_final.csv'. You can now upload this file.")