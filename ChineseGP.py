import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# 1. 2026 CHINA GP CONTEXT (ROUND 2)
weather_data = {"RainProbability": 0.00, "Temperature": 19.0} 
constructor_pts = {"Mercedes": 55, "Ferrari": 40, "McLaren": 18, "Red Bull": 8, "Haas": 7, "RB": 6, "Audi": 2}

# Added: TyreDeg (scale 1-10, where 10 is high wear)
# Added: EnergyHarvest (Efficiency of the 350kW ERS system on the 1.2km straight)
china_data = [
    {"Driver": "Kimi Antonelli", "Team": "Mercedes", "UltimateLap_S": 91.880, "ChinaGapFromPole_S": 0.000, "ChinaGrid": 1, "AustraliaGrid": 2, "RacePace_CleanAir": 95.1, "TyreDeg": 3.2, "EnergyHarvest": 9.8, "ActualLapTime": 96.064},
    {"Driver": "George Russell", "Team": "Mercedes", "UltimateLap_S": 92.010, "ChinaGapFromPole_S": 0.222, "ChinaGrid": 2, "AustraliaGrid": 1, "RacePace_CleanAir": 95.3, "TyreDeg": 3.4, "EnergyHarvest": 9.7, "ActualLapTime": 96.286},
    {"Driver": "Lewis Hamilton", "Team": "Ferrari", "UltimateLap_S": 92.215, "ChinaGapFromPole_S": 0.351, "ChinaGrid": 3, "AustraliaGrid": 7, "RacePace_CleanAir": 95.8, "TyreDeg": 2.8, "EnergyHarvest": 9.2, "ActualLapTime": 96.415},
    {"Driver": "Charles Leclerc", "Team": "Ferrari", "UltimateLap_S": 92.200, "ChinaGapFromPole_S": 0.364, "ChinaGrid": 4, "AustraliaGrid": 4, "RacePace_CleanAir": 95.7, "TyreDeg": 4.1, "EnergyHarvest": 9.3, "ActualLapTime": 96.428},
    {"Driver": "Oscar Piastri", "Team": "McLaren", "UltimateLap_S": 92.350, "ChinaGapFromPole_S": 0.486, "ChinaGrid": 5, "AustraliaGrid": 5, "RacePace_CleanAir": 96.1, "TyreDeg": 3.9, "EnergyHarvest": 8.8, "ActualLapTime": 96.550},
    {"Driver": "Max Verstappen", "Team": "Red Bull", "UltimateLap_S": 92.810, "ChinaGapFromPole_S": 0.938, "ChinaGrid": 8, "AustraliaGrid": 20, "RacePace_CleanAir": 95.6, "TyreDeg": 4.5, "EnergyHarvest": 8.5, "ActualLapTime": 97.002},
]

df = pd.DataFrame(china_data)
df["TeamScore"] = df["Team"].map(constructor_pts).fillna(0)
df["RainProbability"] = weather_data["RainProbability"]
df["Temperature"] = weather_data["Temperature"]

# 2. ENHANCED FEATURE SET
features = [
    "UltimateLap_S", "ChinaGapFromPole_S", "ChinaGrid", 
    "AustraliaGrid", "TeamScore", "RacePace_CleanAir",
    "TyreDeg", "EnergyHarvest"
]

X = df[features]
y = df["ActualLapTime"]

# LOO Validation
loo = LeaveOneOut()
errors = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=39)
    model.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, model.predict(X_test)))

# 3. FINAL RANKING
model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

print("\n" + "="*85)
print("🏁  2026 CHINESE GP PREDICTION 🏁")
print("="*85)
print(f"{'Pos':<6} {'Driver':<18} {'Tyre Deg':<10} {'ERS Eff':<10} {'Clean Pace':<12} {'Pred Time'}")
print("-" * 85)

for idx, row in df.iterrows():
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx} ")
    print(f"{medal:<6} {row['Driver']:<18} {row['TyreDeg']:>8.1f}  {row['EnergyHarvest']:>9}%  {row['RacePace_CleanAir']:>12.1f}s {row['PredictedRaceTime']:>10.3f}s")

print("="*85)
print(f"🔍 Leave-One-Out MAE: {np.mean(errors):.4f} seconds")
print("="*85)