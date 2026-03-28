import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Load FastF1 2026 Australian GP race session (Round 1)
session_2026 = fastf1.get_session(2026, 1, "R")
session_2026.load()

# Extract lap times from 2026 session
laps_2026 = session_2026.laps[["Driver", "LapTime"]].copy()
laps_2026.dropna(subset=["LapTime"], inplace=True)
laps_2026["LapTime (s)"] = laps_2026["LapTime"].dt.total_seconds()

# 2026 Qualifying Data - Australian GP (Melbourne) - Round 1
qualifying_2026 = pd.DataFrame({
    "Driver": [
        "George Russell", "Kimi Antonelli", "Isack Hadjar",
        "Charles Leclerc", "Oscar Piastri", "Lando Norris",
        "Lewis Hamilton", "Liam Lawson", "Arvid Lindblad",
        "Gabriel Bortoleto"
    ],
    "QualifyingTime (s)": [
        78.518,        # P1 - Pole
        78.811,        # P2 +0.293s
        79.303,        # P3 +0.785s
        79.327,        # P4 +0.809s
        79.380,        # P5 +0.862s
        79.475,        # P6 +0.957s
        79.478,        # P7 +0.960s
        79.994,        # P8 +1.476s
        81.247,        # P9 +2.729s
        float('nan')   # P10 - No time set
    ]
})

# 2026 driver name → FastF1 3-letter code mapping
driver_mapping = {
    "George Russell":    "RUS",
    "Kimi Antonelli":    "ANT",
    "Isack Hadjar":      "HAD",
    "Charles Leclerc":   "LEC",
    "Oscar Piastri":     "PIA",
    "Lando Norris":      "NOR",
    "Lewis Hamilton":    "HAM",
    "Liam Lawson":       "LAW",
    "Arvid Lindblad":    "LIN",
    "Gabriel Bortoleto": "BOR"
}

# Map driver codes on 2026 qualifying data
qualifying_2026["DriverCode"] = qualifying_2026["Driver"].map(driver_mapping)

# Drop rows with no qualifying time (e.g. Bortoleto)
qualifying_2026_clean = qualifying_2026.dropna(subset=["QualifyingTime (s)"]).copy()

# Merge 2026 Qualifying Data with 2026 Race Data
merged_data = qualifying_2026_clean.merge(laps_2026, left_on="DriverCode", right_on="Driver")

# Use only "QualifyingTime (s)" as a feature
X = merged_data[["QualifyingTime (s)"]]
y = merged_data["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Check data sources!")

# Train Gradient Boosting Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2026 qualifying times
predicted_lap_times = model.predict(qualifying_2026_clean[["QualifyingTime (s)"]])
qualifying_2026_clean["PredictedRaceTime (s)"] = predicted_lap_times

# ✅ Rank and reset index starting from 1
qualifying_2026_clean = qualifying_2026_clean.sort_values(by="PredictedRaceTime (s)").reset_index(drop=True)
qualifying_2026_clean.index += 1

# ✅ Round times
qualifying_2026_clean["QualifyingTime (s)"] = qualifying_2026_clean["QualifyingTime (s)"].round(3)
qualifying_2026_clean["PredictedRaceTime (s)"] = qualifying_2026_clean["PredictedRaceTime (s)"].round(3)

# ✅ Add medal/position labels
def add_medal(pos):
    if pos == 1: return "🥇"
    if pos == 2: return "🥈"
    if pos == 3: return "🥉"
    return f"P{pos} "

qualifying_2026_clean["Position"] = qualifying_2026_clean.index.map(add_medal)

# ✅ Print formatted results
print("\n" + "="*60)
print("🏁   PREDICTED 2026 AUSTRALIAN GP RACE RESULT   🏁")
print("="*60)
print(f"{'Pos':<6} {'Driver':<22} {'Qual Time':>10} {'Race Time':>10}")
print("-"*60)
for idx, row in qualifying_2026_clean.iterrows():
    print(f"{row['Position']:<6} {row['Driver']:<22} {row['QualifyingTime (s)']:>9.3f}s {row['PredictedRaceTime (s)']:>9.3f}s")
print("="*60)

# Evaluate Model
y_pred = model.predict(X_test)
print(f"🔍 Model MAE: {mean_absolute_error(y_test, y_pred):.2f} seconds")
print("="*60)
