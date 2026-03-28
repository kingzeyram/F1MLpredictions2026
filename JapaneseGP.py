import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

weather_data = {"RainProbability": 0.10, "Temperature": 16.0}

constructor_pts = {
    "Mercedes":     98,
    "Ferrari":      67,
    "McLaren":      18,
    "Haas":         17,
    "Red Bull":     12,
    "Racing Bulls": 12,
    "Alpine":       10,
    "Audi":          2,
}

japan_data = [
    {"Driver": "Kimi Antonelli",  "Team": "Mercedes",  "UltimateLap_S": 88.877, "JapanGapFromPole_S": 0.000, "JapanGrid": 1,  "ChinaGrid": 1,  "RacePace_CleanAir": 92.4, "TyreDeg": 3.8, "Suzuka_Confidence": 9.5, "ActualRaceTime": 97.21},
    {"Driver": "George Russell",  "Team": "Mercedes",  "UltimateLap_S": 89.147, "JapanGapFromPole_S": 0.270, "JapanGrid": 2,  "ChinaGrid": 2,  "RacePace_CleanAir": 92.6, "TyreDeg": 3.9, "Suzuka_Confidence": 9.2, "ActualRaceTime": 97.43},
    {"Driver": "Oscar Piastri",   "Team": "McLaren",   "UltimateLap_S": 89.227, "JapanGapFromPole_S": 0.350, "JapanGrid": 3,  "ChinaGrid": 5,  "RacePace_CleanAir": 93.1, "TyreDeg": 4.2, "Suzuka_Confidence": 8.8, "ActualRaceTime": 97.88},
    {"Driver": "Charles Leclerc", "Team": "Ferrari",   "UltimateLap_S": 89.270, "JapanGapFromPole_S": 0.393, "JapanGrid": 4,  "ChinaGrid": 4,  "RacePace_CleanAir": 93.0, "TyreDeg": 4.0, "Suzuka_Confidence": 9.0, "ActualRaceTime": 97.75},
    {"Driver": "Lando Norris",    "Team": "McLaren",   "UltimateLap_S": 89.380, "JapanGapFromPole_S": 0.503, "JapanGrid": 5,  "ChinaGrid": 20, "RacePace_CleanAir": 93.3, "TyreDeg": 4.3, "Suzuka_Confidence": 8.5, "ActualRaceTime": 98.05},
    {"Driver": "Lewis Hamilton",  "Team": "Ferrari",   "UltimateLap_S": 89.450, "JapanGapFromPole_S": 0.573, "JapanGrid": 6,  "ChinaGrid": 3,  "RacePace_CleanAir": 93.2, "TyreDeg": 3.7, "Suzuka_Confidence": 9.8, "ActualRaceTime": 97.90},
    {"Driver": "Max Verstappen",  "Team": "Red Bull",  "UltimateLap_S": 89.900, "JapanGapFromPole_S": 1.023, "JapanGrid": 11, "ChinaGrid": 8,  "RacePace_CleanAir": 93.8, "TyreDeg": 5.1, "Suzuka_Confidence": 8.0, "ActualRaceTime": 98.95},
]

df = pd.DataFrame(japan_data)
df["TeamScore"] = df["Team"].map(constructor_pts).fillna(0)
df["RainProbability"] = weather_data["RainProbability"]
df["Temperature"] = weather_data["Temperature"]

features = [
    "UltimateLap_S",
    "JapanGapFromPole_S",
    "JapanGrid",
    "ChinaGrid",
    "TeamScore",
    "RacePace_CleanAir",
    "TyreDeg",
    "Suzuka_Confidence",
]

X = df[features]
y = df["ActualRaceTime"]

loo = LeaveOneOut()
errors = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=39)
    model.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, model.predict(X_test)))

model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

print("\n" + "=" * 90)
print("🏁  2026 JAPANESE GP RACE PREDICTION — SUZUKA  🏁")
print("    Round 3 | March 29, 2026 | 53 Laps | 5.807 km")
print("=" * 90)
print(f"{'Pos':<6} {'Driver':<20} {'Grid':<6} {'Tyre Deg':<10} {'Suzuka Conf':<13} {'Race Pace':<12} {'Pred Time'}")
print("-" * 90)

for idx, row in df.iterrows():
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx}  ")
    print(
        f"{medal:<6} "
        f"{row['Driver']:<20} "
        f"P{int(row['JapanGrid']):<5} "
        f"{row['TyreDeg']:>8.1f}  "
        f"{row['Suzuka_Confidence']:>11}/10  "
        f"{row['RacePace_CleanAir']:>10.1f}s  "
        f"{row['PredictedRaceTime']:>9.3f}s"
    )

print("=" * 90)
print(f"📊 Leave-One-Out MAE: {np.mean(errors):.4f} seconds")
print("=" * 90)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n📈 FEATURE IMPORTANCE:")
print("-" * 45)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:<25} {row['Importance']:.4f}  {bar}")
print("=" * 90)