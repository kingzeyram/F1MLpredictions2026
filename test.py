import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# ─── RACE METADATA ────────────────────────────────────────────────────────────
# Round 4 | Miami International Autodrome | May 3, 2026 (RACE DAY)
# 57 Laps | 5.412 km 
# Updated Weather: High humidity, 72% chance of localized thunderstorms during GP.

weather_data = {
    "RainProbability": 0.72,   # Increased: Radar shows active cells moving toward Hard Rock Stadium
    "Temperature": 30.0,       # Humid heat
}

# ─── CONSTRUCTOR STANDINGS (Entering Miami) ───────────────────────────────────
constructor_pts = {
    "Mercedes":     135,
    "Ferrari":       90,
    "McLaren":       56,
    "Red Bull":      16,
    "Racing Bulls":   8,
    "Alpine":         8,
    "Audi":           4,
    "Williams":       2,
    "Haas":           0,
    "Aston Martin":   0,
}

# ─── DRIVER DATA (UPDATED FOR SUNDAY STARTING GRID) ───────────────────────────
# MiamiGrid: Official starting grid after Saturday Qualifying/Sprint outcomes.
# Miami_Confidence: Adjusted based on practice/sprint sector times.
miami_data = [
    {
        "Driver":              "Kimi Antonelli",
        "Team":                "Mercedes",
        "UltimateLap_S":       88.778,
        "JapanGapFromPole_S":  0.000,
        "JapanGrid":           1,
        "MiamiGrid":           1,               # Confirmed Pole
        "RacePace_CleanAir":   93.6,            # Strongest long-run pace in FP2
        "TyreDeg":             2.7,
        "Miami_Confidence":    9.0,             # Handling the pressure of a double-pole well
        "ActualRaceTime":      97.21,
    },
    {
        "Driver":              "George Russell",
        "Team":                "Mercedes",
        "UltimateLap_S":       89.076,
        "JapanGapFromPole_S":  0.298,
        "JapanGrid":           2,
        "MiamiGrid":           3,               # Dropped 1 spot to Norris in Quali
        "RacePace_CleanAir":   93.8,
        "TyreDeg":             2.8,
        "Miami_Confidence":    8.8,
        "ActualRaceTime":      97.43,
    },
    {
        "Driver":              "Lando Norris",
        "Team":                "McLaren",
        "UltimateLap_S":       89.409,
        "JapanGapFromPole_S":  0.631,
        "JapanGrid":           5,
        "MiamiGrid":           2,               # Exploded into P2; Miami specialist
        "RacePace_CleanAir":   94.1,
        "TyreDeg":             3.0,
        "Miami_Confidence":    9.8,             # Exceptionally high after 2024 win & Sprint pace
        "ActualRaceTime":      98.05,
    },
    {
        "Driver":              "Oscar Piastri",
        "Team":                "McLaren",
        "UltimateLap_S":       89.132,
        "JapanGapFromPole_S":  0.354,
        "JapanGrid":           3,
        "MiamiGrid":           4,
        "RacePace_CleanAir":   94.2,
        "TyreDeg":             3.1,
        "Miami_Confidence":    8.5,
        "ActualRaceTime":      97.88,
    },
    {
        "Driver":              "Charles Leclerc",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.405,
        "JapanGapFromPole_S":  0.627,
        "JapanGrid":           4,
        "MiamiGrid":           5,
        "RacePace_CleanAir":   94.0,
        "TyreDeg":             2.5,             # Best tyre conservation on the grid
        "Miami_Confidence":    8.9,
        "ActualRaceTime":      97.75,
    },
    {
        "Driver":              "Lewis Hamilton",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.567,
        "JapanGapFromPole_S":  0.789,
        "JapanGrid":           6,
        "MiamiGrid":           6,
        "RacePace_CleanAir":   94.4,
        "TyreDeg":             2.6,
        "Miami_Confidence":    8.4,
        "ActualRaceTime":      97.90,
    },
    {
        "Driver":              "Max Verstappen",
        "Team":                "Red Bull",
        "UltimateLap_S":       89.992,
        "JapanGapFromPole_S":  1.214,
        "JapanGrid":           11,
        "MiamiGrid":           7,               # Found pace in the low-speed sections
        "RacePace_CleanAir":   95.0,
        "TyreDeg":             3.8,
        "Miami_Confidence":    8.2,             # Improving, but car still 'nervous'
        "ActualRaceTime":      98.95,
    },
    {
        "Driver":              "Pierre Gasly",
        "Team":                "Alpine",
        "UltimateLap_S":       89.691,
        "JapanGapFromPole_S":  0.913,
        "JapanGrid":           7,
        "MiamiGrid":           8,
        "RacePace_CleanAir":   95.8,
        "TyreDeg":             3.5,
        "Miami_Confidence":    7.5,
        "ActualRaceTime":      99.10,
    },
    {
        "Driver":              "Isack Hadjar",
        "Team":                "Racing Bulls",
        "UltimateLap_S":       89.978,
        "JapanGapFromPole_S":  1.200,
        "JapanGrid":           8,
        "MiamiGrid":           9,
        "RacePace_CleanAir":   96.0,
        "TyreDeg":             3.6,
        "Miami_Confidence":    7.2,
        "ActualRaceTime":      99.50,
    },
    {
        "Driver":              "Gabriel Bortoleto",
        "Team":                "Audi",
        "UltimateLap_S":       90.274,
        "JapanGapFromPole_S":  1.496,
        "JapanGrid":           9,
        "MiamiGrid":           10,
        "RacePace_CleanAir":   96.3,
        "TyreDeg":             3.8,
        "Miami_Confidence":    7.0,
        "ActualRaceTime":      99.80,
    },
]

# ─── BUILD DATAFRAME ──────────────────────────────────────────────────────────
df = pd.DataFrame(miami_data)
df["TeamScore"]       = df["Team"].map(constructor_pts).fillna(0)
df["RainProbability"] = weather_data["RainProbability"]
df["Temperature"]     = weather_data["Temperature"]

features = [
    "UltimateLap_S", "JapanGapFromPole_S", "MiamiGrid", "JapanGrid",
    "TeamScore", "RacePace_CleanAir", "TyreDeg", "Miami_Confidence",
    "RainProbability", "Temperature",
]

X, y = df[features], df["ActualRaceTime"]

# ─── LEAVE-ONE-OUT CROSS VALIDATION ───────────────────────────────────────────
loo = LeaveOneOut()
errors = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=39)
    model.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, model.predict(X_test)))

# ─── FINAL PREDICTION ─────────────────────────────────────────────────────────
model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("🏁  2026 MIAMI GP: UPDATED RACE-DAY PREDICTION  🏁")
print(f"    RACE DAY WEATHER: {int(weather_data['RainProbability']*100)}% Storm Risk | {weather_data['Temperature']}°C")
print("    OFFICIAL STARTING GRID APPLIED")
print("=" * 100)
print(f"{'Pos':<6} {'Driver':<20} {'Grid':<6} {'Tyre Deg':<10} {'Miami Conf':<12} {'Race Pace':<12} {'Pred Time'}")
print("-" * 100)

for idx, row in df.iterrows():
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx}  ")
    print(f"{medal:<6} {row['Driver']:<20} P{int(row['MiamiGrid']):<5} {row['TyreDeg']:>8.1f}  {row['Miami_Confidence']:>9}/10  {row['RacePace_CleanAir']:>10.1f}s  {row['PredictedRaceTime']:>9.3f}s")

print("=" * 100)
print(f"📊 Model LOO MAE: {np.mean(errors):.4f}s | Uncertainty High due to late Rain Forecast")
print("=" * 100)