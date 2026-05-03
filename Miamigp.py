"""
2026 Miami Grand Prix — Race Prediction
Model: XGBoost (XGBRegressor)

Install dependencies:
    pip install xgboost pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# ─── RACE METADATA ────────────────────────────────────────────────────────────
# Round 4 | Miami International Autodrome | May 3, 2026
# 57 Laps | 5.412 km | Sprint weekend
# Weather: Dry Fri/Sat, ~40–88% rain probability Sunday (thunderstorm risk)

weather_data = {
    "RainProbability": 0.55,   # Blended: AccuWeather 88% / F1 official 40%
    "Temperature":     28.0,   # Sunday race-day high °C
}

# ─── CONSTRUCTOR STANDINGS (after Round 3, Japan) ─────────────────────────────
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

# ─── DRIVER DATA ──────────────────────────────────────────────────────────────
miami_data = [
    # ── MERCEDES ──────────────────────────────────────────────────────────────
    {
        "Driver":              "Kimi Antonelli",
        "Team":                "Mercedes",
        "UltimateLap_S":       88.778,
        "JapanGapFromPole_S":  0.000,
        "JapanGrid":           1,
        "MiamiGrid":           1,
        "RacePace_CleanAir":   93.8,
        "TyreDeg":             2.8,
        "Miami_Confidence":    8.5,
        "ActualRaceTime":      97.21,
    },
    {
        "Driver":              "George Russell",
        "Team":                "Mercedes",
        "UltimateLap_S":       89.076,
        "JapanGapFromPole_S":  0.298,
        "JapanGrid":           2,
        "MiamiGrid":           2,
        "RacePace_CleanAir":   93.9,
        "TyreDeg":             2.9,
        "Miami_Confidence":    9.2,
        "ActualRaceTime":      97.43,
    },
    # ── McLAREN ───────────────────────────────────────────────────────────────
    {
        "Driver":              "Oscar Piastri",
        "Team":                "McLaren",
        "UltimateLap_S":       89.132,
        "JapanGapFromPole_S":  0.354,
        "JapanGrid":           3,
        "MiamiGrid":           3,
        "RacePace_CleanAir":   94.2,
        "TyreDeg":             3.1,
        "Miami_Confidence":    8.8,
        "ActualRaceTime":      97.88,
    },
    {
        "Driver":              "Lando Norris",
        "Team":                "McLaren",
        "UltimateLap_S":       89.409,
        "JapanGapFromPole_S":  0.631,
        "JapanGrid":           5,
        "MiamiGrid":           4,
        "RacePace_CleanAir":   94.4,
        "TyreDeg":             3.2,
        "Miami_Confidence":    9.5,
        "ActualRaceTime":      98.05,
    },
    # ── FERRARI ───────────────────────────────────────────────────────────────
    {
        "Driver":              "Charles Leclerc",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.405,
        "JapanGapFromPole_S":  0.627,
        "JapanGrid":           4,
        "MiamiGrid":           5,
        "RacePace_CleanAir":   94.3,
        "TyreDeg":             2.7,
        "Miami_Confidence":    8.7,
        "ActualRaceTime":      97.75,
    },
    {
        "Driver":              "Lewis Hamilton",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.567,
        "JapanGapFromPole_S":  0.789,
        "JapanGrid":           6,
        "MiamiGrid":           6,
        "RacePace_CleanAir":   94.5,
        "TyreDeg":             2.8,
        "Miami_Confidence":    8.0,
        "ActualRaceTime":      97.90,
    },
    # ── ALPINE ────────────────────────────────────────────────────────────────
    {
        "Driver":              "Pierre Gasly",
        "Team":                "Alpine",
        "UltimateLap_S":       89.691,
        "JapanGapFromPole_S":  0.913,
        "JapanGrid":           7,
        "MiamiGrid":           7,
        "RacePace_CleanAir":   95.8,
        "TyreDeg":             3.5,
        "Miami_Confidence":    7.5,
        "ActualRaceTime":      99.10,
    },
    # ── RED BULL ──────────────────────────────────────────────────────────────
    {
        "Driver":              "Max Verstappen",
        "Team":                "Red Bull",
        "UltimateLap_S":       89.992,
        "JapanGapFromPole_S":  1.214,
        "JapanGrid":           11,
        "MiamiGrid":           9,
        "RacePace_CleanAir":   95.5,
        "TyreDeg":             4.2,
        "Miami_Confidence":    7.8,
        "ActualRaceTime":      98.95,
    },
    # ── RACING BULLS ──────────────────────────────────────────────────────────
    {
        "Driver":              "Isack Hadjar",
        "Team":                "Racing Bulls",
        "UltimateLap_S":       89.978,
        "JapanGapFromPole_S":  1.200,
        "JapanGrid":           8,
        "MiamiGrid":           8,
        "RacePace_CleanAir":   96.1,
        "TyreDeg":             3.6,
        "Miami_Confidence":    7.0,
        "ActualRaceTime":      99.50,
    },
    # ── AUDI ──────────────────────────────────────────────────────────────────
    {
        "Driver":              "Gabriel Bortoleto",
        "Team":                "Audi",
        "UltimateLap_S":       90.274,
        "JapanGapFromPole_S":  1.496,
        "JapanGrid":           9,
        "MiamiGrid":           10,
        "RacePace_CleanAir":   96.5,
        "TyreDeg":             3.8,
        "Miami_Confidence":    6.8,
        "ActualRaceTime":      99.80,
    },
]

# ─── BUILD DATAFRAME ──────────────────────────────────────────────────────────
df = pd.DataFrame(miami_data)
df["TeamScore"]       = df["Team"].map(constructor_pts).fillna(0)
df["RainProbability"] = weather_data["RainProbability"]
df["Temperature"]     = weather_data["Temperature"]

features = [
    "UltimateLap_S",
    "JapanGapFromPole_S",
    "MiamiGrid",
    "JapanGrid",
    "TeamScore",
    "RacePace_CleanAir",
    "TyreDeg",
    "Miami_Confidence",
    "RainProbability",
    "Temperature",
]

X = df[features]
y = df["ActualRaceTime"]

# ─── XGBOOST HYPERPARAMETERS ──────────────────────────────────────────────────
xgb_params = dict(
    n_estimators     = 200,
    learning_rate    = 0.08,
    max_depth        = 4,
    subsample        = 0.8,     # row sampling per tree
    colsample_bytree = 0.8,     # feature sampling per tree
    reg_lambda       = 1.0,     # L2 regularisation (λ)
    reg_alpha        = 0.0,     # L1 regularisation (α)
    min_child_weight = 1,
    objective        = "reg:squarederror",
    eval_metric      = "mae",
    random_state     = 39,
    verbosity        = 0,
)

# ─── LEAVE-ONE-OUT CROSS VALIDATION ───────────────────────────────────────────
loo    = LeaveOneOut()
errors = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    mdl = XGBRegressor(**xgb_params)
    mdl.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, mdl.predict(X_test)))

loo_mae = np.mean(errors)

# ─── FINAL MODEL FIT & PREDICT ────────────────────────────────────────────────
model = XGBRegressor(**xgb_params)
model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 105)
print("🏁  2026 MIAMI GRAND PRIX RACE PREDICTION  🏁")
print("    Round 4 | May 3, 2026 | 57 Laps | 5.412 km | Miami International Autodrome")
print(f"    ⛈️  Race Day: ~{int(weather_data['RainProbability']*100)}% rain probability | {weather_data['Temperature']}°C")
print("    ⚡  Model: XGBoost (XGBRegressor) | n_estimators=200 | lr=0.08 | λ=1.0 | subsample=0.8")
print("=" * 105)
print(f"{'Pos':<6} {'Driver':<22} {'Team':<16} {'Grid':<6} {'Tyre Deg':<10} {'Conf':<8} {'Pace':<10} {'Pred Time'}")
print("-" * 105)

for idx, row in df.iterrows():
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx}  ")
    print(
        f"{medal:<6} "
        f"{row['Driver']:<22} "
        f"{row['Team']:<16} "
        f"P{int(row['MiamiGrid']):<5} "
        f"{row['TyreDeg']:>7.1f}   "
        f"{row['Miami_Confidence']:>5}/10  "
        f"{row['RacePace_CleanAir']:>8.1f}s  "
        f"{row['PredictedRaceTime']:>9.3f}s"
    )

print("=" * 105)
print(f"📊 Leave-One-Out MAE: {loo_mae:.4f} seconds")
print("=" * 105)

# ─── FEATURE IMPORTANCE (XGBoost gain) ───────────────────────────────────────
print("\n📈 FEATURE IMPORTANCE (XGBoost — gain):")
print("-" * 60)
scores = model.get_booster().get_score(importance_type="gain")
importance_df = pd.DataFrame({
    "Feature": list(scores.keys()),
    "Gain":    list(scores.values()),
}).sort_values("Gain", ascending=False).reset_index(drop=True)
importance_df["Norm"] = importance_df["Gain"] / importance_df["Gain"].sum()

for _, row in importance_df.iterrows():
    bar = "█" * int(row["Norm"] * 50)
    print(f"  {row['Feature']:<28} {row['Norm']:.4f}  {bar}")

print("=" * 105)
print("\n📝 MODEL NOTES:")
print("  • XGBoost uses gain-based feature importance (avg. loss reduction per split)")
print("  • subsample=0.8 & colsample_bytree=0.8 add stochastic regularisation")
print("  • reg_lambda=1.0 (L2) mirrors the HistGBR version; tune reg_alpha for L1")
print("  • Grid positions estimated — Sprint Qualifying ran today (May 3)")
print("  • RainProbability = 0.55 (blended AccuWeather 88% / F1 official 40%)")
print("  • Rain/SC scenario: favours Norris (2024 winner), Leclerc (aggressive strategy)")
print("  • FIA ERS changes may compress the midfield gap vs Japan")
print("=" * 105)

# ─── OPTIONAL: matplotlib feature importance chart ────────────────────────────
# Uncomment to save a chart to disk:
# from xgboost import plot_importance
# import matplotlib.pyplot as plt
# plot_importance(model, importance_type="gain", max_num_features=10)
# plt.tight_layout()
# plt.savefig("feature_importance.png", dpi=150)
# plt.show()