import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# ─── RACE METADATA ────────────────────────────────────────────────────────────
# Round 4 | Miami International Autodrome | May 3, 2026
# 57 Laps | 5.412 km | Sprint weekend
# Weather: Dry Fri/Sat, ~40–88% rain probability Sunday (thunderstorm risk)

weather_data = {
    "RainProbability": 0.55,   # Blended forecast: AccuWeather 88%, F1 official 40%
    "Temperature": 28.0,       # Sunday race-day high °C
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
    "Haas":           0,   # Bearman crash Japan, Ocon no points yet
    "Aston Martin":   0,
}

# ─── DRIVER DATA ──────────────────────────────────────────────────────────────
# Sources:
#  - JapanGrid          : 2026 Japanese GP qualifying grid (actual)
#  - JapanGapFromPole_S : Japan Q3 gap to Antonelli's pole (1:28.778)
#  - MiamiGrid          : Estimated grid (Sprint Qualifying not yet run;
#                         based on Japan pace + ERS reg changes + analyst intel)
#  - RacePace_CleanAir  : Estimated per-lap race pace at Miami (5.412 km circuit)
#  - TyreDeg            : Miami tyre deg index (low-med deg circuit, resurfaced 2023)
#  - Miami_Confidence   : Driver Miami circuit confidence/historical form /10
#  - ActualRaceTime     : Japan actual race time used as training target (seconds/lap avg)
#
# Key context for Miami vs Japan:
#  - FIA ERS deployment & qualifying energy limit changes → could close Mercedes gap
#  - All teams bringing upgrades after 5-week break
#  - McLaren & Ferrari expected to have narrowed gap; Red Bull still struggling
#  - Rain risk could scramble strategy significantly

miami_data = [
    # ── MERCEDES ──────────────────────────────────────────────────────────────
    {
        "Driver":              "Kimi Antonelli",
        "Team":                "Mercedes",
        "UltimateLap_S":       88.778,          # Japan pole time
        "JapanGapFromPole_S":  0.000,
        "JapanGrid":           1,
        "MiamiGrid":           1,               # Championship leader, sprint pole fav
        "RacePace_CleanAir":   93.8,            # Miami slightly slower than Suzuka
        "TyreDeg":             2.8,             # Miami low-deg circuit
        "Miami_Confidence":    8.5,             # Strong recent form; limited Miami history
        "ActualRaceTime":      97.21,           # Japan race lap avg (training target)
    },
    {
        "Driver":              "George Russell",
        "Team":                "Mercedes",
        "UltimateLap_S":       89.076,          # Japan P2
        "JapanGapFromPole_S":  0.298,
        "JapanGrid":           2,
        "MiamiGrid":           2,               # Podium finisher in Miami 2025
        "RacePace_CleanAir":   93.9,
        "TyreDeg":             2.9,
        "Miami_Confidence":    9.2,             # Strong Miami history
        "ActualRaceTime":      97.43,
    },
    # ── McLAREN ───────────────────────────────────────────────────────────────
    {
        "Driver":              "Oscar Piastri",
        "Team":                "McLaren",
        "UltimateLap_S":       89.132,          # Japan P3
        "JapanGapFromPole_S":  0.354,
        "JapanGrid":           3,
        "MiamiGrid":           3,               # Strong recent form, Japan P2
        "RacePace_CleanAir":   94.2,
        "TyreDeg":             3.1,
        "Miami_Confidence":    8.8,
        "ActualRaceTime":      97.88,
    },
    {
        "Driver":              "Lando Norris",
        "Team":                "McLaren",
        "UltimateLap_S":       89.409,          # Japan P5
        "JapanGapFromPole_S":  0.631,
        "JapanGrid":           5,
        "MiamiGrid":           4,               # 2024 Miami winner; high confidence
        "RacePace_CleanAir":   94.4,
        "TyreDeg":             3.2,
        "Miami_Confidence":    9.5,             # Won maiden GP here 2024
        "ActualRaceTime":      98.05,
    },
    # ── FERRARI ───────────────────────────────────────────────────────────────
    {
        "Driver":              "Charles Leclerc",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.405,          # Japan P4
        "JapanGapFromPole_S":  0.627,
        "JapanGrid":           4,
        "MiamiGrid":           5,
        "RacePace_CleanAir":   94.3,
        "TyreDeg":             2.7,             # Ferrari strong tyre management
        "Miami_Confidence":    8.7,
        "ActualRaceTime":      97.75,
    },
    {
        "Driver":              "Lewis Hamilton",
        "Team":                "Ferrari",
        "UltimateLap_S":       89.567,          # Japan P6
        "JapanGapFromPole_S":  0.789,
        "JapanGrid":           6,
        "MiamiGrid":           6,
        "RacePace_CleanAir":   94.5,
        "TyreDeg":             2.8,
        "Miami_Confidence":    8.0,             # Seeking return to form in US
        "ActualRaceTime":      97.90,
    },
    # ── ALPINE ────────────────────────────────────────────────────────────────
    {
        "Driver":              "Pierre Gasly",
        "Team":                "Alpine",
        "UltimateLap_S":       89.691,          # Japan P7
        "JapanGapFromPole_S":  0.913,
        "JapanGrid":           7,
        "MiamiGrid":           7,               # Strong recent form for Alpine
        "RacePace_CleanAir":   95.8,
        "TyreDeg":             3.5,
        "Miami_Confidence":    7.5,
        "ActualRaceTime":      99.10,
    },
    # ── RED BULL ──────────────────────────────────────────────────────────────
    {
        "Driver":              "Max Verstappen",
        "Team":                "Red Bull",
        "UltimateLap_S":       89.992,          # Japan P11 (Q2 exit)
        "JapanGapFromPole_S":  1.214,
        "JapanGrid":           11,
        "MiamiGrid":           9,               # Better Miami circuit suit expected
        "RacePace_CleanAir":   95.5,
        "TyreDeg":             4.2,
        "Miami_Confidence":    7.8,             # Struggling with 2026 RB car
        "ActualRaceTime":      98.95,
    },
    # ── RACING BULLS ──────────────────────────────────────────────────────────
    {
        "Driver":              "Isack Hadjar",
        "Team":                "Racing Bulls",
        "UltimateLap_S":       89.978,          # Japan P8
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
        "UltimateLap_S":       90.274,          # Japan P9
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

# ─── LEAVE-ONE-OUT CROSS VALIDATION ───────────────────────────────────────────
loo    = LeaveOneOut()
errors = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=4, random_state=39
    )
    model.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, model.predict(X_test)))

# ─── FINAL MODEL FIT & PREDICTION ─────────────────────────────────────────────
model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("🏁  2026 MIAMI GRAND PRIX RACE PREDICTION  🏁")
print("    Round 4 | May 3, 2026 | 57 Laps | 5.412 km | Miami International Autodrome")
print(f"    ⛈️  Race Day Weather: ~{int(weather_data['RainProbability']*100)}% rain probability | {weather_data['Temperature']}°C")
print("    ⚠️  FIA ERS rule changes in effect | Teams with 5-week upgrade packages")
print("=" * 100)
print(f"{'Pos':<6} {'Driver':<20} {'Grid':<6} {'Tyre Deg':<10} {'Miami Conf':<12} {'Race Pace':<12} {'Pred Time'}")
print("-" * 100)

for idx, row in df.iterrows():
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx}  ")
    print(
        f"{medal:<6} "
        f"{row['Driver']:<20} "
        f"P{int(row['MiamiGrid']):<5} "
        f"{row['TyreDeg']:>8.1f}  "
        f"{row['Miami_Confidence']:>9}/10  "
        f"{row['RacePace_CleanAir']:>10.1f}s  "
        f"{row['PredictedRaceTime']:>9.3f}s"
    )

print("=" * 100)
print(f"📊 Leave-One-Out MAE: {np.mean(errors):.4f} seconds")
print("=" * 100)

importance_df = pd.DataFrame({
    "Feature":    features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print("\n📈 FEATURE IMPORTANCE:")
print("-" * 55)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:<28} {row['Importance']:.4f}  {bar}")

print("=" * 100)
print("\n📝 MODEL NOTES:")
print("  • Grid positions are estimated (Sprint Qualifying runs today, May 1)")
print("  • Miami_Confidence replaces Suzuka_Confidence — reflects circuit-specific form")
print("  • RainProbability elevated to 0.55 (blended AccuWeather 88% / F1 official 40%)")
print("  • FIA ERS changes may compress the field vs Japan — extra uncertainty baked in")
print("  • Rain/SC scenario could benefit Norris (2024 winner), Leclerc (aggressive strategy)")
print("=" * 100)