"""
2026 Canadian Grand Prix — Race Prediction v2 (UPDATED WITH REAL WEEKEND DATA)
Model: XGBoost (XGBRegressor) v2

REAL DATA NOW AVAILABLE:
  - Qualifying result (Sat May 23): Full grid confirmed
  - Sprint result (Sat May 23): Russell 1st, Norris 2nd, Antonelli 3rd
  - Quali pole: Russell 1:12.578 (-0.068 Antonelli, -0.293 Norris)
  - Race start: Sunday May 24, 16:00 local / 20:00 UTC — IMMINENT
  - Rain declared: FIA rain hazard flag active (40%+ rain probability)
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

# ─── RACE METADATA ────────────────────────────────────────────────────────────
weather_data = {
    "RainProbability": 0.70,   # FIA rain hazard declared; upgraded from 65%
    "Temperature":     17.0,   # Cooler than forecast; overcast race day
}

# ─── CONSTRUCTOR STANDINGS (after Round 4, Miami) ─────────────────────────────
constructor_pts = {
    "Mercedes":     180,
    "Ferrari":      110,
    "McLaren":       94,
    "Red Bull":      30,
    "Alpine":        23,
    "Haas":          20,
    "Racing Bulls":   8,
    "Williams":       5,
    "Audi":           4,
    "Aston Martin":   0,
    "Cadillac":       0,
}

# ─── REAL QUALIFYING GAP DATA (Saturday May 23 Q3) ────────────────────────────
# Source: confirmed qualifying results
# P1  Russell    1:12.578  gap=0.000
# P2  Antonelli  1:12.646  gap=0.068
# P3  Norris     1:12.871  gap=0.293
# P4  Piastri    1:12.912  gap=0.334
# P5  Hamilton   1:12.868  gap=0.290 (mistake on 2nd run; actual time 1:12.868 ~ gap 0.290)
# P6  Verstappen 1:12.907  gap=0.329
# P7  Hadjar     1:12.935  gap=0.357
# P8  Leclerc    1:12.976  gap=0.398
# P9  Lindblad   1:13.280  gap=0.702
# P10 Colapinto  1:13.697  gap=1.119

# Sprint result (Saturday, 23 laps):
# P1 Russell, P2 Norris, P3 Antonelli, P4 Piastri, P5 Leclerc, P6 Hamilton
# P7 Verstappen, P8 Lindblad  (Hadjar: engine issues, classified P21)

canada_data = [
    # name, team, real_quali_gap, real_grid, sprint_pos, race_pace, tyre_deg,
    # track_fit, upgrade_boost, dnf_risk, penalty_risk, rain_perf, constructor_pts
    {
        "Driver":              "George Russell",
        "Team":                "Mercedes",
        "QualiGap_S":          0.000,   # Pole: 1:12.578
        "RaceGrid":            1,
        "SprintPos":           1,       # Won the sprint
        "RacePace_CleanAir":   94.0,
        "TyreDeg":             2.7,
        "TrackFit":            9.5,
        "UpgradeBoost":        0.15,
        "DNF_Risk":            0.07,
        "PenaltyRisk":         0.12,
        "RainPerformance":     9.0,
        "ConstructorPts":      180,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Kimi Antonelli",
        "Team":                "Mercedes",
        "QualiGap_S":          0.068,   # P2: 1:12.646
        "RaceGrid":            2,
        "SprintPos":           3,       # Sprint P3 (twice went off track)
        "RacePace_CleanAir":   94.2,
        "TyreDeg":             2.6,
        "TrackFit":            8.2,
        "UpgradeBoost":        0.15,
        "DNF_Risk":            0.10,
        "PenaltyRisk":         0.22,    # Sprint contact; aggressive starts
        "RainPerformance":     8.5,
        "ConstructorPts":      180,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Lando Norris",
        "Team":                "McLaren",
        "QualiGap_S":          0.293,   # P3: 1:12.871
        "RaceGrid":            3,
        "SprintPos":           2,       # Sprint P2 — excellent form
        "RacePace_CleanAir":   94.5,
        "TyreDeg":             3.0,
        "TrackFit":            8.8,
        "UpgradeBoost":        0.08,
        "DNF_Risk":            0.10,
        "PenaltyRisk":         0.18,
        "RainPerformance":     8.8,
        "ConstructorPts":      94,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Oscar Piastri",
        "Team":                "McLaren",
        "QualiGap_S":          0.334,   # P4: 1:12.912
        "RaceGrid":            4,
        "SprintPos":           4,
        "RacePace_CleanAir":   94.6,
        "TyreDeg":             3.1,
        "TrackFit":            8.0,
        "UpgradeBoost":        0.08,
        "DNF_Risk":            0.10,
        "PenaltyRisk":         0.10,
        "RainPerformance":     7.8,
        "ConstructorPts":      94,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Lewis Hamilton",
        "Team":                "Ferrari",
        "QualiGap_S":          0.290,   # P5: 1:12.868
        "RaceGrid":            5,
        "SprintPos":           6,       # Sprint P6
        "RacePace_CleanAir":   94.9,
        "TyreDeg":             2.8,
        "TrackFit":            9.2,     # 7 wins at CGV
        "UpgradeBoost":        0.03,
        "DNF_Risk":            0.09,
        "PenaltyRisk":         0.12,
        "RainPerformance":     9.5,     # Legendary wet-weather driver
        "ConstructorPts":      110,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Max Verstappen",
        "Team":                "Red Bull",
        "QualiGap_S":          0.329,   # P6: 1:12.907
        "RaceGrid":            6,
        "SprintPos":           7,       # Sprint P7 (struggling with drivability)
        "RacePace_CleanAir":   95.3,
        "TyreDeg":             3.8,
        "TrackFit":            8.5,
        "UpgradeBoost":        0.05,
        "DNF_Risk":            0.12,
        "PenaltyRisk":         0.28,
        "RainPerformance":     9.2,
        "ConstructorPts":      30,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Isack Hadjar",
        "Team":                "Red Bull",
        "QualiGap_S":          0.357,   # P7: 1:12.935
        "RaceGrid":            7,
        "SprintPos":           21,      # Engine failure in Sprint — HIGH DNF risk
        "RacePace_CleanAir":   95.5,
        "TyreDeg":             3.5,
        "TrackFit":            7.0,
        "UpgradeBoost":        0.02,
        "DNF_Risk":            0.30,    # Engine issue in Sprint raises red flag for race
        "PenaltyRisk":         0.18,
        "RainPerformance":     7.2,
        "ConstructorPts":      30,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Charles Leclerc",
        "Team":                "Ferrari",
        "QualiGap_S":          0.398,   # P8: 1:12.976 (difficult weekend vs Hamilton)
        "RaceGrid":            8,
        "SprintPos":           5,       # Sprint P5 (better than expected)
        "RacePace_CleanAir":   94.8,
        "TyreDeg":             2.7,
        "TrackFit":            7.8,
        "UpgradeBoost":        0.03,
        "DNF_Risk":            0.15,
        "PenaltyRisk":         0.40,    # Still elevated
        "RainPerformance":     7.5,
        "ConstructorPts":      110,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Arvid Lindblad",
        "Team":                "Racing Bulls",
        "QualiGap_S":          0.702,   # P9: 1:13.280 — impressive for rookie
        "RaceGrid":            9,
        "SprintPos":           8,       # Sprint P8, scored a point
        "RacePace_CleanAir":   96.0,
        "TyreDeg":             3.4,
        "TrackFit":            7.0,
        "UpgradeBoost":        0.01,
        "DNF_Risk":            0.15,
        "PenaltyRisk":         0.18,
        "RainPerformance":     7.0,
        "ConstructorPts":      8,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Franco Colapinto",
        "Team":                "Alpine",
        "QualiGap_S":          1.119,   # P10: 1:13.697 — made Q3!
        "RaceGrid":            10,
        "SprintPos":           9,       # Sprint P9
        "RacePace_CleanAir":   96.0,
        "TyreDeg":             3.4,
        "TrackFit":            7.0,
        "UpgradeBoost":        0.02,
        "DNF_Risk":            0.15,
        "PenaltyRisk":         0.22,
        "RainPerformance":     7.0,
        "ConstructorPts":      23,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Nico Hulkenberg",
        "Team":                "Audi",
        "QualiGap_S":          1.308,   # P11: 1:13.886
        "RaceGrid":            11,
        "SprintPos":           15,      # Sprint penalty applied
        "RacePace_CleanAir":   96.5,
        "TyreDeg":             3.6,
        "TrackFit":            7.2,
        "UpgradeBoost":        0.02,
        "DNF_Risk":            0.12,
        "PenaltyRisk":         0.30,    # Sprint penalty
        "RainPerformance":     7.5,
        "ConstructorPts":      4,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Liam Lawson",
        "Team":                "Racing Bulls",
        "QualiGap_S":          1.319,   # P12: 1:13.897
        "RaceGrid":            12,
        "SprintPos":           11,
        "RacePace_CleanAir":   96.2,
        "TyreDeg":             3.5,
        "TrackFit":            6.8,
        "UpgradeBoost":        0.01,
        "DNF_Risk":            0.14,
        "PenaltyRisk":         0.18,
        "RainPerformance":     7.2,
        "ConstructorPts":      8,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Gabriel Bortoleto",
        "Team":                "Audi",
        "QualiGap_S":          1.493,   # P13: 1:14.071
        "RaceGrid":            13,
        "SprintPos":           12,
        "RacePace_CleanAir":   96.8,
        "TyreDeg":             3.7,
        "TrackFit":            6.5,
        "UpgradeBoost":        0.02,
        "DNF_Risk":            0.12,
        "PenaltyRisk":         0.18,
        "RainPerformance":     7.0,
        "ConstructorPts":      4,
        "SyntheticRaceTime":   None,
    },
    {
        "Driver":              "Carlos Sainz",
        "Team":                "Williams",
        "QualiGap_S":          1.695,   # P15: 1:14.273
        "RaceGrid":            15,
        "SprintPos":           10,
        "RacePace_CleanAir":   96.3,
        "TyreDeg":             3.3,
        "TrackFit":            8.0,
        "UpgradeBoost":        0.02,
        "DNF_Risk":            0.10,
        "PenaltyRisk":         0.14,
        "RainPerformance":     8.5,
        "ConstructorPts":      5,
        "SyntheticRaceTime":   None,
    },
]

np.random.seed(42)
df = pd.DataFrame(canada_data)
df["RainProbability"] = weather_data["RainProbability"]
df["Temperature"]     = weather_data["Temperature"]

# Rain compression: with FIA hazard declared, rain equaliser is stronger
# Rain effectively narrows clean-air pace gaps by ~45% when RainProb>=0.70
rain_comp = 0.45

df["SyntheticRaceTime"] = (
    70 * df["RacePace_CleanAir"]
    - (df["TrackFit"] - 7.0) * 0.9 * (1 - rain_comp)
    - df["UpgradeBoost"] * 70
    + df["RaceGrid"] * 0.28
    + df["TyreDeg"] * 0.5
    - df["RainPerformance"] * weather_data["RainProbability"] * 0.7   # rain perf more impactful
    + (df["SprintPos"] - 1) * 0.3     # sprint pace signal
    - (df["QualiGap_S"] * 12)         # quali gap = strong pace proxy (real data now)
    + np.random.normal(0, 0.25, len(df))
).round(3)

# ─── FEATURES ─────────────────────────────────────────────────────────────────
features = [
    "QualiGap_S",        # REAL qualifying gap — primary pace signal
    "RaceGrid",          # REAL race grid position
    "SprintPos",         # REAL sprint result — recent form on this circuit
    "ConstructorPts",
    "RacePace_CleanAir",
    "TyreDeg",
    "TrackFit",
    "UpgradeBoost",
    "RainProbability",
    "Temperature",
    "RainPerformance",
]

X = df[features]
y = df["SyntheticRaceTime"]

xgb_params = dict(
    n_estimators=300, learning_rate=0.06, max_depth=3,
    subsample=0.85, colsample_bytree=0.75,
    reg_lambda=1.0, reg_alpha=0.1,
    min_child_weight=2,
    objective="reg:squarederror", eval_metric="mae",
    random_state=42, verbosity=0,
)

loo = LeaveOneOut()
errors = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    mdl = XGBRegressor(**xgb_params)
    mdl.fit(X_train, y_train)
    errors.append(mean_absolute_error(y_test, mdl.predict(X_test)))
loo_mae = np.mean(errors)

model = XGBRegressor(**xgb_params)
model.fit(X, y)
df["PredictedRaceTime"] = model.predict(X).round(3)
df["GapToLeader_S"]     = (df["PredictedRaceTime"] - df["PredictedRaceTime"].min()).round(3)
df = df.sort_values("PredictedRaceTime").reset_index(drop=True)
df.index += 1

print("\n" + "=" * 125)
print("🏁  2026 CANADIAN GRAND PRIX — UPDATED RACE PREDICTION (v2, real quali + sprint data)  🏁")
print("    Round 5 | May 24, 2026 | 70 Laps | Circuit Gilles Villeneuve | FIA RAIN HAZARD DECLARED")
print(f"    🌧️  {int(weather_data['RainProbability']*100)}% rain | {weather_data['Temperature']}°C | Tyre: C3 Hard / C4 Med / C5 Soft")
print("=" * 125)
print(f"{'Pos':<5} {'Driver':<22} {'Team':<16} {'Grid':<5} {'Quali Gap':<10} {'Sprint P':<9} {'Rain':<7} {'Gap to P1':>10}  {'Pred Time':>10}")
print("-" * 125)

for idx, row in df.iterrows():
    medal   = {1: "🥇", 2: "🥈", 3: "🥉"}.get(idx, f"P{idx:<3}")
    dnf_tag = " ⚠️ " if row["DNF_Risk"] > 0.20 else ""
    pen_tag = " 🟡" if row["PenaltyRisk"] > 0.35 else ""
    eng_tag = " 🔴ENG" if row["Driver"] == "Isack Hadjar" else ""
    flags   = dnf_tag + pen_tag + eng_tag
    sprint_display = f"P{int(row['SprintPos'])}" if row["SprintPos"] <= 20 else "DNF"
    print(
        f"{str(medal):<5} "
        f"{row['Driver']:<22} "
        f"{row['Team']:<16} "
        f"P{int(row['RaceGrid']):<4} "
        f"+{row['QualiGap_S']:.3f}s   "
        f"{sprint_display:<9} "
        f"{row['RainPerformance']:>4.1f}/10  "
        f"{row['GapToLeader_S']:>+10.3f}s  "
        f"{row['PredictedRaceTime']:>10.3f}s"
        f"{flags}"
    )

print("=" * 125)
print(f"📊 LOO MAE: {loo_mae:.3f}s")
print("=" * 125)

print("\n📈 FEATURE IMPORTANCE (gain):")
print("-" * 60)
scores = model.get_booster().get_score(importance_type="gain")
imp_df = pd.DataFrame({"Feature": list(scores.keys()), "Gain": list(scores.values())
                       }).sort_values("Gain", ascending=False).reset_index(drop=True)
imp_df["Norm"] = imp_df["Gain"] / imp_df["Gain"].sum()
for _, row in imp_df.iterrows():
    bar = "█" * int(row["Norm"] * 50)
    print(f"  {row['Feature']:<28} {row['Norm']:.4f}  {bar}")

print("\n⚠️  KEY RISK FLAGS:")
print("-" * 60)
print("  🔴 Isack Hadjar      — Engine failure in Sprint. Race DNF risk: 0.30")
print("  🟡 Charles Leclerc   — Penalty risk 0.40. Struggled vs Hamilton all weekend")
print("  🟡 Nico Hulkenberg   — 10s Sprint penalty applied; aggressive in traffic")
print("  🟡 Max Verstappen    — Drivability issues in quali; RB22 still a concern")
print("  ⚠️  Franco Colapinto  — Made Q3 but DNF risk at Wall of Champions")

print("\n📝 RACE NOTES (real weekend data incorporated):")
print("  • Russell confirmed pole 1:12.578 — 3rd consecutive CGV pole")
print("  • Sprint: Russell 1st, Norris 2nd, Antonelli 3rd (twice ran wide battling Russell)")
print("  • Hamilton 5th in quali (1:12.868) — abandoned 2nd run; stronger vs Leclerc (P8)")
print("  • Verstappen 6th in quali — 'lack of straight-line speed' confirmed RB22 issue")
print("  • Hadjar engine issue in Sprint — race reliability question mark (DNF risk elevated to 0.30)")
print("  • Leclerc 8th in quali despite being within 0.398s — Ferrari balance issues persist")
print("  • Lindblad P9 & Colapinto P10 — both exceeded expectations in Q3")
print("  • FIA rain hazard flag declared — first wet race of 2026 season possible")
print("  • Rain scenario boosts Hamilton (9.5/10), Verstappen (9.2/10), Norris (8.8/10)")
print("=" * 125)