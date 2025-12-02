# roy_predictor_to_csv.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats, DraftBoard

#SOURCES:
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#
# -----------------------------
# Config
# -----------------------------
CSV_FILE = "nba_api_roy_dataset_2010_2024.csv"
OUTPUT_CSV = "predictions.csv"
season_2025_26 = "2025-26"

# -----------------------------
# Load historical dataset
# -----------------------------
df = pd.read_csv(CSV_FILE)
drop_cols = ['PLAYER_ID','PLAYER_NAME','TEAM_ID','TEAM_ABBREVIATION','Season','Season_End_Year','winner']
X = df.drop(columns=drop_cols, errors='ignore')
X = X.select_dtypes(include='number').dropna(axis=1, how='all')
numeric_columns = X.columns.tolist()
y = df['winner']

# -----------------------------
# Impute & scale
# -----------------------------
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# -----------------------------
# Train Logistic Regression
# -----------------------------
clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
clf.fit(X_scaled, y)

# -----------------------------
# Fetch 2025–26 rookies
# -----------------------------
def get_rookies_for_season(season):
    df = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
        player_experience_nullable='Rookie'
    ).get_data_frames()[0]
    return df.reset_index(drop=True)

def get_team_wins(season):
    try:
        team_stats = LeagueDashTeamStats(season=season, season_type_all_star='Regular Season',
                                         per_mode_detailed='PerGame').get_data_frames()[0]
        return dict(zip(team_stats['TEAM_ABBREVIATION'], team_stats['W']))
    except:
        return {}

def get_draft_positions(season):
    try:
        draft_df = DraftBoard(season=season).get_data_frames()[0]
        draft_df = draft_df[draft_df['PLAYER_NAME'].notnull()]
        return dict(zip(draft_df['PLAYER_NAME'], draft_df['PK']))
    except:
        return {}

rookies_25_26 = get_rookies_for_season(season_2025_26)
rookies_25_26['Team_Wins'] = rookies_25_26['TEAM_ABBREVIATION'].map(get_team_wins(season_2025_26))
rookies_25_26['Draft_Pick'] = rookies_25_26['PLAYER_NAME'].map(get_draft_positions(season_2025_26))

# -----------------------------
# Prepare features
# -----------------------------
X_25_26 = rookies_25_26[numeric_columns].copy()
X_25_26_imputed = imputer.transform(X_25_26)
X_25_26_scaled = scaler.transform(X_25_26_imputed)

# -----------------------------
# Predict probabilities
# -----------------------------
probs = clf.predict_proba(X_25_26_scaled)[:,1]
rookies_25_26['probability'] = probs

# -----------------------------
# Output predictions CSV
# -----------------------------
predictions = rookies_25_26[['PLAYER_NAME', 'probability']].rename(columns={'PLAYER_NAME': 'player_name'})
predictions['probability'] = predictions['probability'].clip(0, 1)

# Optional: round probabilities to 4 decimal places
predictions['probability'] = predictions['probability'].round(4)

# Sort from highest to lowest
predictions = predictions.sort_values(by='probability', ascending=False)

# Save CSV
predictions.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV} with probabilities rounded and clipped!")
print(f"Predictions saved to {OUTPUT_CSV}, sorted by highest ROY probability!")

