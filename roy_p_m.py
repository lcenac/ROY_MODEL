import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from nba_api.stats.endpoints import LeagueDashPlayerStats, DraftBoard

# Sources
#https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/

CSV_FILE = "nba_api_roy_dataset_2010_2024.csv"
OUTPUT_CSV = "predictions.csv"
season_2025_26 = "2025-26"


df = pd.read_csv(CSV_FILE)


drop_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION','Season', 'Season_End_Year', 'winner', 'WNBA_FANTASY_PTS_RANK','WNBA_FANTASY_PTS', 'Team_Wins', 'NBA_FANTASY_PTS']
rookie_stats = df.drop(columns=drop_cols, errors='ignore')
rookie_stats = rookie_stats.select_dtypes(include='number').dropna(axis=1, how='all')
numeric_columns = rookie_stats.columns.tolist()
won_roy = df['winner']

#fill in values that may be empty
imputer = SimpleImputer(strategy='median')
rookie_stats_imputed = imputer.fit_transform(rookie_stats)

scaler = StandardScaler()
rookie_stats_scaled = scaler.fit_transform(rookie_stats_imputed)


clf = LogisticRegression(random_state=1, class_weight='balanced', max_iter=1000)
clf.fit(rookie_stats_scaled, won_roy)


def get_rookies_for_season(season):
    df = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
        player_experience_nullable='Rookie'
    ).get_data_frames()[0]
    df['Team_Wins'] = df['W']  
    return df.reset_index(drop=True)

def get_draft_positions(season):
    try:
        draft_df = DraftBoard(season=season).get_data_frames()[0]
        draft_df = draft_df[draft_df['PLAYER_NAME'].notnull()]
        return dict(zip(draft_df['PLAYER_NAME'], draft_df['PK']))
    except:
        return {}

rookies_25_26 = get_rookies_for_season(season_2025_26)
rookies_25_26['Draft_Pick'] = rookies_25_26['PLAYER_NAME'].map(get_draft_positions(season_2025_26))

rookie_stats_25_26 = rookies_25_26[numeric_columns].copy()
rookie_stats_25_26_imputed = imputer.transform(rookie_stats_25_26)
rookie_stats_25_26_scaled = scaler.transform(rookie_stats_25_26_imputed)


roy_probabilities = clf.predict_proba(rookie_stats_25_26_scaled)[:, 1]
rookies_25_26['probability'] = roy_probabilities




predictions = rookies_25_26[['PLAYER_NAME', 'probability']].rename(
    columns={'PLAYER_NAME': 'player_name'}
)
predictions['probability'] = predictions['probability'].clip(0, 1).round(4)
predictions = predictions.sort_values(by='probability', ascending=False)

predictions.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
