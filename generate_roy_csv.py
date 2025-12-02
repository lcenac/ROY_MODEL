import pandas as pd
import time
from nba_api.stats.endpoints import LeagueDashPlayerStats, DraftBoard

# -----------------------------
# Config
# -----------------------------
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2005, 2025)]
WINNERS_CSV = "roy_winners.csv"  
OUTPUT_CSV = "nba_api_roy_dataset_2010_2024.csv"

# -----------------------------
# Fetch rookie stats
# -----------------------------
def get_rookies_for_season(season):
    df = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
        player_experience_nullable='Rookie'
    ).get_data_frames()[0]
    df['Season'] = season
    # Use the 'W' column directly from player stats as Team Wins
    df['Team_Wins'] = df['W']
    return df.reset_index(drop=True)

# -----------------------------
# Fetch draft pick
# -----------------------------
def get_draft_positions(season):
    try:
        draft_df = DraftBoard(season=season).get_data_frames()[0]
        draft_df = draft_df[draft_df['PLAYER_NAME'].notnull()]
        return dict(zip(draft_df['PLAYER_NAME'], draft_df['PK']))
    except:
        return {}

# -----------------------------
# Robust Season → End Year
# -----------------------------
def season_string_to_end_year(season):
    if not isinstance(season, str):
        return None
    season = season.replace('–', '-') 
    parts = season.split('-')
    if len(parts) != 2:
        return None
    start, end = parts
    try:
        end_year = int(end)
        if end_year < 100:
            end_year += 2000
        return end_year
    except ValueError:
        return None

# -----------------------------
# Build dataset
# -----------------------------
all_data = []

for season in SEASONS:
    rookies = get_rookies_for_season(season)
    if rookies.empty:
        continue

    # Add draft pick
    draft_pos = get_draft_positions(season)
    rookies['Draft_Pick'] = rookies['PLAYER_NAME'].map(draft_pos)

    all_data.append(rookies)
    time.sleep(1)  # avoid rate limits

df = pd.concat(all_data, ignore_index=True)

# -----------------------------
# Add ROY labels
# -----------------------------
winners = pd.read_csv(WINNERS_CSV)
winners['Season_End_Year'] = winners['SEASON'].apply(season_string_to_end_year)

df['Season_End_Year'] = df['Season'].apply(season_string_to_end_year)
df = df[df['Season_End_Year'].notnull()] 

df = df.merge(
    winners[['Season_End_Year', 'player_name']],
    left_on=['Season_End_Year', 'PLAYER_NAME'],
    right_on=['Season_End_Year', 'player_name'],
    how='left'
)
df['winner'] = df['player_name'].notnull().astype(int)
df = df.drop(columns=['player_name'])

# -----------------------------
# Save dataset
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"Dataset saved to {OUTPUT_CSV}")
