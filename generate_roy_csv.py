# generate_roy_dataset_nba_api_safe.py
import pandas as pd
import time
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats, DraftBoard

# -----------------------------
# Config
# -----------------------------
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2005, 2025)]
WINNERS_CSV = "roy_winners.csv"  # CSV with columns SEASON,player_name
OUTPUT_CSV = "nba_api_roy_dataset_2010_2024.csv"

# -----------------------------
# Fetch rookie stats
# -----------------------------
def get_rookies_for_season(season):
    print(f"Fetching rookie stats for {season}...")
    df = LeagueDashPlayerStats(
        season=season,
        season_type_all_star='Regular Season',
        per_mode_detailed='PerGame',
        player_experience_nullable='Rookie'
    ).get_data_frames()[0]
    df['Season'] = season
    return df.reset_index(drop=True)

# -----------------------------
# Fetch team wins
# -----------------------------
def get_team_wins(season):
    print(f"Fetching team wins for {season}...")
    try:
        team_stats = LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame'
        ).get_data_frames()[0]
        return dict(zip(team_stats['TEAM_ABBREVIATION'], team_stats['W']))
    except:
        return {}

# -----------------------------
# Fetch draft pick
# -----------------------------
def get_draft_positions(season):
    print(f"Fetching draft picks for {season}...")
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
    """
    Convert season string like '2010–11' or '2010-11' → 2011
    Returns None if season is invalid.
    """
    if not isinstance(season, str):
        return None
    season = season.replace('–', '-')  # normalize dash
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

    # Add team wins
    team_wins = get_team_wins(season)
    rookies['Team_Wins'] = rookies['TEAM_ABBREVIATION'].map(team_wins)

    # Add draft pick
    draft_pos = get_draft_positions(season)
    rookies['Draft_Pick'] = rookies['PLAYER_NAME'].map(draft_pos)

    all_data.append(rookies)
    time.sleep(1)  # avoid rate limits

df = pd.concat(all_data, ignore_index=True)
print(f"Collected stats for {len(df)} rookies.")

# -----------------------------
# Add ROY labels
# -----------------------------
winners = pd.read_csv(WINNERS_CSV)
winners['Season_End_Year'] = winners['SEASON'].apply(season_string_to_end_year)

df['Season_End_Year'] = df['Season'].apply(season_string_to_end_year)
df = df[df['Season_End_Year'].notnull()]  # drop malformed season rows

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
print(f"Dataset saved to {OUTPUT_CSV}. Ready for ML!")
