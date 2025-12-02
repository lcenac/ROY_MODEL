# NBA Rookie of the Year Prediction Model

## Project Description 
This project predicts the probability of each NBA rookie winning the Rookie of the Year (ROY) award using historical data from 2005-2024 and a logistic regression model. The model analyzes per-game statistics, draft positions, and team performance to generate probability scores for current season rookies.


## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Model
```bash
# Generate predictions for the 2025-26 season:
python roy_p_m.py

# This will create predictions.csv with player_name and probability columns
```

### Optional: Regenerate Training Data
```bash
# To rebuild the dataset from scratch, this may take a few minutes
python generate_roy_csv.py

# This will create nba_api_roy_dataset_2010_2024.csv
```

## How It Works

### Data Collection (`generate_roy_csv.py`)
1. Fetches rookie statistics for seasons 2005-06 through 2024-25 from NBA API
2. Retrieves draft positions for each rookie
3. Merges with historical ROY winners to create binary labels (winner/non-winner)
4. Saves complete dataset to CSV for model training

### Prediction Model (`roy_p_m.py`)
1. **Data Loading**: Loads historical rookie data with ROY labels
2. **Feature Engineering**: 
   - Drops non-predictive columns (IDs, names, teams)
   - Selects only numeric features
   - Imputes missing values using median strategy
   - Standardizes features using StandardScaler
3. **Model Training**: 
   - Logistic Regression with balanced class weights (to handle few winners vs many rookies)
   - Random state set to 1 for reproducibility
4. **Current Season Prediction**:
   - Fetches 2025-26 rookie data from NBA API
   - Applies same preprocessing pipeline
   - Generates probability scores for each rookie
5. **Output**: Saves predictions sorted by probability (highest first)

## Model Approach

### Features Used
- Per-game statistics: points, rebounds, assists, field goal %, 3-point %, free throw %, etc.
- Advanced metrics: plus/minus
- Draft position
- Games played and minutes per game

### Key Modeling Decisions
- **Per-game stats** rather than totals to normalize for playing time
- **Balanced class weights** to address severe class imbalance (1 winner per ~60 rookies per season)
- **Median imputation** for missing values to handle rookies with incomplete statistics
- **Feature standardization** to ensure all statistics are on comparable scales

### Assumptions
- Historical patterns from 2005-2024 are predictive of future ROY outcomes
- Per-game statistics capture rookie performance better than volume stats
- Draft position provides useful signal about expectations and opportunity
- Missing statistics can be reasonably imputed with median values

## Data Sources
- **NBA API** (`nba_api` Python package)
  - LeagueDashPlayerStats endpoint: Rookie statistics by season
  - DraftBoard endpoint: Draft position information
  - Documentation: https://github.com/swar/nba_api
- **Historical ROY Winners**: Manually compiled from NBA official records in `roy_winners.csv`

## External Resources and Citations
- Logistic Regression concepts: https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
- Scikit-learn LogisticRegression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn tutorial: https://www.codecademy.com/article/linear-regression-with-scikit-learn-a-step-by-step-guide-using-python
- NBA API documentation: https://github.com/swar/nba_api

## Output Format

The `predictions.csv` file contains two columns:
- `player_name`: Name of the rookie player
- `probability`: Probability of winning ROY (0-1 scale, rounded to 4 decimals)

Predictions are sorted in descending order by probability.

## Troubleshooting

**If API calls fail:**
- Check internet connection
- NBA API may be temporarily unavailable - try again later
- Increase sleep time between API calls if rate limited

**If predictions.csv is empty or has errors:**
- Ensure 2025-26 season has started and rookie data is available
- Verify `nba_api_roy_dataset_2010_2024.csv` exists and is not corrupted
- Check that all required columns are present in the training data

**If features don't match:**
- Ensure the same preprocessing steps (columns dropped, imputation, scaling) are applied consistently
- Verify numeric_columns list matches between training and prediction
