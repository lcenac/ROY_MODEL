# NBA Rookie of the Year Prediction Model

## Project Description 
This project predicts the probability of each NBA rookie winning the Rookie of the Year (ROY) award using historical data from 2005-2025 and a logistic regression model. The model analyzes per-game statistics, draft positions, and team performance to generate probability scores for current season rookies.


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

## External Resources and Citations
- https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://www.geeksforgeeks.org/machine-learning/ml-logistic-regression-using-python/
- NBA API documentation: https://github.com/swar/nba_api


