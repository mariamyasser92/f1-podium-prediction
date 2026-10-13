# Formula 1 Podium Prediction Companion

CSEN 903 – Advanced Computer Lab  
Milestone 1

## Project Goal

This project predicts whether a Formula 1 driver will finish on the podium using only information available after qualifying and before the race starts.

The prediction target is:

`positionOrder <= 3`

The modeling table contains one row per driver-race entry.

## Dataset

Formula 1 World Championship dataset from Kaggle:

rohanrao / Formula 1 World Championship 1950–2020

The supplied dataset contains historical Formula 1 data through 2024 and consists of 14 relational CSV files.

The raw dataset is not included in this repository.

## Dataset Setup

On Kaggle:

1. Open the notebook.
2. Attach the Formula 1 World Championship dataset by rohanrao.
3. Make sure all 14 CSV files are available.
4. Enable Internet if SHAP or LIME need to be installed.
5. Restart the session and use Run All.

For local execution:

1. Install dependencies:

```bash
pip install -r requirements.txt
