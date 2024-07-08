from sklearn.metrics import r2_score
from pathlib import Path
from src import logger

import pandas as pd
import numpy as np
import joblib
import json
import mlflow

def evaluate_model(data_dir: Path, model_path: Path):
    logger.info("Loading Datasets....")
    X = pd.read_csv(data_dir / "X_test.csv")
    y = pd.read_csv(data_dir / "y_test.csv").squeeze()  # Convert to Series

    logger.info("Loading Model....")
    model = joblib.load(model_path)

    logger.info("Evaluating model....")
    y_pred = model.predict(X)
    score = r2_score(y, y_pred)

    return score


def main() -> None:
    """main function to call other functions"""
    home_dir = Path(__file__).parent.parent.parent
    data_dir = home_dir / "data/interim"
    model_path = home_dir / "models/model.joblib"

    r2_score = evaluate_model(data_dir, model_path)
    d = {"r2_score": r2_score}
    
    Path(home_dir / 'reports/figures/').mkdir(parents=True, exist_ok=True)
    with open(home_dir / 'reports/figures/results.json', 'w') as f:
        json.dump(d, f)

    mlflow.log_metric("r2_score", r2_score)


if __name__ == "__main__":
    main()