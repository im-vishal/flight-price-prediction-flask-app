from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from src import logger

import mlflow
import pandas as pd
import numpy as np
import joblib

def train_model(interim_dir: Path, processed_dir: Path, model_dir: Path) -> Pipeline:
    logger.info("Loading Datasets....")
    X_train = pd.read_csv(interim_dir / "X_train.csv")
    y_train = pd.read_csv(interim_dir / "y_train.csv").squeeze()    # Convert to Series

    logger.info("Loading  preprocessor....")
    preprocessor = joblib.load(processed_dir / "preprocessor.joblib")

    logger.info("Training Model....")
    model = Pipeline(steps=[
        ("pre", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=50))
    ])

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, model_dir / "model.joblib")

    return model

def main() -> None:
    """main function to call other functions"""
    home_dir = Path(__file__).parent.parent.parent
    interim_dir = Path(home_dir) / "data/interim"
    processed_dir = Path(home_dir) / "data/processed"
    model_dir = Path(home_dir) / "models"

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    train_model(interim_dir, processed_dir, model_dir)



if __name__ == "__main__":
    main()
