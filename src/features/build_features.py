from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from feature_engine.datetime import DatetimeFeatures
from pathlib import Path
from src import logger

import sklearn
import pandas as pd
import numpy as np
import joblib

sklearn.set_config(transform_output="pandas")

def build_features(X_path: Path, y_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    logger.info("Loading Datasets....")
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    dt_cols = ["date_of_journey", "dep_time", "arrival_time"]
    num_cols = ["duration", "total_stops"]
    cat_cols = [col for col in X.columns if (col not in dt_cols) and (col not in num_cols)]

    logger.info("Building Transformers....")
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    doj_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("extractor", DatetimeFeatures(features_to_extract=["month", "week", "day_of_week", "day_of_month"], format="mixed")),
        ("scaler", StandardScaler())
    ])

    time_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("extractor", DatetimeFeatures(features_to_extract=["hour", "minute"], format="mixed")),
        ("scaler", StandardScaler())
    ])


    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
        ("doj", doj_transformer, ["date_of_journey"]),
        ("time", time_transformer, ["dep_time", "arrival_time"])
    ])

    logger.info("Transforming Datasets....")
    X_processed = preprocessor.fit_transform(X)

    

    return X_processed, y,  preprocessor


def main() -> None:
    """main function to call other functions"""
    home_dir = Path(__file__).parent.parent.parent
    interim_dir = home_dir / "data/interim"
    X_path = Path(interim_dir) / "X_train.csv"
    y_path = Path(interim_dir) / "y_train.csv"
    processed_dir = home_dir / "data/processed"

    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Saving preprocessor and transformed data....")

    X_processed, y_processed, preprocessor = build_features(X_path, y_path)
    X_processed.to_csv(Path(processed_dir) / "X_processed.csv")
    y_processed.to_csv(Path(processed_dir) / "y_processed.csv")

    joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")




if __name__ == "__main__":
    main()