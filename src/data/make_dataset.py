from pathlib import Path
from src import logger

import pandas as pd

def split_data(data_dir: Path) ->  tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    logger.info("Loading Data....")
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    val_df = pd.read_csv(Path(data_dir) / "val.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    data = pd.concat([train_df, val_df], axis=0)

    X = data.drop(columns="price")
    y = data["price"].copy()
    X_test = test_df.drop(columns="price")
    y_test = test_df["price"].copy()

    return X, y, X_test, y_test


def main() -> None:
    """main function to call other functions"""
    home_dir = Path(__file__).parent.parent.parent
    raw_dir = home_dir / "data/raw"
    interim_dir = home_dir / "data/interim"

    # print(f"{raw_dir = }")
    logger.info("Saving interim dataset")
    X, y, X_test, y_test = split_data(raw_dir)

    Path(interim_dir).mkdir(parents=True, exist_ok=True)
    X.to_csv(interim_dir / "X_train.csv", index=False)
    y.to_csv(interim_dir / "y_train.csv", index=False)
    X_test.to_csv(interim_dir / "X_test.csv", index=False)
    y_test.to_csv(interim_dir / "y_test.csv", index=False)

if __name__ == "__main__":
    main()

    