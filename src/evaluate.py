import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from .data_loader import JobDataLoader
from .preprocessing import DataPreprocessor


def evaluate_features():
    # 1. Load Model & Data
    model_path = os.path.join(
        os.path.dirname(__file__), "../models/salary_model_v1.pkl"
    )
    model = joblib.load(model_path)
    loader = JobDataLoader()
    df = loader.load_and_clean()

    prep = DataPreprocessor()
    X_train, X_test, y_train, y_test = prep.prepare_split(df)

    # 2. Extract Feature Importance
    # We reach into the pipeline to get the regressor
    importances = model.named_steps["regressor"].feature_importances_

    # Get the names of the columns after they were encoded
    # Note: This is a bit advanced because One-Hot Encoding creates new columns
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    # 3. Plotting
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances.nlargest(10).plot(kind="barh")
    plt.title("Top 10 Factors Driving Salary")
    plt.tight_layout()
    plt.savefig("../salary_features.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    evaluate_features()
