import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    def __init__(self):
        # Define which column get which treatment
        self.ordinal_cols = ["experience_level", "company_size"]
        self.nominal_cols = ["remote_type", "industry", "company_type"]
        self.numeric_cols = ["min_experience_years", "skill_count"]

        # Define the order for ordinal categories
        self.exp_order = ["Entry", "Mid", "Senior"]
        self.size_order = ["Small", "Medium", "Large"]

    def get_transformer(self):
        # Create the individual tools
        numeric_transformer = StandardScaler()

        nominal_transformer = OneHotEncoder(handle_unknown="ignore")

        ordinal_transformer = OrdinalEncoder(
            categories=[self.exp_order, self.size_order]
        )

        # Combine them into one "Master Transformer"
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("nom", nominal_transformer, self.nominal_cols),
                ("ord", ordinal_transformer, self.ordinal_cols),
            ]
        )
        return preprocessor

    def prepare_split(self, df):
        # 1. Define X (Features) and y (Target)
        X = df.drop(
            columns=[
                "salary_median_usd",
                "salary_min_usd",
                "salary_max_usd",
                "job_id",
                "job_title",
            ]
        )
        y = df["salary_median_usd"]

        # 2. Split into Training (80%) and Testing (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test
