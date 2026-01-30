import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Import your own modules from the sc=rc folder
from .data_loader import JobDataLoader
from .preprocessing import DataPreprocessor


def train_model():
    # 1. Load Data
    print("--- Loading Data ---")
    loader = JobDataLoader()
    df = loader.load_and_clean()

    # 2. Preprocess & Split
    print("--- Processing Data ---")
    preprocessor_obj = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor_obj.prepare_split(df)

    # Get the transformer (the logic for scaling/encoding)
    transformer = preprocessor_obj.get_transformer()

    # 3. Create a pipeline
    # This bundles the trandformation anf=d the model into one object
    print("--- Initializing Model ---")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # 4. Train the Model
    print("--- Training Started ---")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate the Model
    print("--- Evaluation ---")
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"R2 Score (Accuracy): {r2:,.4f}")

    # 6. Save the Model to the 'models/'folder
    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(pipeline, "models/salary_model_v1.pkl")
    print("--- Model Saved to models/salary_model_v1.pkl ---")


if __name__ == "__main__":
    train_model()
