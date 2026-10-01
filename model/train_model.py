"""
Train salary prediction model and save to disk.
Features: Education, Experience, Location, Job_Title, Age, Gender
Target:   Salary
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "salary_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "salary_model.pkl")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    return df


def build_pipeline() -> Pipeline:
    categorical_features = ["Education", "Location", "Job_Title", "Gender"]
    numeric_features = ["Experience", "Age"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def train():
    df = load_and_clean(DATA_PATH)

    feature_cols = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
    target_col = "Salary"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[Model Training Complete]")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  Saved: {MODEL_PATH}")

    joblib.dump(pipeline, MODEL_PATH)

    # Save feature meta for frontend dropdowns
    meta = {
        "Education": sorted(df["Education"].unique().tolist()),
        "Location": sorted(df["Location"].unique().tolist()),
        "Job_Title": sorted(df["Job_Title"].unique().tolist()),
        "Gender": sorted(df["Gender"].unique().tolist()),
        "Experience_min": int(df["Experience"].min()),
        "Experience_max": int(df["Experience"].max()),
        "Age_min": int(df["Age"].min()),
        "Age_max": int(df["Age"].max()),
    }
    import json
    meta_path = os.path.join(os.path.dirname(__file__), "feature_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta : {meta_path}")


if __name__ == "__main__":
    train()
