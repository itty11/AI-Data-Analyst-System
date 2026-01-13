import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ------------------ Task Detection ------------------
def detect_problem_type(df, target):
    if df[target].dtype == "object" or df[target].nunique() <= 10:
        return "classification"
    return "regression"


# ------------------ Data Prep ------------------
def prepare_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns


# ------------------ AutoML Engine ------------------
def train_model(df, target):
    problem_type = detect_problem_type(df, target)
    (X_train, X_test, y_train, y_test), feature_names = prepare_data(df, target)

    models = []

    if problem_type == "regression":
        models.append(
            ("RandomForest", RandomForestRegressor(random_state=42))
        )
        if XGB_AVAILABLE:
            models.append(
                ("XGBoost", XGBRegressor(random_state=42, verbosity=0))
            )

        best_score = float("inf")
        best_model = None

        for name, model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds) ** 0.5

            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_name = name

        cv_rmse = (
            cross_val_score(
                best_model, X_train, y_train,
                cv=5,
                scoring="neg_root_mean_squared_error"
            ).mean() * -1
        )

        importances = (
            best_model.feature_importances_
            if hasattr(best_model, "feature_importances_")
            else None
        )

        return {
            "type": "regression",
            "model": best_name,
            "rmse": best_score,
            "r2": r2_score(y_test, best_model.predict(X_test)),
            "cv_rmse": cv_rmse,
            "feature_importance": importances,
            "features": feature_names,
        }

    # ---------------- Classification ----------------
    else:
        models.append(
            ("RandomForest", RandomForestClassifier(random_state=42))
        )
        if XGB_AVAILABLE:
            models.append(
                ("XGBoost", XGBClassifier(random_state=42, eval_metric="logloss"))
            )

        best_score = 0
        best_model = None

        for name, model in models:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name

        cv_acc = cross_val_score(
            best_model, X_train, y_train, cv=5, scoring="accuracy"
        ).mean()

        preds = best_model.predict(X_test)

        importances = (
            best_model.feature_importances_
            if hasattr(best_model, "feature_importances_")
            else None
        )

        return {
            "type": "classification",
            "model": best_name,
            "accuracy": best_score,
            "precision": precision_score(y_test, preds, average="weighted"),
            "recall": recall_score(y_test, preds, average="weighted"),
            "f1": f1_score(y_test, preds, average="weighted"),
            "cv_accuracy": cv_acc,
            "feature_importance": importances,
            "features": feature_names,
        }
