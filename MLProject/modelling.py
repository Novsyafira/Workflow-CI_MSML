import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def main():
    mlflow.set_experiment("CI_Obesity_Classification")
    mlflow.sklearn.autolog(log_models=False)

    df = pd.read_csv("diabetes_preprocessing.csv")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5]
    }

    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    f1_macro = report["macro avg"]["f1-score"]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp = cm[0][0], cm[0][1]
    specificity = tn / (tn + fp)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("best_n_estimators", best_model.n_estimators)
    mlflow.log_param("best_max_depth", best_model.max_depth)

    mlflow.log_metric("f1_macro", f1_macro)
    mlflow.log_metric("specificity", specificity)

    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_model",
        signature=signature
    )

    print("✅ CI retraining via MLflow Project selesai")


if __name__ == "__main__":
    main()
