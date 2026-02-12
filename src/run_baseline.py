from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "artifacts").mkdir(exist_ok=True)

    df = pd.read_csv(root / "data" / "diabetes.csv")
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    (root / "results" / "metrics_diabetes_baseline.json").write_text(
        json.dumps(metrics, indent=2)
    )
    joblib.dump(model, root / "results" / "artifacts" / "diabetes_logreg.joblib")
    print(metrics)


if __name__ == "__main__":
    main()
