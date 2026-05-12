import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "datasets", "student-mat.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, sep=";")

    print("Student Performance Dataset:")
    print(df.head())
    print("\nDataset shape:", df.shape)

    df["pass_fail"] = df["G3"].apply(lambda grade: 1 if grade >= 10 else 0)

    X = df.drop(["G3", "pass_fail"], axis=1)
    y = df["pass_fail"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Student Performance Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(RESULTS_DIR, "student_confusion_matrix.png"))
    plt.close()

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind="bar")
    plt.title("Student Performance Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "student_feature_importance.png"))
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()

