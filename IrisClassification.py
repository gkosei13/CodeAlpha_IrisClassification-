
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # works without a display; remove if running in a notebook
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = "outputs"
RANDOM_STATE = 42


def load_data(csv_path=None):
    """Load Iris data from a CSV (CodeAlpha download) or from scikit-learn."""
    if csv_path:
        df = pd.read_csv(csv_path)
        # Drop an ID column if present (the Kaggle-style Iris.csv has one)
        df = df.drop(columns=[c for c in df.columns if c.lower() == "id"])
        # Normalise column names to a consistent format
        df.columns = [c.lower().replace("cm", "").replace("(", "").replace(")", "")
                      .strip().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={df.columns[-1]: "species"})
        df["species"] = df["species"].str.replace("Iris-", "", regex=False)
    else:
        iris = load_iris(as_frame=True)
        df = iris.frame.copy()
        df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]
        df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
        df = df.drop(columns="target")
    return df


def explore(df):
    """Basic EDA: structure, missing values, class balance, and plots."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe().round(2))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df["species"].value_counts())

    # Pairplot: shows how well the species separate by feature
    pair = sns.pairplot(df, hue="species", corner=True)
    pair.figure.suptitle("Iris Feature Relationships by Species", y=1.02)
    pair.savefig(os.path.join(OUTPUT_DIR, "pairplot.png"), dpi=150,
                 bbox_inches="tight")
    plt.close("all")

    # Correlation heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(df.drop(columns="species").corr(), annot=True, cmap="coolwarm",
                fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def compare_models(X_train, X_test, y_train, y_test):
    """Train several classifiers, compare via cross-validation and test accuracy."""
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=200)),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }

    results = []
    fitted = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))
        results.append({
            "Model": name,
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "Test Accuracy": test_acc,
        })
        fitted[name] = model

    results_df = (pd.DataFrame(results)
                  .sort_values("Test Accuracy", ascending=False)
                  .reset_index(drop=True))
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    # Bar chart of test accuracy
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=results_df, x="Test Accuracy", y="Model", color="#4C72B0")
    plt.xlim(0.8, 1.02)
    plt.title("Model Comparison - Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()

    return fitted, results_df


def evaluate_best(model_name, model, X_test, y_test, labels):
    """Detailed evaluation of the chosen model."""
    y_pred = model.predict(X_test)
    print("\n" + "=" * 60)
    print(f"DETAILED EVALUATION: {model_name}")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred, target_names=labels))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def feature_importance(fitted, feature_names):
    """Show which measurements matter most (using the Random Forest)."""
    rf = fitted["Random Forest"]
    imp = (pd.Series(rf.feature_importances_, index=feature_names)
           .sort_values())
    plt.figure(figsize=(6, 4))
    imp.plot(kind="barh", color="#55A868")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("\nFeature importance (Random Forest):")
    print(imp.sort_values(ascending=False).round(3).to_string())


def predict_new_flower(model, feature_names):
    """Example: classify a brand-new flower from its measurements."""
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_names)
    print("\nExample prediction for", sample.values.tolist()[0],
          "->", model.predict(sample)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to Iris CSV (optional)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(args.csv)
    explore(df)

    X = df.drop(columns="species")
    y = df["species"]
    labels = sorted(y.unique())

    # Stratified split keeps the class proportions equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    fitted, results_df = compare_models(X_train, X_test, y_train, y_test)

    best_name = results_df.loc[0, "Model"]
    evaluate_best(best_name, fitted[best_name], X_test, y_test, labels)
    feature_importance(fitted, list(X.columns))
    predict_new_flower(fitted[best_name], list(X.columns))

    print(f"\nAll plots and results saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()