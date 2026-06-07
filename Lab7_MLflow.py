import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=1
    )

    mlflow.set_experiment("Lab7_Klasyfikacja")

    n_estimators = 100
    max_depth = 5

    with mlflow.start_run():
        print(f"Trenowanie modelu (n_estimators={n_estimators}, max_depth={max_depth})...")

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"Wynik modelu (Accuracy): {acc:.4f}")

if __name__ == "__main__":
    main()
