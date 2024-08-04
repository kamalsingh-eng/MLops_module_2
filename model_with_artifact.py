import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Set the MLflow tracking URI to PostgreSQL
#mlflow.set_tracking_uri('postgresql://user:password@localhost/mlflow_db')

# Load data from CSV
data = pd.read_csv('fetal_health.csv')

# Assuming the last column is the target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Track experiments
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("max_depth", model.max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    # Log an artifact (Feature importance plot)
    importances = model.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features[:20], importances[:20])  # Displaying top 20 feature importances
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importance Plot')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')

