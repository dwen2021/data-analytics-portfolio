# src/training.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import joblib

categorical_features = ["pitch_name", "stand"]
numerical_features = [
    "release_extension", "release_pos_x", "release_pos_y", "release_pos_z",
    "release_speed", "release_spin_rate", "spin_axis", "plate_x", "plate_z",
    "pfx_x", "pfx_z", "balls", "strikes", "outs_when_up", "sz_top", "sz_bot"
]

df = pd.read_csv("data/raw/swing_data.csv")
X = df[categorical_features + numerical_features]
y = df["swing"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])


model = RandomForestClassifier(random_state=42)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


param_grid = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [20, 40, None],
    "model__min_samples_split": [4, 10, 20]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="neg_log_loss",
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)


y_test_proba = grid_search.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, y_test_proba)
brier = brier_score_loss(y_test, y_test_proba)
logloss = log_loss(y_test, y_test_proba)

print(f"ROC AUC: {roc:.3f}")
print(f"Brier Score: {brier:.3f}")
print(f"Log Loss: {logloss:.3f}")
print("Best Parameters:", grid_search.best_params_)

joblib.dump(grid_search.best_estimator_, "models/swing_probability_model.joblib")
print("✅ Model saved to models/swing_probability_model.joblib")
