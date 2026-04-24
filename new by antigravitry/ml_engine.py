"""
ML Engine – Core machine-learning logic for the dashboard.
Handles: loading, preprocessing, training, evaluation, plotting, inference.
"""

import io
import base64
import pickle
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
)

# Classification models
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

# Regression models
from sklearn.linear_model    import LinearRegression, Ridge, Lasso
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor


# ─────────────────────────────────────────────────────────────────
# Hyper-parameter grids for GridSearchCV / RandomizedSearchCV
# ─────────────────────────────────────────────────────────────────
PARAM_GRIDS: Dict[str, Dict] = {
    "Logistic Regression":      {"C": [0.01, 0.1, 1, 10], "max_iter": [200, 500]},
    "Random Forest Classifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "SVM":                      {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
    "KNN":                      {"n_neighbors": [3, 5, 7, 11]},
    "Linear Regression":        {},          # no hyper-params to tune
    "Ridge":                    {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso":                    {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "Random Forest Regressor":  {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
    "Gradient Boosting Regressor": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
    "Gradient Boosting Classifier": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2]},
}

# Model factory
CLASSIFIERS: Dict[str, Any] = {
    "Logistic Regression":          LogisticRegression(max_iter=200),
    "Random Forest Classifier":     RandomForestClassifier(random_state=42),
    "SVM":                          SVC(probability=True, random_state=42),
    "KNN":                          KNeighborsClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
}

REGRESSORS: Dict[str, Any] = {
    "Linear Regression":            LinearRegression(),
    "Ridge":                        Ridge(),
    "Lasso":                        Lasso(),
    "Random Forest Regressor":      RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor":  GradientBoostingRegressor(random_state=42),
}


# ─────────────────────────────────────────────────────────────────
class MLEngine:
    """Encapsulates the entire ML pipeline state."""

    def __init__(self):
        self._reset()

    # ── internal reset ──────────────────────────────────────────
    def _reset(self):
        self.df:            Optional[pd.DataFrame] = None
        self.filepath:      Optional[str]          = None
        self.target:        Optional[str]          = None
        self.features:      List[str]              = []
        self.problem_type:  Optional[str]          = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler:        Optional[StandardScaler] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.trained_models: Dict[str, Any] = {}
        self.results:        Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.best_model:     Optional[Any]  = None
        self.dataset_info:   Dict[str, Any] = {}
        self.feature_names:  List[str]      = []

    # ── 1. Load dataset ─────────────────────────────────────────
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Read CSV and compute summary statistics."""
        self._reset()
        self.filepath = filepath
        self.df       = pd.read_csv(filepath)

        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)

        preview = self.df.head(5).to_dict(orient="records")

        # Convert non-serialisable types
        def _safe(v):
            if pd.isna(v):
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v

        preview = [{k: _safe(vv) for k, vv in row.items()} for row in preview]

        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

        self.dataset_info = {
            "shape":        {"rows": int(self.df.shape[0]), "cols": int(self.df.shape[1])},
            "columns":      list(self.df.columns),
            "dtypes":       dtypes,
            "missing":      {col: int(cnt) for col, cnt in missing.items()},
            "missing_pct":  {col: float(pct) for col, pct in missing_pct.items()},
            "preview":      preview,
            "numeric_cols": list(self.df.select_dtypes(include=np.number).columns),
            "cat_cols":     list(self.df.select_dtypes(include="object").columns),
        }
        return self.dataset_info

    # ── 2. Configure target / features ──────────────────────────
    def configure(self, target: str, features: List[str]) -> str:
        """Set target + features and auto-detect problem type."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        self.target   = target
        self.features = features
        self.problem_type = self._detect_problem_type(target)
        return self.problem_type

    def _detect_problem_type(self, target: str) -> str:
        """Classify → categorical / few uniques; Regression → continuous."""
        col = self.df[target]
        if col.dtype == "object" or col.nunique() <= 20:
            return "classification"
        return "regression"

    # ── 3. Preprocessing ────────────────────────────────────────
    def _preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        df = self.df[self.features + [self.target]].copy()

        # Fill missing values
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == "object":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)

        # Encode categorical features
        for col in self.features:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Encode target (classification only)
        if self.problem_type == "classification" and df[self.target].dtype == "object":
            le_target = LabelEncoder()
            df[self.target] = le_target.fit_transform(df[self.target].astype(str))
            self.label_encoders["__target__"] = le_target

        X = df[self.features].values
        y = df[self.target].values

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.feature_names = self.features
        return X, y

    # ── 4. Train models ──────────────────────────────────────────
    def train_models(self, selected: List[str], tune: bool = False) -> Dict[str, Any]:
        """Train all selected models and evaluate them."""
        X, y = self._preprocess()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_pool = CLASSIFIERS if self.problem_type == "classification" else REGRESSORS

        model_results: Dict[str, Any] = {}
        best_score: float = -np.inf
        self.best_model_name = None

        for name in selected:
            if name not in model_pool:
                continue
            import copy
            model = copy.deepcopy(model_pool[name])

            # Hyper-parameter tuning
            best_params: Dict = {}
            if tune and PARAM_GRIDS.get(name):
                try:
                    search = RandomizedSearchCV(
                        model, PARAM_GRIDS[name],
                        n_iter=min(10, len(list(PARAM_GRIDS[name].values())[0]) *
                                   len(list(PARAM_GRIDS[name].values()))),
                        cv=3, n_jobs=-1, random_state=42
                    )
                    search.fit(self.X_train, self.y_train)
                    model       = search.best_estimator_
                    best_params = search.best_params_
                except Exception:
                    model.fit(self.X_train, self.y_train)
            else:
                model.fit(self.X_train, self.y_train)

            y_pred = model.predict(self.X_test)

            # Metrics
            if self.problem_type == "classification":
                avg = "weighted"
                metrics = {
                    "accuracy":  float(round(accuracy_score(self.y_test, y_pred), 4)),
                    "precision": float(round(precision_score(self.y_test, y_pred, average=avg, zero_division=0), 4)),
                    "recall":    float(round(recall_score(self.y_test, y_pred, average=avg, zero_division=0), 4)),
                    "f1_score":  float(round(f1_score(self.y_test, y_pred, average=avg, zero_division=0), 4)),
                    "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
                }
                score = metrics["accuracy"]
            else:
                r2  = float(round(r2_score(self.y_test, y_pred), 4))
                mae = float(round(mean_absolute_error(self.y_test, y_pred), 4))
                mse = float(round(mean_squared_error(self.y_test, y_pred), 4))
                metrics = {"r2": r2, "mae": mae, "mse": mse}
                score   = r2

            model_results[name] = {
                "metrics":     metrics,
                "best_params": best_params,
                "score":       score,
            }

            self.trained_models[name] = model

            if score > best_score:
                best_score           = score
                self.best_model_name = name
                self.best_model      = model

        self.results = {
            "problem_type": self.problem_type,
            "model_results": model_results,
            "best_model":    self.best_model_name,
            "best_score":    float(best_score),
        }
        return self.results

    # ── 5. Persist best model ────────────────────────────────────
    def save_model(self, path: str):
        """Pickle the best model, scaler, and encoders together."""
        bundle = {
            "model":          self.best_model,
            "scaler":         self.scaler,
            "label_encoders": self.label_encoders,
            "features":       self.feature_names,
            "problem_type":   self.problem_type,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    # ── 6. Inference ─────────────────────────────────────────────
    def predict(self, input_values: Dict[str, Any]) -> Tuple[Any, Optional[float]]:
        """Run the best model on new user input."""
        if self.best_model is None:
            raise ValueError("No model trained yet.")

        row = []
        for feat in self.feature_names:
            val = input_values.get(feat, 0)
            if feat in self.label_encoders:
                le = self.label_encoders[feat]
                try:
                    val = le.transform([str(val)])[0]
                except ValueError:
                    val = 0
            row.append(float(val))

        X_new = np.array(row).reshape(1, -1)
        X_new = self.scaler.transform(X_new)

        prediction = self.best_model.predict(X_new)[0]
        confidence = None

        # Decode label if classification
        if self.problem_type == "classification":
            if hasattr(self.best_model, "predict_proba"):
                proba = self.best_model.predict_proba(X_new)[0]
                confidence = float(round(max(proba) * 100, 2))
            if "__target__" in self.label_encoders:
                prediction = self.label_encoders["__target__"].inverse_transform(
                    [int(prediction)]
                )[0]
            else:
                prediction = int(prediction)
        else:
            prediction = float(round(prediction, 4))

        return prediction, confidence

    # ── 7. Plots ─────────────────────────────────────────────────
    def generate_plots(self) -> Dict[str, str]:
        """Return base64-encoded PNG plots as a dict."""
        plots: Dict[str, str] = {}

        if self.best_model is None:
            return plots

        y_pred = self.best_model.predict(self.X_test)

        sns.set_theme(style="darkgrid", palette="muted")

        # ── Confusion matrix (classification) ──────────────────
        if self.problem_type == "classification":
            cm = confusion_matrix(self.y_test, y_pred)
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=0.5, linecolor="white")
            ax.set_title(f"Confusion Matrix – {self.best_model_name}", fontsize=14, pad=12)
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label",      fontsize=11)
            plt.tight_layout()
            plots["confusion_matrix"] = self._fig_to_b64(fig)

        # ── Actual vs Predicted (regression) – ALL data points ─
        if self.problem_type == "regression":
            y_pred_train = self.best_model.predict(self.X_train)
            y_pred_test  = y_pred

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(self.y_train, y_pred_train, alpha=0.45, color="#a78bfa",
                       edgecolors="white", linewidths=0.4, label=f"Train ({len(self.y_train)})")
            ax.scatter(self.y_test, y_pred_test, alpha=0.7, color="#4a9eff",
                       edgecolors="white", linewidths=0.4, label=f"Test ({len(self.y_test)})")

            all_actual = np.concatenate([self.y_train, self.y_test])
            all_pred   = np.concatenate([y_pred_train, y_pred_test])
            mn = min(all_actual.min(), all_pred.min())
            mx = max(all_actual.max(), all_pred.max())
            ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect Fit")

            ax.set_title(f"Actual vs Predicted – {self.best_model_name} (All Data)", fontsize=14, pad=12)
            ax.set_xlabel("Actual",    fontsize=11)
            ax.set_ylabel("Predicted", fontsize=11)
            ax.legend()
            plt.tight_layout()
            plots["actual_vs_predicted"] = self._fig_to_b64(fig)

        # ── Feature importance (tree-based models) ─────────────
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            indices     = np.argsort(importances)[::-1]
            names       = [self.feature_names[i] for i in indices]

            fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
            bars   = ax.barh(names[::-1], importances[indices][::-1], color=colors[::-1])
            ax.set_title(f"Feature Importance – {self.best_model_name}", fontsize=14, pad=12)
            ax.set_xlabel("Importance", fontsize=11)
            plt.tight_layout()
            plots["feature_importance"] = self._fig_to_b64(fig)

        # ── Model comparison bar chart ──────────────────────────
        if self.results.get("model_results"):
            model_names  = list(self.results["model_results"].keys())
            scores       = [self.results["model_results"][m]["score"] for m in model_names]
            metric_label = "Accuracy" if self.problem_type == "classification" else "R² Score"

            fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.4), 5))
            bar_colors = ["#4a9eff" if m != self.best_model_name else "#00d4aa"
                          for m in model_names]
            bars = ax.bar(model_names, scores, color=bar_colors, width=0.5,
                          edgecolor="white", linewidth=0.8)
            ax.set_title("Model Performance Comparison", fontsize=14, pad=12)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_ylim(0, 1.05)
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{score:.3f}", ha="center", va="bottom", fontsize=9)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plots["model_comparison"] = self._fig_to_b64(fig)

        # ── ROC-AUC area plot (binary classification) ───────────
        if (self.problem_type == "classification" and
                hasattr(self.best_model, "predict_proba")):
            try:
                from sklearn.metrics import roc_curve, auc
                from sklearn.preprocessing import label_binarize

                classes = np.unique(self.y_test)
                if len(classes) == 2:
                    proba    = self.best_model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, proba)
                    roc_auc  = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.plot(fpr, tpr, color="#4a9eff", lw=2,
                            label=f"ROC Curve (AUC = {roc_auc:.3f})")
                    ax.fill_between(fpr, tpr, alpha=0.15, color="#4a9eff")
                    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
                    ax.set_title(f"ROC Curve – {self.best_model_name}", fontsize=14, pad=12)
                    ax.set_xlabel("False Positive Rate", fontsize=11)
                    ax.set_ylabel("True Positive Rate",  fontsize=11)
                    ax.legend(loc="lower right")
                    plt.tight_layout()
                    plots["roc_curve"] = self._fig_to_b64(fig)
            except Exception:
                pass

        # ── Residuals plot (regression) ─────────────────────────
        if self.problem_type == "regression":
            residuals = self.y_test - y_pred
            fig, ax   = plt.subplots(figsize=(7, 5))
            ax.scatter(y_pred, residuals, alpha=0.6, color="#f97316", edgecolors="white", linewidths=0.4)
            ax.axhline(0, color="red", linestyle="--", lw=2)
            ax.set_title(f"Residuals Plot – {self.best_model_name}", fontsize=14, pad=12)
            ax.set_xlabel("Predicted Values", fontsize=11)
            ax.set_ylabel("Residuals",        fontsize=11)
            plt.tight_layout()
            plots["residuals"] = self._fig_to_b64(fig)

        return plots

    # ── Utilities ────────────────────────────────────────────────
    @staticmethod
    def _fig_to_b64(fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor="#1a1a2e", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def get_last_results(self) -> Dict[str, Any]:
        if not self.results:
            raise ValueError("No results available.")
        return self.results

    def get_dataset_info(self) -> Dict[str, Any]:
        if not self.dataset_info:
            raise ValueError("No dataset loaded.")
        return self.dataset_info
