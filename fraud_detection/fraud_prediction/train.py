import os
import argparse
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
import xgboost as xgb

# Local modules
from data_preprocess  import preprocess
from model_evaluation import (
    evaluate_all,
    plot_roc_curves,
    plot_feature_importance,
)


# ── Config 
DEFAULT_DATA_PATH  = "train_data.csv"
MODELS_DIR         = "models"
OUTPUTS_DIR        = "charts"


# ── 1. Build Pipelines 
def build_pipelines(scale_pos: float) -> dict:
    """Construct all model pipelines."""
    print("[INFO] Building pipelines...")

    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(
            max_iter=300,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])

    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipe_xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            tree_method='hist',
            scale_pos_weight=scale_pos,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    pipelines = {
        'Logistic Regression'      : pipe_lr,
        'Random Forest Classifier' : pipe_rf,
        'XGBoost Classifier'       : pipe_xgb,
    }

    print(f"[INFO] {len(pipelines)} pipelines ready: {list(pipelines.keys())}")
    return pipelines


# ── 2. Train All Pipelines 
def train_all(pipelines: dict, X_train, y_train) -> dict:
    """Fit all pipelines on training data."""
    print("\n[INFO] ── Training Models ───────────────────────────")
    for name, pipe in pipelines.items():
        print(f"  Training → {name} ...")
        pipe.fit(X_train, y_train)
        print(f"  ✅ Done   → {name}")
    print()
    return pipelines


# ── 3. Save Best Model 
def save_best_model(pipelines: dict, results: list) -> str:
    """
    Pick best model by AUC-ROC and save it.
    Also saves all models individually.
    Returns name of best model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Find best by AUC-ROC
    best = max(results, key=lambda r: r['AUC-ROC'])
    best_name = best['Model']
    best_pipe = pipelines[best_name]

    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipe, best_path)

    print(f"\n🏆 Best Model  : {best_name}")
    print(f"   AUC-ROC     : {best['AUC-ROC']}")
    print(f"   F1-Score    : {best['F1-Score']}")
    print(f"   Saved to    : {best_path}\n")

    return best_name


# ── 4. Generate All Plots 
def generate_plots(pipelines: dict, results: list,
                   feature_names: list, X_test, y_test) -> None:
    """Generate and save all evaluation charts."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("[INFO] Generating charts...")
    plot_roc_curves(
        results, y_test,
        save_path=os.path.join(OUTPUTS_DIR, "roc_curves.png")
    )
    plot_feature_importance(
        pipelines, feature_names,
        save_path=os.path.join(OUTPUTS_DIR, "feature_importance.png")
    )
    print("[INFO] All charts saved to outputs/\n")


# ── 5. Load Saved Model (for inference) 
def load_best_model(model_path: str = None):
    """Load best saved model for prediction."""
    path = model_path or os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    pipe = joblib.load(path)
    print(f"[INFO] Loaded model from: {path}")
    return pipe


# ── 6. Main 
def main(data_path: str = DEFAULT_DATA_PATH) -> None:

    print("=" * 55)
    print("   FRAUD DETECTION — ML PIPELINE")
    print("=" * 55)

    # Step 1 — Preprocess
    X_train, X_test, y_train, y_test, feature_names, scale_pos = preprocess(data_path)

    # Step 2 — Build pipelines
    pipelines = build_pipelines(scale_pos)

    # Step 3 — Train
    pipelines = train_all(pipelines, X_train, y_train)

    # Step 4 — Evaluate
    results = evaluate_all(pipelines, X_test, y_test)

    # Step 5 — Save best model
    best_name = save_best_model(pipelines, results)

    # Step 6 — Generate plots
    generate_plots(pipelines, results, feature_names, X_test, y_test)

    print("=" * 55)
    print(f"   ✅ PIPELINE COMPLETE")
    print(f"   Best Model   : {best_name}")
    print(f"   Models saved : {MODELS_DIR}/")
    print(f"   Charts saved : {OUTPUTS_DIR}/")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument(
        '--data',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to training CSV (default: {DEFAULT_DATA_PATH})"
    )
    args = parser.parse_args()
    main(data_path=args.data)