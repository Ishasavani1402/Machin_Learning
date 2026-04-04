import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (safe for scripts)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import warnings
warnings.filterwarnings('ignore')


# ── 1. Single Model Evaluation ────────────────────────────────────────────────
def evaluate_model(name: str, pipe, X_test, y_test) -> dict:
    """Evaluate one pipeline and return metrics dict."""
    y_pred       = pipe.predict(X_test)
    y_proba      = pipe.predict_proba(X_test)[:, 1]

    accuracy     = accuracy_score(y_test, y_pred)
    roc_auc      = roc_auc_score(y_test, y_proba)
    report       = classification_report(y_test, y_pred, output_dict=True)
    cm           = confusion_matrix(y_test, y_pred)

    result = {
        'Model'     : name,
        'Accuracy'  : round(accuracy, 4),
        'Precision' : round(report['1']['precision'], 4),
        'Recall'    : round(report['1']['recall'],    4),
        'F1-Score'  : round(report['1']['f1-score'],  4),
        'AUC-ROC'   : round(roc_auc, 4),
        'y_pred'    : y_pred,
        'y_proba'   : y_proba,
    }

    # Print detailed report
    print(f"\n{'='*50}")
    print(f"  Model        : {name}")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  AUC-ROC      : {roc_auc:.4f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    print("Confusion Matrix:")
    print(cm)
    print(f"{'='*50}\n")

    return result


# ── 2. Evaluate All Models ────────────────────────────────────────────────────
def evaluate_all(pipelines: dict, X_test, y_test) -> list:
    """Evaluate all pipelines and return list of result dicts."""
    print("\n[INFO] ─ Evaluating All Models ─")
    results = []
    for name, pipe in pipelines.items():
        result = evaluate_model(name, pipe, X_test, y_test)
        results.append(result)
    return results


# ── 3. Comparison Table ───────────────────────────────────────────────────────
def print_comparison_table(results: list) -> pd.DataFrame:
    """Print and return a sorted comparison dataframe."""
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_results = pd.DataFrame(results)[cols].sort_values('AUC-ROC', ascending=False)

    print("\n📊 Model Comparison (sorted by AUC-ROC):")
    print(df_results.to_string(index=False))
    return df_results


# ── 4. ROC Curve Plot ─────────────────────────────────────────────────────────
def plot_roc_curves(results: list, y_test, save_path: str = "outputs/roc_curves.png") -> None:
    """Plot ROC curves for all models and save."""
    plt.figure(figsize=(8, 6))

    colors = ['steelblue', 'seagreen', 'darkorange']
    for i, result in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        auc_score   = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 color=colors[i % len(colors)],
                 label=f"{result['Model']} (AUC = {auc_score:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — Fraud Detection")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] ROC curve saved → {save_path}")


# ── 5. Feature Importance Plot ────────────────────────────────────────────────
def plot_feature_importance(pipelines: dict, feature_names: list,
                             save_path: str = "outputs/feature_importance.png") -> None:
    """Plot top-10 feature importance for all 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Top 10 Feature Importance — All Models',
                 fontsize=16, fontweight='bold', y=1.02)

    model_configs = [
        ('Logistic Regression',      'steelblue',   'Absolute Coefficients'),
        ('Random Forest Classifier', 'seagreen',    'Gini Importance'),
        ('XGBoost Classifier',       'darkorange',  'F-Score Importance'),
    ]

    for ax, (name, color, method) in zip(axes, model_configs):
        if name not in pipelines:
            continue

        pipe  = pipelines[name]
        model = pipe.named_steps['model']

        # Get importance values per model type
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            importance = model.feature_importances_

        indices  = np.argsort(importance)[::-1][:10]
        features = [feature_names[i] for i in indices]
        values   = importance[indices]

        bars = ax.barh(features[::-1], values[::-1], color=color, edgecolor='white')
        ax.set_title(f"{name}\n({method})", fontweight='bold')
        ax.set_xlabel('Importance Score')

        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Feature importance chart saved → {save_path}")


# ── 6. Confusion Matrix Plot ──────────────────────────────────────────────────
# def plot_confusion_matrices(pipelines: dict, X_test, y_test,
#                              save_path: str = "outputs/confusion_matrices.png") -> None:
#     """Plot confusion matrices for all models side by side."""
#     n      = len(pipelines)
#     fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
#     if n == 1:
#         axes = [axes]

#     for ax, (name, pipe) in zip(axes, pipelines.items()):
#         y_pred = pipe.predict(X_test)
#         cm     = confusion_matrix(y_test, y_pred)

#         im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
#         ax.set_title(name, fontweight='bold')
#         ax.set_xlabel('Predicted Label')
#         ax.set_ylabel('True Label')
#         ax.set_xticks([0, 1]); ax.set_xticklabels(['Not Fraud', 'Fraud'])
#         ax.set_yticks([0, 1]); ax.set_yticklabels(['Not Fraud', 'Fraud'])

#         for i in range(2):
#             for j in range(2):
#                 ax.text(j, i, f'{cm[i, j]:,}',
#                         ha='center', va='center',
#                         color='white' if cm[i, j] > cm.max() / 2 else 'black',
#                         fontsize=13, fontweight='bold')
#         plt.colorbar(im, ax=ax)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"[INFO] Confusion matrices saved → {save_path}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] model_evaluation.py loaded successfully.")
    print("       Import and call evaluate_all(), plot_roc_curves(), etc.")

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# from sklearn.metrics import accuracy_score , classification_report , confusion_matrix , roc_auc_score, roc_curve ,auc


# def model_pipeline(x_train , y_train):
#     Pipe_lr = Pipeline([
#     ('scaler' , StandardScaler()),
#     ('model' , LogisticRegression(
#         max_iter=300,
#         random_state=42,
#         class_weight='balanced',
#         n_jobs=-1 
#     ))
# ])
    
#     pipe_rf = Pipeline([
#     ('scaler' , StandardScaler()),
#     ('model' , RandomForestClassifier(
#         n_estimators=50,
#         max_depth=10,
#         min_samples_leaf=100,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#     ))
# ])
    
#     fraud_count     = (y_train == 0).sum()
#     non_fraud_count = (y_train == 1).sum()
#     scale_pos       = fraud_count / non_fraud_count  
#     scale_pos  = fraud_count / non_fraud_count   # auto balance ratio



#     pipe_xg = Pipeline([
#     ('scaler' , StandardScaler()),
#     ('model' , xgb.XGBClassifier(
#         n_estimators = 100,
#         max_depth = 6,
#         learning_rate = 0.1,
#         subsample = 0.8,
#         tree_method = 'hist',
#         n_jobs = -1,
#         scale_pos_weight = scale_pos
#     ))
# ])
    
#     train_pipelines = {
#     'Logistic Regression' : Pipe_lr,
#     'Random Forest Classifier' : pipe_rf,
#     "XGBoost Classifier" : pipe_xg
# }

#     for name , pipe in train_pipelines.items():
#         print(f"train : {name}")
#         pipe.fit(x_train , y_train)
#         print(f"model : {name} trained sucessfully")

# def evaluate_model(train_pipelines , x_test , y_test):
#     results = []
#     for name , pipe in train_pipelines.items():
#         y_pred = pipe.predict(x_test)
#         accuracy =  accuracy_score(y_test , y_pred)
#         y_pred_proba = pipe.predict_proba(x_test)[:,1]
#         roc_auc = roc_auc_score(y_test , y_pred_proba)
#         report = classification_report(y_test , y_pred , output_dict=True)

#         results.append({
#             'Model'     : name,
#             'Precision' : round(report['1']['precision'], 4),
#             'Recall'    : round(report['1']['recall'],    4),
#             'F1-Score'  : round(report['1']['f1-score'],  4), # i need to understand this 
#             'AUC-ROC'   : round(roc_auc, 4)
#         })

#         print(f"model : {name}")
#         # print(f"prediction : {y_pred}")
#         print(f"accuracy : {accuracy}")
#         print(f"ROC_AUC : {roc_auc}")
#         print(f"{'='*45}")
#         print(f"classification report : \n {classification_report(y_test , y_pred)}")
#         print(f"{'='*45}")
#         print("confusion matrix:")
#         print(confusion_matrix(y_test , y_pred))
#         print(f"{'='*45} \n ")

