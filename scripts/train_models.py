
"""Train and evaluate text and image models, output metrics and figures."""
import numpy as np, json, pathlib, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score, ConfusionMatrixDisplay
)

# Ensure output dirs
pathlib.Path("figures").mkdir(exist_ok=True)

# TEXT MODEL
X_tr_text = __import__('scipy').sparse.load_npz("artefacts/X_train_text.npz")
X_te_text = __import__('scipy').sparse.load_npz("artefacts/X_test_text.npz")
y_tr_text = np.load("artefacts/y_train_text.npy")
y_te_text = np.load("artefacts/y_test_text.npy")

model_text = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
model_text.fit(X_tr_text, y_tr_text)
y_pred_text = model_text.predict(X_te_text)
y_proba_text = model_text.predict_proba(X_te_text)[:,1]

report_text = classification_report(y_te_text, y_pred_text, output_dict=True)
cm_text = confusion_matrix(y_te_text, y_pred_text)
precision_text, recall_text, _ = precision_recall_curve(y_te_text, y_proba_text)
auc_text = roc_auc_score(y_te_text, y_proba_text)

# Save text figures
plt.figure(); plt.plot(recall_text, precision_text, label=f"AUC={auc_text:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Text Model PR Curve")
plt.legend(); plt.savefig("figures/pr_curve_text.png"); plt.close()
fig, ax = plt.subplots(); ConfusionMatrixDisplay(cm_text).plot(ax=ax)
plt.title("Text Model Confusion Matrix"); fig.savefig("figures/cm_text.png"); plt.close()

# IMAGE MODEL
X_tr_img = np.load("artefacts/X_train_img.npy")
X_te_img = np.load("artefacts/X_test_img.npy")
y_tr_img = np.load("artefacts/y_train_img.npy")
y_te_img = np.load("artefacts/y_test_img.npy")

model_img = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_img.fit(X_tr_img, y_tr_img)
y_pred_img = model_img.predict(X_te_img)
y_proba_img = model_img.predict_proba(X_te_img)[:,1]

report_img = classification_report(y_te_img, y_pred_img, output_dict=True)
cm_img = confusion_matrix(y_te_img, y_pred_img)
precision_img, recall_img, _ = precision_recall_curve(y_te_img, y_proba_img)
auc_img = roc_auc_score(y_te_img, y_proba_img)

# Save image figures
plt.figure(); plt.plot(recall_img, precision_img, label=f"AUC={auc_img:.2f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Image Model PR Curve")
plt.legend(); plt.savefig("figures/pr_curve_img.png"); plt.close()
fig, ax = plt.subplots(); ConfusionMatrixDisplay(cm_img).plot(ax=ax)
plt.title("Image Model Confusion Matrix"); fig.savefig("figures/cm_img.png"); plt.close()

# Save results
results = {"text": {"report": report_text, "auc": auc_text},
           "image": {"report": report_img, "auc": auc_img}}
json.dump(results, open("results.json","w"), indent=2)

print("Training and evaluation complete. Results in results.json and figures/") 
