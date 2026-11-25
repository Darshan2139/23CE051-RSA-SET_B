"""
Responsible and Safe AI (AIML4001) - External Practical, Set B
Explaining Instant EMI-Finance Decisions at an Electronics Showroom
Dataset : German Credit (UCI Statlog, 1000 applicants, 20 attributes) -> german_credit.csv
Tools   : Python 3, Pandas, Scikit-learn, SHAP, LIME, Matplotlib
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

import shap
from lime.lime_tabular import LimeTabularExplainer

RS = 42
np.random.seed(RS)
SENSITIVE = ["age", "personal_status", "foreign_worker"]

# ====================================================================
# TASK 1 : Load, encode categorical attributes, stratified 80:20 split
# ====================================================================
df = pd.read_csv("german_credit.csv")
print("Dataset shape :", df.shape)
print("Class counts  :", df["class"].value_counts().to_dict())

y = (df["class"] == "bad").astype(int)          # bad = positive class (1)
X_raw = df.drop(columns="class")

cat_cols = X_raw.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X_raw.columns if c not in cat_cols]
print("Categorical   :", len(cat_cols), cat_cols)
print("Numerical     :", len(num_cols), num_cols)

enc = OrdinalEncoder()                            # one column per original attribute
X = X_raw.copy()
X[cat_cols] = enc.fit_transform(X_raw[cat_cols]).astype(int)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RS)
X_test_raw = X_raw.loc[X_test.index]             # readable values for the same rows
print(f"Train : {X_train.shape}  bad-rate={y_train.mean():.2f}")
print(f"Test  : {X_test.shape}  bad-rate={y_test.mean():.2f}")

# ====================================================================
# TASK 2 : Random Forest (200 trees, depth 8) - accuracy, ROC-AUC
# ====================================================================
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RS, n_jobs=-1)
rf.fit(X_train, y_train)

proba_bad = rf.predict_proba(X_test)[:, 1]
y_pred = (proba_bad >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, proba_bad)

print("\nModel : RandomForest (200 trees, depth 8)")
print(f"Test accuracy : {acc:.2f}      ROC-AUC : {auc:.2f}")
print("Confusion matrix [[TN FP][FN TP]] (positive = bad):\n", confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["good", "bad"], digits=3))

# ====================================================================
# TASK 3 : SHAP (TreeExplainer) global importance + summary plot
# ====================================================================
explainer = shap.TreeExplainer(rf)
sv_all = explainer.shap_values(X_test)            # (n, features, 2) or list of 2 arrays
sv = sv_all[1] if isinstance(sv_all, list) else sv_all[:, :, 1]   # class = bad
base_bad = np.atleast_1d(explainer.expected_value)[1]

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=feature_names).sort_values(ascending=False)
top5 = mean_abs.head(5)

print("\nGLOBAL FEATURE IMPORTANCE (mean |SHAP|, class = bad)")
print(f"{'RANK':<5}{'FEATURE':<24}{'MEAN|SHAP|':>10}")
print("-" * 39)
for r, (f, v) in enumerate(top5.items(), 1):
    print(f"{r:>3}  {f:<24}{v:>10.3f}")
print("-" * 39)

plt.figure()
shap.summary_plot(sv, X_test, feature_names=feature_names, show=False, max_display=20)
plt.title("SHAP summary (beeswarm) - class = bad", fontsize=11)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

plt.figure()
shap.summary_plot(sv, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
plt.title("Mean |SHAP| feature importance - class = bad", fontsize=11)
plt.tight_layout(); plt.savefig("shap_summary_bar.png", dpi=150, bbox_inches="tight"); plt.close()
print("Plot saved : shap_summary.png, shap_summary_bar.png")

# ====================================================================
# TASK 4 : One rejected applicant -> SHAP waterfall, top 3 reasons
# ====================================================================
# Rule: the first applicant in the test set whose predicted class is "bad" (P(bad) >= 0.5)
pos = int(np.where(y_pred == 1)[0][0])            # position inside the test set
row_id = X_test.index[pos]                       # original row number in german_credit.csv
x_enc = X_test.iloc[pos]
x_raw = X_test_raw.iloc[pos]
p_bad = proba_bad[pos]

print(f"\nApplicant (test index {pos}, CSV row {row_id}) : P(bad) = {p_bad:.2f}  ->  REJECTED")
print(f"Actual label of this applicant : {'bad' if y_test.iloc[pos] == 1 else 'good'}")
print("Applicant profile:\n", x_raw.to_string())

def fv(f):                                       # "feature=value" in readable form
    return f"{f}={x_raw[f]}"

app_sv = pd.Series(sv[pos], index=feature_names)
shap_reasons = app_sv[app_sv > 0].sort_values(ascending=False)   # push towards "bad"
shap_top3 = shap_reasons.head(3)
print("\nSHAP top 3 reasons (positive = pushes towards rejection):")
for f, v in shap_top3.items():
    print(f"  {fv(f):<45}{v:+.3f}")
print(f"Check: base value {base_bad:.3f} + sum(SHAP) {app_sv.sum():+.3f} = {base_bad + app_sv.sum():.3f}"
      f"  (model P(bad) = {p_bad:.3f})")

exp_one = shap.Explanation(values=sv[pos], base_values=base_bad,
                           data=x_raw.values, feature_names=feature_names)
plt.figure()
shap.plots.waterfall(exp_one, max_display=12, show=False)
plt.title(f"SHAP waterfall - test applicant {pos}  (P(bad) = {p_bad:.2f})", fontsize=11)
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight"); plt.close()

shap.force_plot(base_bad, sv[pos], x_raw.values, feature_names=feature_names,
                matplotlib=True, show=False, text_rotation=15)
plt.savefig("shap_force.png", dpi=150, bbox_inches="tight"); plt.close()
print("Plot saved : shap_waterfall.png, shap_force.png")

# ====================================================================
# TASK 5 : LIME (num_features = 5) for the same applicant + comparison
# ====================================================================
cat_idx = [feature_names.index(c) for c in cat_cols]
cat_names = {feature_names.index(c): list(enc.categories_[i]) for i, c in enumerate(cat_cols)}

def predict_fn(arr):
    return rf.predict_proba(pd.DataFrame(arr, columns=feature_names))

lime_exp = LimeTabularExplainer(
    training_data=X_train.values, feature_names=feature_names,
    class_names=["good", "bad"], categorical_features=cat_idx,
    categorical_names=cat_names, discretize_continuous=True,
    mode="classification", random_state=RS)

lexp = lime_exp.explain_instance(x_enc.values, predict_fn, num_features=5,
                                 labels=(1,), num_samples=5000)
lime_list = lexp.as_list(label=1)                        # ranked by |weight|
lime_feat = [feature_names[i] for i, _ in lexp.as_map()[1]]

print("\nLIME top 5 (weight towards P(bad)):")
for (rule, w) in lime_list:
    print(f"  {rule:<45}{w:+.3f}")
print(f"LIME local-model R^2 = {lexp.score:.2f}, local prediction = {lexp.local_pred[0]:.2f}")

fig = lexp.as_pyplot_figure(label=1)
fig.set_size_inches(8, 4); plt.title(f"LIME explanation - test applicant {pos} (class = bad)")
plt.tight_layout(); plt.savefig("lime_explanation.png", dpi=150, bbox_inches="tight"); plt.close()
print("Plot saved : lime_explanation.png")

shap_top3_f = list(shap_top3.index)
lime_top3_f = lime_feat[:3]
agree = sorted(set(shap_top3_f) & set(lime_top3_f))
print("\nSHAP (top 3)                                 LIME (top 5)")
print("-" * 92)
for i in range(5):
    left = f"{fv(shap_top3_f[i])} {shap_top3.iloc[i]:+.3f}" if i < 3 else ""
    right = f"{lime_list[i][0]} {lime_list[i][1]:+.3f}"
    print(f"{left:<45}{right}")
print("-" * 92)
print(f"Top-3 agreement (SHAP vs LIME) : {len(agree)} / 3   common = {agree}")

# ====================================================================
# TASK 6 : Plain-language rejection message + sensitive-attribute check
# ====================================================================
lime_pos_feat = [feature_names[i] for i, w in lexp.as_map()[1] if w > 0]
reason_feats = list(dict.fromkeys(shap_top3_f + lime_pos_feat))   # all reasons shown
sensitive_found = [f for f in reason_feats if f in SENSITIVE]
print("\nSensitive attributes in reasons :",
      (", ".join(sensitive_found) + " -> REVIEW") if sensitive_found else "none -> OK")

def phrase(f):
    v = x_raw[f]
    t = {
        "checking_status": {"<0": "your current (checking) account balance is negative",
                            "0<=X<200": "your current account balance is low (below 200 DM)",
                            "no checking": "you do not have a current account with a bank we can verify",
                            ">=200": "of your current account position"}.get(v),
        "duration": f"the requested repayment period ({v} months) is long",
        "credit_amount": f"the requested loan amount ({v} DM) is high",
        "savings_status": {"<100": "your savings balance is low (below 100 DM)",
                           "100<=X<500": "your savings balance is modest",
                           "no known savings": "we could not verify any savings"}.get(v, "of your savings level"),
        "credit_history": f"of your past credit record ({v})",
        "purpose": f"of the loan purpose ({v})",
        "employment": f"your time in current employment is short ({v} years)",
        "installment_commitment": f"the instalment would take a large share of your income (rate {v} of 4)",
        "property_magnitude": f"of limited collateral/property ({v})",
        "other_payment_plans": f"of other running instalment plans ({v})",
        "housing": f"of your housing situation ({v})",
        "existing_credits": f"you already have {v} running loan(s)",
        "other_parties": "there is no guarantor or co-applicant",
        "job": f"of your job category ({v})",
        "own_telephone": "no registered telephone contact was provided",
        "residence_since": f"of a short time at your current address ({v} years)",
        "num_dependents": f"of the number of dependants ({v})",
    }.get(f)
    return t or f"of your {f.replace('_', ' ')}"

legit = [f for f in shap_reasons.index if f not in SENSITIVE][:2]   # top legitimate SHAP reasons
msg = ("Your EMI request was not approved mainly because " +
       " and ".join(phrase(f) for f in legit) + ". "
       "You may re-apply after improving these factors, or ask for a smaller amount / shorter tenure. "
       "You can request a manual review of this decision at the counter.")
print("\nMessage (legitimate reasons only):\n" + msg)

# ---- Extra check: does the decision survive without the sensitive attributes? ----
sens_contrib = float(app_sv[SENSITIVE].sum())
print(f"\nSHAP contribution of sensitive attributes for this applicant : {sens_contrib:+.3f}")
print(f"P(bad) with that contribution removed : {p_bad - sens_contrib:.3f}")

keep = [c for c in feature_names if c not in SENSITIVE]
rf_fair = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RS, n_jobs=-1)
rf_fair.fit(X_train[keep], y_train)
pf = rf_fair.predict_proba(X_test[keep])[:, 1]
acc_f, auc_f = accuracy_score(y_test, (pf >= 0.5).astype(int)), roc_auc_score(y_test, pf)
print(f"Model WITHOUT {SENSITIVE}: accuracy {acc_f:.2f}  ROC-AUC {auc_f:.2f}  "
      f"-> applicant {pos}: P(bad) = {pf[pos]:.2f} "
      f"({'still REJECTED' if pf[pos] >= 0.5 else 'now APPROVED'})")

# ---- save a machine-readable summary for the report ----------------
import json
summary = dict(
    acc=acc, auc=auc, cm=confusion_matrix(y_test, y_pred).tolist(),
    top5=[(f, float(v)) for f, v in top5.items()],
    all_imp=[(f, float(v)) for f, v in mean_abs.items()],
    pos=pos, row_id=int(row_id), p_bad=float(p_bad), actual=int(y_test.iloc[pos]),
    base=float(base_bad), shap_sum=float(app_sv.sum()),
    profile={k: (str(v)) for k, v in x_raw.items()},
    shap_top3=[(fv(f), f, float(v)) for f, v in shap_top3.items()],
    shap_all=[(fv(f), float(v)) for f, v in app_sv.sort_values(key=abs, ascending=False).items()],
    lime=[(r, float(w)) for r, w in lime_list], lime_feat=lime_feat,
    lime_r2=float(lexp.score), lime_local=float(lexp.local_pred[0]),
    agree=agree, sensitive=sensitive_found, legit=legit, msg=msg,
    n_train=len(X_train), n_test=len(X_test), n_reject=int(y_pred.sum()),
    sens_contrib=sens_contrib, acc_f=acc_f, auc_f=auc_f, p_fair=float(pf[pos]),
    sv_sens={f: float(app_sv[f]) for f in SENSITIVE},
)
with open("results.json", "w") as fh:
    json.dump(summary, fh, indent=2)
