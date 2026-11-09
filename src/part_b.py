"""Part B: RFM segmentation (K-means + PCA) and lapse prediction (logistic regression / random forest)."""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.makedirs("outputs", exist_ok=True)
SEED = 42
FINAL_K = 4

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")
comp = orders[orders["status"] == "completed"]

# ======================================================================== task 1: RFM
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"last_ts": g["order_ts"].max(),
                    "frequency": g["order_id"].nunique(),
                    "monetary": g["order_amount"].sum()})
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
print("=== RFM ===")
print(f"customers {len(rfm)} | recency median {rfm.recency.median():.0f} min {rfm.recency.min()} max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()} | "
      f"monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ======================================================================== task 2: K-means
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== K-means sweep ===\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    print(f"{k}  {km.inertia_:8.1f}  {silhouette_score(X, km.labels_):.3f}")

# ======================================================================== task 3: profile, name, PCA
km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=SEED).fit(X)
rfm["segment_id"] = km.labels_
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print(f"\n=== Segment profile (K={FINAL_K}) ===\n{prof.to_string()}")

# names are assigned from the profile, never from the arbitrary cluster number
premium_id = prof["monetary"].idxmax()
rest = prof.drop(index=premium_id)
loyal_id = rest["frequency"].idxmax()
rest = rest.drop(index=loyal_id)
atrisk_id = rest["recency"].idxmax()
names = {premium_id: "Premium Big Spenders", loyal_id: "Loyal Regulars", atrisk_id: "Lapsing / At-Risk"}
for sid in prof.index:
    names.setdefault(sid, "Occasional Buyers")
rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
print({int(k): v for k, v in names.items()}, "| premium segment id:", int(premium_id))
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    "outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X)
evr = pca.explained_variance_ratio_
print(f"PCA explained variance ratio: {evr[0]:.3f}, {evr[1]:.3f}, total {evr.sum():.3f}")

# ======================================================================== task 4: lapse problem
CUTOFF = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")
pre = orders[orders["order_ts"] < CUTOFF].merge(products[["product_id", "category"]], on="product_id")   # only pre-cutoff rows
pre_c = pre[pre["status"] == "completed"]
cohort_ids = pre_c["customer_id"].unique()
future = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < END)]
active_sep = set(future["customer_id"])

pg = pre_c.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - pg["order_ts"].max()).dt.days,
    "frequency": pg["order_id"].nunique(),
    "monetary": pg["order_amount"].sum(),
    "avg_discount": pg["discount_pct"].mean(),
    "avg_delivery_min": pg["delivery_minutes"].mean(),
    "fav_category": pg["category"].agg(lambda s: s.value_counts().sort_index().sort_values(ascending=False, kind="stable").index[0]),
})
final = pre[pre["status"].isin(["completed", "cancelled"])]
feat["cancel_rate"] = (final.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
                       .reindex(feat.index))
feat = feat.reset_index().merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
feat["tenure_days"] = (CUTOFF - feat["signup_date"]).dt.days
feat["lapsed"] = (~feat["customer_id"].isin(active_sep)).astype(int)
feat = feat.drop(columns="signup_date")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery_min", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X_all, y = feat[NUM + CAT], feat["lapsed"]
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=SEED)
base_pred = np.zeros(len(y_te), dtype=int)
print(f"\n=== Lapse problem ===\ncohort {len(feat)} | lapsed {int(y.sum())} ({y.mean()*100:.2f}%) | train {len(X_tr)} | test {len(X_te)}")
print(f"baseline (always not lapsed): accuracy {accuracy_score(y_te, base_pred)*100:.1f}% | "
      f"lapsed F1 {f1_score(y_te, base_pred, zero_division=0):.2f}")

# ======================================================================== task 5: preprocessing without leakage
def make_pre():
    return ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])

print("\n=== Preprocessing ===")
print("missing values in feature columns:", int(X_all.isna().sum().sum()))
assert X_all.isna().sum().sum() == 0
chk = make_pre().fit(X_tr)
fi = NUM.index("frequency")
print(f"scaler mean of frequency (train rows only): {chk.named_transformers_['num'].mean_[fi]:.4f} | "
      f"full-cohort mean: {feat['frequency'].mean():.4f}")

# ======================================================================== task 6: train, tune, evaluate
def metrics(y_true, y_pred):
    return (precision_score(y_true, y_pred, zero_division=0), recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))

lr_pipe = Pipeline([("prep", make_pre()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_tr, y_tr)
rf_pipe = Pipeline([("prep", make_pre()), ("clf", RandomForestClassifier(n_estimators=100, random_state=SEED))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(X_tr, y_tr)
print("\n=== Tuning ===")
print(f"LR  best {lr_gs.best_params_} CV F1 {lr_gs.best_score_:.4f}")
print(f"RF  best {rf_gs.best_params_} CV F1 {rf_gs.best_score_:.4f}")

print("\n=== Test-set evaluation (lapsed = positive class) ===")
results = {}
for name, gs in [("Logistic regression", lr_gs), ("Random forest", rf_gs)]:
    pred = gs.predict(X_te)
    p, r, f = metrics(y_te, pred)
    results[name] = (p, r, f)
    print(f"{name}: precision {p:.3f} recall {r:.3f} F1 {f:.3f}\n  confusion [[TN FP],[FN TP]] =\n{confusion_matrix(y_te, pred)}")

SELECTED_NAME = max(results, key=lambda k: results[k][2])
selected = lr_gs if SELECTED_NAME == "Logistic regression" else rf_gs
print(f"selected model: {SELECTED_NAME} (test F1 {results[SELECTED_NAME][2]:.3f}) -> must be >= 0.55")
assert results[SELECTED_NAME][2] >= 0.55

# ======================================================================== task 7: class-balance experiment
rng = np.random.RandomState(SEED)
lapsed_idx = np.where(y_tr.values == 1)[0]
n_add = int((y_tr == 0).sum() - (y_tr == 1).sum())
extra = rng.choice(lapsed_idx, size=n_add, replace=True)
X_bal = pd.concat([X_tr, X_tr.iloc[extra]])
y_bal = pd.concat([y_tr, y_tr.iloc[extra]])
print(f"\n=== Class balance === training counts before {y_tr.value_counts().to_dict()} after {y_bal.value_counts().to_dict()}")
lr_bal = Pipeline([("prep", make_pre()), ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))]).fit(X_bal, y_bal)
p0, r0, f0 = results["Logistic regression"]
p1, r1, f1 = metrics(y_te, lr_bal.predict(X_te))
print(f"LR un-balanced: precision {p0:.3f} recall {r0:.3f} F1 {f0:.3f}")
print(f"LR over-sampled: precision {p1:.3f} recall {r1:.3f} F1 {f1:.3f}")

# ======================================================================== task 8: interpret, persist, use
print("\n=== Interpretation ===")
rf_best = rf_gs.best_estimator_
names_out = rf_best.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names_out).sort_values(ascending=False)
print("top-5 RF importances:\n", imp.head(5).round(4).to_string())
lr_best = lr_gs.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0], index=lr_best.named_steps["prep"].get_feature_names_out())
print("LR coefficients (standardised):\n", coef[["num__recency", "num__frequency", "num__avg_discount", "num__cancel_rate"]].round(3).to_string())

joblib.dump(selected.best_estimator_, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(reloaded.predict(X_te), selected.best_estimator_.predict(X_te))
print("reloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
te = feat.loc[X_te.index, ["customer_id"]].copy()
te["lapse_probability"] = selected.best_estimator_.predict_proba(X_te)[:, 1]
te["actually_lapsed"] = y_te.values
ret = (te[te["customer_id"].isin(premium_ids)].sort_values("lapse_probability", ascending=False)
       .head(10).reset_index(drop=True))
ret["lapse_probability"] = ret["lapse_probability"].round(4)
ret.to_csv("outputs/retention_list.csv", index=False)
print(f"\nretention list rows: {len(ret)} | actually lapsed: {int(ret['actually_lapsed'].sum())}")
print(f"premium customers in test set: {int(te['customer_id'].isin(premium_ids).sum())}")
print(ret.to_string(index=False))
