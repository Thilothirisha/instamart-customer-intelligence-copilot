"""Part A: audit raw exports, clean them with explicit rules, store CSV + SQLite."""
import os
import sqlite3

import numpy as np
import pandas as pd

RAW, CLEAN, DB = "data/raw", "data/clean", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)

customers = pd.read_csv(f"{RAW}/customers.csv")
products = pd.read_csv(f"{RAW}/products.csv")
# keep the literal string "N/A" (pandas would silently treat it as missing); only truly empty cells are NaN
orders = pd.read_csv(f"{RAW}/orders.csv", keep_default_na=False, na_values=[""])


# ------------------------------------------------------------------ audit
def audit_table(name, df):
    print(f"\n=== {name}: shape {df.shape} ===")
    rep = pd.DataFrame({"dtype": df.dtypes.astype(str),
                        "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)})
    print(rep.to_string())


for n, d in [("customers", customers), ("products", products), ("orders", orders)]:
    audit_table(n, d)

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
raw_gender = customers["gender"].dropna()
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders["status"] == "pending").sum()),
    "distinct order_id": int(orders["order_id"].nunique()),
    "non-numeric order_amount": int(amount_num.isna().sum()),
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((customers["age"] < 10) | (customers["age"] > 100)).sum()),
    "distinct raw city spellings": int(customers["city"].nunique()),
    "distinct non-blank raw gender spellings": int(raw_gender.nunique()),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== AUDIT FIGURES ===")
for k, v in audit.items():
    print(f"{k}: {v}")

# ------------------------------------------------------------------ clean customers
c = customers.copy()
c["city"] = c["city"].str.strip().str.title()                                   # (a)
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")  # (b)
n_bad_age = int(((c["age"] < 10) | (c["age"] > 100)).sum())
c.loc[(c["age"] < 10) | (c["age"] > 100), "age"] = np.nan                       # (c)
n_age_filled = int(c["age"].isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
c["signup_date"] = pd.to_datetime(c["signup_date"])                              # (d)
print("\n=== CLEAN CUSTOMERS ===")
print(c["city"].value_counts().to_string())
print(c["gender"].value_counts().to_string())
print(f"impossible ages set to missing: {n_bad_age}; total ages filled with city median: {n_age_filled}")
print(f"missing ages: {int(c['age'].isna().sum())}; age range {c['age'].min()}-{c['age'].max()}")

# ------------------------------------------------------------------ clean orders
o = orders.copy()
o["order_ts"] = pd.to_datetime(o["order_ts"])                                    # 1
n0 = len(o)
o = o.drop_duplicates()                                                          # 2
print(f"\n=== CLEAN ORDERS ===\nrule 2: removed {n0 - len(o)} exact duplicate rows")
n1 = len(o)
o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")  # 3
print(f"rule 3: removed {n1 - len(o)} superseded (earlier pending) records; rows now {len(o)}")
o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")  # 4
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
miss_amt = o["order_amount"].isna()
o.loc[miss_amt, "order_amount"] = (o["price_inr"] * o["quantity"] * (1 - o["discount_pct"] / 100)).round(2)[miss_amt]
print(f"rule 4: unparseable amounts rebuilt from price: {int(miss_amt.sum())}")
neg = o["delivery_minutes"] < 0                                                  # 5
o.loc[neg, "delivery_minutes"] = np.nan
comp = o["status"] == "completed"
med = o.loc[comp].groupby("category")["delivery_minutes"].transform("median")
to_fill = comp & o["delivery_minutes"].isna()
o.loc[to_fill, "delivery_minutes"] = med.reindex(o.index)[to_fill]
print(f"rule 5: negative delivery times set missing: {int(neg.sum())}; "
      f"completed orders filled with category median: {int(to_fill.sum())}; "
      f"cancelled orders left missing: {int(((o['status'] == 'cancelled') & o['delivery_minutes'].isna()).sum())}")
o = o.drop(columns=["category", "price_inr"]).sort_values("order_ts").reset_index(drop=True)

print(f"final rows {len(o)}; status counts {o['status'].value_counts().to_dict()}")
print(f"missing amount {int(o['order_amount'].isna().sum())}; "
      f"missing delivery (completed) {int(o.loc[o.status == 'completed', 'delivery_minutes'].isna().sum())}; "
      f"missing delivery (cancelled) {int(o.loc[o.status == 'cancelled', 'delivery_minutes'].isna().sum())}")
print(f"completed revenue INR {o.loc[o.status == 'completed', 'order_amount'].sum():,.2f}")

# ------------------------------------------------------------------ store
c_out = c.copy(); c_out["signup_date"] = c_out["signup_date"].dt.strftime("%Y-%m-%d")
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
c_out.to_csv(f"{CLEAN}/customers.csv", index=False)
products.to_csv(f"{CLEAN}/products.csv", index=False)
o_out.to_csv(f"{CLEAN}/orders.csv", index=False)
if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    c_out.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    o_out.to_sql("orders", con, index=False)
print("\nwrote data/clean/*.csv and data/instamart.db")
