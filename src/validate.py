"""Assert the Part A reference checkpoints against the clean CSVs and the SQLite database."""
import sqlite3

import pandas as pd

o = pd.read_csv("data/clean/orders.csv")
c = pd.read_csv("data/clean/customers.csv")
con = sqlite3.connect("data/instamart.db")

checks = [
    ("12,309 order rows", len(o) == 12309),
    ("order_id unique", o["order_id"].is_unique),
    ("11,110 completed", (o["status"] == "completed").sum() == 11110),
    ("1,199 cancelled", (o["status"] == "cancelled").sum() == 1199),
    ("no pending rows", (o["status"] == "pending").sum() == 0),
    ("0 missing order_amount", o["order_amount"].isna().sum() == 0),
    ("0 missing delivery (completed)", o.loc[o.status == "completed", "delivery_minutes"].isna().sum() == 0),
    ("1,199 missing delivery (cancelled)", o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum() == 1199),
    ("completed revenue = 2,998,668.80", round(o.loc[o.status == "completed", "order_amount"].sum(), 2) == 2998668.80),
    ("exactly 6 cities", c["city"].nunique() == 6),
    ("city counts", c["city"].value_counts().to_dict() ==
     {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("gender counts", c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}),
    ("0 missing ages, all within 18-60", c["age"].isna().sum() == 0 and c["age"].between(18, 60).all()),
    ("db has 3 tables", {r[0] for r in con.execute("select name from sqlite_master where type='table'")} ==
     {"customers", "products", "orders"}),
    ("db orders rows = 12,309", con.execute("select count(*) from orders").fetchone()[0] == 12309),
]
failed = [name for name, ok in checks if not bool(ok)]
for name, ok in checks:
    print(("PASS " if ok else "FAIL ") + name)
assert not failed, f"failed checkpoints: {failed}"
print("ALL CHECKPOINTS PASSED")
