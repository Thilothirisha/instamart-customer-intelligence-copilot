"""Part A: run the seven SQL queries from sql/queries.sql and cross-check/explore with pandas."""
import re
import sqlite3

import pandas as pd

DB = "data/instamart.db"
sql_text = open("sql/queries.sql", encoding="utf-8").read()
queries = {}
parts = re.split(r"-- name: (Q\d)\n", sql_text)
for name, body in zip(parts[1::2], parts[2::2]):
    queries[name] = body.strip()

con = sqlite3.connect(DB)
res = {k: pd.read_sql_query(q, con) for k, q in queries.items()}

q1, q2, q3, q4, q5, q6, q7 = (res[f"Q{i}"] for i in range(1, 8))
print("Q1 customers:", len(q1))
print("Q2 rows:", len(q2), "| total:", round(q2["revenue"].sum(), 2), "| largest cell:", q2.iloc[0].to_dict())
print("Q3 products above average price:", len(q3))
print("Q4 customers without completed order:", len(q4))
print("Q5 customers above average:", len(q5), "| average spend:", q5["avg_total"].iloc[0])
top = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]]
print("Q6 top customer per city:\n", top.to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER:", int((q6["rnk"] != q6["row_num"]).sum()))
print("Q7 daily rows:", len(q7), "| last cumulative:", q7["cumulative_revenue"].iloc[-1])

# ---------------------------------------------------------------- pandas cross-check
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
comp = orders[orders["status"] == "completed"]
m = comp.merge(customers[["customer_id", "city"]], on="customer_id").merge(
    products[["product_id", "category"]], on="product_id")
pv = pd.pivot_table(m, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("\nPivot grand total:", round(pv.loc["All", "All"], 2), "| equals SQL total:",
      abs(pv.loc["All", "All"] - q2["revenue"].sum()) < 0.01)
city_stats = (m.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                    n_orders=("order_amount", "count")).round(2))
print(city_stats.to_string())

# ---------------------------------------------------------------- exploratory statistics
a = comp["order_amount"]
q1_, q3_ = a.quantile(.25), a.quantile(.75)
iqr = q3_ - q1_
lo, hi = q1_ - 1.5 * iqr, q3_ + 1.5 * iqr
print(f"\nmean {a.mean():.2f} | median {a.median():.2f} | skew {a.skew():.2f}")
print(f"Q1 {q1_:.2f} | Q3 {q3_:.2f} | IQR {iqr:.2f} | fences [{lo:.2f}, {hi:.2f}]")
print(f"outliers below {int((a < lo).sum())} | above {int((a > hi).sum())}")
corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
pairs = corr.where(~pd.DataFrame(
    [[i == j for j in corr.columns] for i in corr.index], index=corr.index, columns=corr.columns)).abs().stack()
print("largest |corr| pair:", pairs.idxmax(), round(corr.loc[pairs.idxmax()], 3))
