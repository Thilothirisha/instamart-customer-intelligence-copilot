"""Part C: support-ticket triage with guardrails, redaction, retries, validation, budget and routing."""
import json
import os
import random
import re
import sys
import time

import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from mock_llm import TransientLLMError, est_tokens

os.makedirs("outputs", exist_ok=True)
CATS = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

# ------------------------------------------------------------------ redaction (order matters)
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text, counter=None):
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        if counter is not None:
            counter[name] = counter.get(name, 0) + n
    return text


def contains_pii(text):
    return any(pat.search(text) for _, pat, _ in PATTERNS)


# ------------------------------------------------------------------ prompts
JSON_RULE = ('Reply with JSON only, exactly in the form {"category": "...", "reason": "..."} '
             "and nothing else.")
ZERO_SHOT = ("You CLASSIFY customer support tickets for a grocery delivery app. Allowed categories: "
             "billing, delivery, technical, fraud. " + JSON_RULE)
FEW_SHOT = ("You CLASSIFY customer support tickets for a grocery delivery app. Allowed categories: "
            "billing, delivery, technical, fraud. " + JSON_RULE + "\n"
            "EXAMPLES:\n"
            'Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment.\n'
            'Answer: {"category": "billing", "reason": "duplicate charge"}\n'
            'Ticket: There is an unauthorized transaction on my card. I did not place it.\n'
            'Answer: {"category": "fraud", "reason": "unauthorized transaction"}\n'
            'Ticket: The app crashes every time I open the cart.\n'
            'Answer: {"category": "technical", "reason": "app crash"}\n'
            'Ticket: My order has not arrived and the rider is not picking up calls.\n'
            'Answer: {"category": "delivery", "reason": "order not arrived"}')
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "EXAMPLES:" in FEW_SHOT


# ------------------------------------------------------------------ resilient call
class RunStats:
    def __init__(self):
        self.transient_errors = 0
        self.recovered = 0
        self.waits = []          # (retry_number, wait)


def call_with_retry(messages, ticket_id, stats, verbose=True):
    """Up to MAX_RETRIES retries; wait before retry n = BASE_DELAY * 2**n * uniform(0.9, 1.15)."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = mock_llm.chat(messages)
            if attempt > 0:
                stats.recovered += 1
            return resp
        except TransientLLMError:
            stats.transient_errors += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(0.9, 1.15)
            stats.waits.append((attempt, wait))
            if verbose:
                print(f"  [retry] ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s")
            time.sleep(wait)


# ------------------------------------------------------------------ output validation
def validate_answer(text):
    """Return (status, category). Never guesses."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("category") in CATS:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", ""          # valid JSON, category outside the list
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = [c for c in CATS if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""


# ------------------------------------------------------------------ pipeline
def run_pipeline(system_prompt, token_budget=None, verbose=True, seed=42):
    mock_llm.reset()                      # fresh counters for each full run
    random.seed(seed)
    stats, rows, total_tokens = RunStats(), [], 0
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "",
               "tokens_used": 0, "true_category": t.true_category}
        text = t.text
        if not text.strip():                                      # guardrail (a)
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:                 # guardrail (b)
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": redact(text)}]
            assert not contains_pii(messages[1]["content"]) and not contains_pii(messages[0]["content"])
            resp = call_with_retry(messages, t.ticket_id, stats, verbose)
            used = resp.usage.total_tokens
            total_tokens += used
            row["tokens_used"] = used
            row["status"], row["category"] = validate_answer(resp.choices[0].message.content)
        rows.append(row)
    return pd.DataFrame(rows), stats, total_tokens


if __name__ == "__main__":
    # ---------------- guardrails and redaction report
    print("=== Guardrails ===")
    empty = int((tickets["text"].str.strip() == "").sum())
    long_ = int(tickets["text"].map(lambda s: est_tokens(s) > MAX_INPUT_TOKENS).sum())
    long_tok = sorted({est_tokens(s) for s in tickets["text"] if est_tokens(s) > MAX_INPUT_TOKENS})
    print(f"empty tickets: {empty} | long tickets: {long_} (est tokens {long_tok})")

    print("\n=== Redaction ===")
    counter = {}
    redacted = [redact(s, counter) for s in tickets["text"]]
    print("substitutions:", counter, "| total:", sum(counter.values()))
    print("PII left after redaction:", sum(contains_pii(s) for s in redacted))
    assert sum(contains_pii(s) for s in redacted) == 0

    # ---------------- full runs, both prompts
    summary = {}
    for label, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        print(f"\n=== Full run: {label} (retry log) ===")
        df, stats, total = run_pipeline(prompt)
        cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
        acc = (cls["category"] == cls["true_category"]).mean()
        summary[label] = dict(df=df, stats=stats, total=total, cls=cls, acc=acc)
        print(f"-- {label}: status counts {df['status'].value_counts().to_dict()}")
        print(f"   transient errors {stats.transient_errors}, recovered {stats.recovered}")
        ws = [w for _, w in stats.waits]
        print(f"   retry numbers used {sorted({n for n, _ in stats.waits})}, wait range {min(ws):.4f}-{max(ws):.4f}s")
        assert all(0.05 * 0.9 <= w <= 0.05 * 1.15 for w in ws)

    print("\n=== Zero-shot vs few-shot ===")
    print("run        classified  correct  accuracy  total_tokens  tokens/classified")
    for label, s in summary.items():
        n = len(s["cls"]); ok = int((s["cls"]["category"] == s["cls"]["true_category"]).sum())
        print(f"{label:<10} {n:>10} {ok:>8} {s['acc']*100:>8.1f}% {s['total']:>12} {s['total']/n:>17.2f}")
    print("\nZero-shot confusion table (rows = true, columns = predicted):")
    print(pd.crosstab(summary["zero-shot"]["cls"]["true_category"], summary["zero-shot"]["cls"]["category"]).to_string())

    # ---------------- budget demo (zero-shot)
    print("\n=== Budget demo: zero-shot, TOKEN_BUDGET=6000 ===")
    demo, _, demo_total = run_pipeline(ZERO_SHOT, token_budget=6000, verbose=False)
    demo.drop(columns="true_category").to_csv("outputs/triage_budget_demo.csv", index=False)
    print("status counts:", demo["status"].value_counts().to_dict(), "| tokens spent", demo_total)

    # ---------------- final routed run (few-shot, budget 100000)
    print("\n=== Final run: few-shot, TOKEN_BUDGET=100000 ===")
    fin, _, fin_total = run_pipeline(FEW_SHOT, token_budget=100000, verbose=False)
    seg = pd.read_csv("outputs/customer_segments.csv")
    seg_name = dict(zip(seg["customer_id"], seg["segment_name"]))
    seg_prem = dict(zip(seg["customer_id"], seg["is_premium"]))
    HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

    def route(r):
        if r["status"] in HUMAN:
            return "human_review", "NORMAL"
        if r["category"] == "fraud":
            return "security_team", "HIGH"
        if bool(seg_prem.get(r["customer_id"], False)):
            return "premium_desk", "HIGH"
        return "support_queue", "NORMAL"

    fin[["route", "priority"]] = [route(r) for _, r in fin.iterrows()]
    fin["segment_name"] = fin["customer_id"].map(seg_name).fillna("")
    out = fin[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]
    out.to_csv("outputs/triage_results.csv", index=False)
    print("rows:", len(out), "| status:", out["status"].value_counts().to_dict())
    print("routes:", out["route"].value_counts().to_dict())
    print("priorities:", out["priority"].value_counts().to_dict())
    print("tickets with a customer missing from segment file:", int((out["segment_name"] == "").sum()))
