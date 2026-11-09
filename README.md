# Instamart Customer Intelligence & Support Copilot

## Overview

One repository, four connected Parts, all on synthetic data (all money is in INR, nothing is real Instamart data).

| Part | What it does | Main files |
|---|---|---|
| A | Audits and cleans the raw exports, stores them in SQLite, answers seven SQL questions, cross-checks in pandas | `src/part_a_clean.py`, `src/validate.py`, `sql/queries.sql`, `src/part_a_sql.py` |
| B | RFM + K-means customer segments, PCA, lapse-prediction model | `src/part_b.py` |
| C | Support-ticket triage: guardrails, redaction, retries, output validation, token budget, routing | `src/part_c.py` |
| D | TF-IDF policy RAG assistant and a tool-calling refund agent (ReAct, safe stops, checkpoint/resume) | `src/rag.py`, `src/agent.py` |

`generate_data.py` and `mock_llm.py` are the supplied files, committed unmodified. Every language-model call goes through `mock_llm.py` (offline, no key, no network). Libraries: standard library, pandas, numpy, scikit-learn, joblib. Seeds are fixed (`random_state=42`, `random.seed(42)` in the retry run).

Layout after the Reproduce commands: `data/raw/`, `data/policies/`, `data/clean/`, `data/instamart.db`, `outputs/` (`customer_segments.csv`, `lapse_model.joblib`, `retention_list.csv`, `triage_results.csv`, `triage_budget_demo.csv`, `agent_runs.jsonl`, `refund_ledger.jsonl`, plus `agent_demos.jsonl` for the three Part D demonstrations) and `state/` (agent checkpoints).

## Reproduce

Run from the repository root (Python 3.11+).

1. `python -m venv .venv`
2. `source .venv/bin/activate`   (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`   (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/part_a_clean.py`   (audit, cleaning, CSV + SQLite)
6. `python src/validate.py`   (prints `ALL CHECKPOINTS PASSED`)
7. `python src/part_a_sql.py`   (seven SQL queries, pandas cross-check, statistics)
8. `python src/part_b.py`   (segments, lapse model, retention list)
9. `python src/part_c.py`   (triage; takes about 10 seconds because of the retry back-off sleeps)
10. `python src/rag.py`   (chunking, retrieval, evaluation with overlap 10 and 0)
11. `python src/agent.py`   (seven scenarios, stop-safely demos, crash and resume)

Each script prints the numbers reported below. Part B must run before Part C (routing reads `outputs/customer_segments.csv`), and Part A before everything else. `src/agent.py` clears and rewrites `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `outputs/agent_demos.jsonl` and `state/` on every run, so re-running is safe.

## Part A

### Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). `tickets.csv` is read with `keep_default_na=False` wherever used. For `orders.csv` I also use `keep_default_na=False, na_values=[""]`, because pandas would otherwise silently turn the literal text `N/A` into a missing value and hide part of the unparseable amounts.

Data types: all text columns are strings; `age` and `delivery_minutes` are float64 (they contain blanks); `price_inr`, `quantity`, `discount_pct` are int64.

Missing values (only columns with any; all other columns have 0, 0.00%):

| Table | Column | Missing | % |
|---|---|---|---|
| customers | age | 61 | 5.08 |
| customers | gender | 30 | 2.50 |
| orders | delivery_minutes | 2541 | 19.42 |

Audit figures (all match the reference):

| Figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and what each changed

Customers
- (a) `city` stripped and title-cased: 24 raw spellings became 6 cities.
- (b) `gender` mapped (`m/male/Male/MALE` to male, `f/female/Female/FEMALE` to female, blank to unknown): 10 spellings became 3 values; 30 blanks became `unknown`.
- (c) 6 impossible ages (outside 10-100) set to missing; then all 67 missing ages (61 blank + 6) filled with the city median via `groupby("city").transform("median")`.
- (d) `signup_date` converted to datetime.
- Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.

Orders, applied in this order
1. `order_ts` converted to datetime.
2. Exact duplicate rows removed: 119 rows.
3. For each `order_id` only the record with the latest `order_ts` kept: 656 earlier `pending` records removed, leaving 12,309 rows (one per order).
4. `order_amount` converted to numeric (unparseable becomes missing) and the 112 missing amounts rebuilt as `round(price_inr * quantity * (1 - discount_pct/100), 2)`. (120 unparseable in the raw file; 8 sat on rows already removed by rules 2-3.)
5. 105 negative `delivery_minutes` set to missing (106 in raw; one was on a removed row). For completed orders, 774 missing delivery times (including those) were filled with the median of the product category among completed orders (`groupby` + `transform`). The 1,199 cancelled orders keep a missing delivery time, because a cancelled order was never delivered and imputing a time would invent data.

Final: 12,309 rows; 11,110 completed and 1,199 cancelled; no pending; 0 missing `order_amount`; 0 missing delivery among completed, 1,199 among cancelled; completed revenue INR 2,998,668.80.

### Store and validate

Clean tables are written to `data/clean/*.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate.py` asserts 15 checkpoints (row counts, status counts, missing values, revenue, city and gender counts, age range, database tables) and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE then HAVING | 30 customers |
| Q2 three-table join | 30 rows; revenues add to INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTE | 550 customers above the average total of INR 2,552.06 |
| Q6 window ranking | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties cause the Q6 difference: customers with the same total spend in the same city get the same RANK (and the next rank is skipped), while ROW_NUMBER always gives every row its own consecutive number, so the two disagree for tied rows and for every row after a tie.

### pandas cross-check

- `pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, equal to the SQL total.
- Per city (completed orders), sum / mean / count: Bengaluru 523,643.40 / 276.47 / 1,894; Chennai 463,629.15 / 272.72 / 1,700; Delhi 502,829.80 / 243.62 / 2,064; Hyderabad 528,589.90 / 284.95 / 1,855; Mumbai 479,058.50 / 297.00 / 1,613; Pune 500,918.05 / 252.48 / 1,984.

### Exploratory statistics (completed `order_amount`)

| Statistic | Value |
|---|---|
| mean / median | 269.91 / 135.00 |
| skewness | 4.61 |
| Q1 / Q3 / IQR | 64.80 / 297.00 / 232.20 |
| 1.5 x IQR fences | -283.50 and 645.30 |
| outliers below / above | 0 / 897 |

Correlation matrix (completed orders): order_amount vs discount_pct -0.095; order_amount vs quantity +0.087; order_amount vs delivery_minutes -0.003; discount_pct vs quantity -0.000; discount_pct vs delivery_minutes -0.010; quantity vs delivery_minutes -0.013.

Answers (under 200 words):
1. **What is it about?** Six months (April-September 2025) of quick-commerce orders from 1,175 buying customers across six cities, in five categories, in INR.
2. **Distributions?** Order values are strongly right-skewed (mean 269.91 against median 135.00, skewness 4.61); most baskets are small groceries.
3. **Outliers?** 897 orders above the INR 645.30 fence and none below. They are mostly electronics, i.e. genuine big baskets rather than errors. The raw file's real anomalies (duplicates, pending records, negative times, unparseable amounts) were fixed in cleaning.
4. **Relationships?** All correlations are weak. The largest in absolute value is order_amount with discount_pct (-0.095), just above order_amount with quantity (+0.087). A correlation this small does not establish causation, and with these numbers it hardly even describes a pattern.

## Part B

All behaviour uses completed orders only. No encoder, scaler or resampler saw test rows, no data from 2025-09-01 onwards entered the lapse features, and the segment labels from task 3 were not used as features.

### RFM (snapshot 2025-10-01 00:00:00)

1,175 customers. Recency: median 19, min 0, max 180. Frequency: mean 9.46, max 33. Monetary: mean 2,552.06, max 14,766.55.

### K-means (StandardScaler, `n_init=10`, `random_state=42`)

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The elbow is at K=4: inertia drops by 525 going from K=3 to K=4 but only 126 from K=4 to K=5. K=4 also has the highest silhouette (0.526), and four segments give four distinct campaigns (big spenders, loyal regulars, occasional buyers, lapsing customers).

Why inertia alone cannot pick K: adding clusters can only shrink the sum of squared distances to the nearest centre (K=8 is lower than K=2, and K equal to the number of customers would give 0), so inertia always favours more clusters regardless of whether the extra clusters mean anything. The choice needs the elbow, the silhouette and what the business can act on.

### Segment profile (K = 4)

Cluster numbers are only labels with no ordering; names come from the profile.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | Premium Big Spenders (**is_premium**, highest mean monetary) | 138 | 24.12 | 6.77 | 7,129.70 | Early access to electronics, free delivery, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Subscription and bundle offers, loyalty points; protect delivery reliability |
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Category cross-sell and small nudges to raise order frequency |
| 3 | Lapsing / At-Risk | 182 | 109.04 | 2.20 | 685.99 | Cheap win-back message with a time-limited coupon |

PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. `outputs/customer_segments.csv` has columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`.

### Lapse problem

Cutoff 2025-09-01. Cohort = customers with a completed order before the cutoff: **1,156**; lapsed (no completed order in September) **443 (38.32%)**; split 80/20 stratified, `random_state=42`: **924 train / 232 test**. Features, from orders before the cutoff only: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days.

Baseline (always predict not lapsed): accuracy 61.6%, lapsed-class F1 0.00.

### Preprocessing without leakage

A `ColumnTransformer` inside a scikit-learn `Pipeline`: `StandardScaler` for the 8 numeric features and `OneHotEncoder(drop="first")` for city, gender and favourite category (k-1 columns each), fitted on training rows only (inside `GridSearchCV` it is re-fitted on each training fold). Missing values in the feature columns: 0 (asserted in code). Learned scaler mean of frequency (train rows only) = 8.1147 versus full-cohort mean 8.0285: they differ because the scaler only ever sees the 924 training rows, never the test rows, so it learns the training subset's mean.

### Tuning and test results (lapsed = positive class)

| Model | Best parameters | Best CV F1 |
|---|---|---|
| Logistic regression (`C` in 0.01, 0.1, 1, 10, 100) | C = 100 | 0.6506 |
| Random forest (depth 3/5/8 x leaf 5/10/20, 45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.6811 |

| Test set | Confusion [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** |

**Selected: random forest** (test F1 0.655, above the 0.55 requirement). Both models find the same 58 of 89 lapsers; the forest wastes two fewer offers and has the better cross-validated F1 (0.681 versus 0.651). Missing a customer who is about to lapse costs all their future orders, whereas wrongly targeting a loyal customer costs one coupon, so recall matters more than precision here; with recall equal, the forest is the better choice and the slightly higher precision comes free. If the team can afford to over-contact, the balanced model below is an option.

### Class-balance experiment (training set only)

Lapsed training rows were duplicated at random (`random_state=42`, with replacement) from 354 to 570, matching the 570 non-lapsed rows; the best logistic regression was refitted and evaluated on the unchanged test set.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| un-balanced | 0.644 | 0.652 | 0.648 |
| over-sampled | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision did not rise (it fell by 0.05). For the retention team this means the balanced model reaches many more would-be lapsers but also contacts more loyal customers, so it suits a cheap campaign, not an expensive one.

### Interpretation

Top-5 random-forest importances: frequency 0.280, monetary 0.271, recency 0.115, average delivery minutes 0.061, cancellation rate 0.052. Business reading: how often and how much a customer buys is by far the best predictor; recency matters less once frequency is known; slow delivery and cancellations add a little signal.

Logistic-regression coefficient signs (standardised features): frequency **-1.70** (more orders, less likely to lapse); recency **+0.04** (a longer gap slightly raises lapse risk, but the effect is weak once frequency is in the model); cancellation rate **-0.12** (small and counter-intuitive, probably noise, so not a basis for action).

### Persist and use

The selected model (full pipeline) is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original test predictions exactly (checked with `np.array_equal`). `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`) holds the 10 highest-probability customers among the 40 Premium customers in the test set; **1 of the 10 truly lapsed**. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes the September window defining the lapse label.

## Part C

Input guardrails: 4 empty tickets get `skipped_empty`, 4 long tickets (734 estimated tokens each, limit `MAX_INPUT_TOKENS = 400`) get `flagged_long`; none is sent to the model.

### Redaction (email, then phone, then order id; user message = redacted text only)

| Type | Substitutions |
|---|---|
| email `[EMAIL]` | 55 |
| phone `[PHONE]` | 91 |
| order id `[ORDER_ID]` | 132 |
| total | 278 |

Re-applying the three patterns to the redacted texts finds 0 matches, and every prompt sent to the model is asserted PII-free in code.

### Prompts

Both contain `CLASSIFY`, the four categories and a JSON-only instruction `{"category": "...", "reason": "..."}`. The zero-shot prompt has no `EXAMPLES:`; the few-shot prompt has an `EXAMPLES:` line followed by four labelled examples (billing, fraud, technical, delivery).

### Retries

`call_with_retry`: at most 3 retries per ticket, wait before retry n = `0.05 * 2**n * random.uniform(0.9, 1.15)`, one log line per retry (`[retry] ticket=T0002 retry_number=0 wait=0.0530s`). `mock_llm.reset()` runs before every full run. Each full run over the 156 eligible tickets had **19 TransientLLMErrors, all 19 recovered**; all retries were retry number 0, with waits between 0.0453 s and 0.0562 s (inside 0.045-0.0575); no ticket ended with a permanent error.

### Validation of answers

Valid JSON with an allowed category gives `classified_json`; non-JSON text is searched with `re.findall(r"[a-z]+", text.lower())` and accepted only if exactly one allowed category word appears (`classified_keyword`); anything else, including JSON with category `shipping`, is `flagged_invalid_output` and goes to human review. Result of one full run (identical for both prompts): **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output** (plus 4 skipped_empty and 4 flagged_long).

### Zero-shot vs few-shot (no budget)

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 10,804 | 73.50 |
| few-shot | 147 | 147 | 100.0% | 26,248 | 178.56 |

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

When the few-shot tokens are worth paying: few-shot costs 105 more tokens per classified ticket (178.56 versus 73.50, about 2.4 times) and 15,444 more tokens in total, which buys 21 extra correct tickets (about 735 tokens each) and lifts accuracy by 14.3 points. Every zero-shot error is a fraud ticket sent to billing (11) or a technical ticket sent to billing (10); a missed fraud ticket loses its HIGH-priority security routing, so here the extra tokens are clearly worth it. For low-stakes categories or very high volume with a human check downstream, the cheaper zero-shot prompt could be enough.

### Budget demo and final routing

- **Budget demo:** zero-shot with `TOKEN_BUDGET = 6000` saved to `outputs/triage_budget_demo.csv`: 76 classified_json, 6 classified_keyword, 5 flagged_invalid_output, 4 skipped_empty, 4 flagged_long and **69 budget_exhausted**; 6,033 tokens were spent (the check happens before each call, so the last call may cross the budget).
- **Final run:** few-shot with `TOKEN_BUDGET = 100000` saved to `outputs/triage_results.csv` (164 rows; columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`).

Routes in the final file: support_queue 102, security_team 34 (priority HIGH, fraud), premium_desk 11 (HIGH), human_review 17 (NORMAL). Priorities: 45 HIGH, 119 NORMAL. All 164 customers have a row in the segment file (none treated as non-premium by default).

**Rule-validated versus human:** the 147 tickets with status `classified_json` or `classified_keyword` were passed by the validator and routed automatically by the rules. The 17 tickets in `human_review` are the 4 empty, 4 long and 9 invalid-output tickets. The model never guesses on those.

## Part D

### Chunking and retrieval

Chunks of 40 words, overlap 10: **15 chunks** (refund 4, delivery 3, escalation 3, payment 3, privacy 2). `TfidfVectorizer()` is fitted on the chunk texts; a query is transformed with the same fitted vectorizer and chunks are ranked by cosine similarity (`retrieve(query, k=3)` returns dictionaries with `source, score, text`). For `refund policy late delivery` all top three chunks come from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word appears in a chunk; IDF (inverse document frequency) down-weights words that appear in many chunks, so rare, distinctive words count more. A limitation: TF-IDF matches words, not meaning, so a question phrased as "money back" will not match a chunk that says "refund".

### Evaluation (`data/raw/policy_eval.csv`, 14 questions)

| Setting | Answerable: expected doc in top 3 | Answerable: expected fact in answer | Unanswerable: fallback "I could not find" |
|---|---|---|---|
| overlap 10 (15 chunks) | 12/12 | 12/12 | 2/2 |
| overlap 0 (13 chunks) | 12/12 | 10/12 | 2/2 |

With overlap 0 two questions break: "What delivery time makes a completed order eligible for a full refund?" (the sentence is cut after "delivered in" so the phrase "more than 45 minutes" lands in the neighbouring chunk) and "How soon is money returned when a payment fails but is deducted?" (the sentence "...returned automatically within 48 hours" is split across a chunk boundary, so the answer lacks "48 hours"). Overlap keeps a copy of the boundary words in both chunks.

### Tools

`TOOLS_SPEC` lists five tools with a name, one-sentence description and JSON-style parameters; `TOOL_MAP` maps names to functions: `get_order` (reads `data/instamart.db`), `search_policy` (returns the retrieval output under `results`), `calculate_refund` (more than 45 minutes gives 100% of the order amount, else 0), `issue_refund` and `escalate_to_human` (each append a JSON line with timestamp, tool name and run id to `outputs/refund_ledger.jsonl`). The loop looks the name up in `TOOL_MAP` and calls the function with the model's arguments.

### Seven scenarios (default limits: 6 iterations, 6000 tokens, 10 seconds)

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,102 |
| S2 | REFUND | 135.00 | 5 | 1,102 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,089 |
| S4 | DENIED | 0.00 | 4 | 769 |
| S5 | DENIED | 0.00 | 3 | 460 |
| S6 | NOT_FOUND | 0.00 | 2 | 154 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,246 |

All refund amounts equal `expected_refund_inr` within 0.01 (asserted in the script). Runs are in `outputs/agent_runs.jsonl` (one line per scenario with every step's tool, arguments and observation).

### Stopping safely

| Demonstration | Status | Detail |
|---|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped after 3 model calls (477 tokens) without executing the third tool; only `get_order` and `search_policy` had run |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (0 calls, 0 tokens) |

S7 finishes all six model calls under the default 6000-token limit (3,246 tokens) and ends with FALLBACK_MAX_ITERATIONS. Every fallback returns the steps done so far and a short hand-off string, for example: `HANDOFF to human agent: run stopped with FALLBACK_TOKEN_LIMIT for scenario S1 after 3 model calls and 2 tool steps (get_order, search_policy). Please review the request manually.`

### Crash and resume (S1)

After each tool observation a checkpoint is written to `state/<run_id>.json` (run id, scenario id, full message list, tokens used, step number, tools executed so far, plus steps and limits). The demo raises a simulated crash right after the second tool's checkpoint (step 2, 184 tokens used), then `resume_agent("S1_crash_demo")` continues from the checkpoint. Tools executed across the crash and the resume, in order, with none repeated:

1. `get_order`
2. `search_policy`  (crash here)
3. `calculate_refund`  (after resume)
4. `issue_refund`

Final status REFUND with INR 128.25. The demonstration lines are saved in `outputs/agent_demos.jsonl`; the ledger has 4 lines (S1, S2 refunds, S3 escalation, and the S1 resume refund, each tagged with its run id).

### ReAct versus plan-and-execute

My ReAct agent makes one model call per step: 5 calls for a refund or escalation (S1-S3, about 1,100 tokens), 4 for an on-time denial (S4), 3 for a cancelled order (S5) and 2 for a missing order (S6). The whole growing message list is re-sent each call, so tokens grow faster than calls. A plan-and-execute design would ask the model once for the full step list (get order, check policy, calculate, issue or escalate) and then run it with no model calls in between, probably 1 or 2 calls and fewer tokens in total.

The difference shows on unexpected observations. ReAct reads each observation and stops at once: on NOT_FOUND it ended after 2 calls, and on a cancelled order after 3. A fixed plan would keep executing steps that no longer make sense (calculate a refund for an order that does not exist) until a check triggers a replan, which costs an extra model call and is the point where mistakes creep in. ReAct also pairs naturally with the per-step checkpoint and the iteration, token and time limits.

For this refund workflow I would choose ReAct: the paths are short (at most 5 calls), money is involved, and checking each observation before the next action is worth the extra calls. Plan-and-execute would pay off for long, predictable workflows with many steps.

## Limitations and Next Steps

- **Retention list is illustrative.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and the retention list is illustrative, not a leakage-free evaluation. (The model's own features use only orders before 2025-09-01.) Only 1 of the 10 listed customers truly lapsed, although 31.9% of all Premium customers in the cohort lapsed, so the model ranks customers within this segment poorly. The leak is visible in the other segments too: every customer in the whole-window Lapsing / At-Risk segment (mean recency 109 days) is lapsed by construction, because that recency counts the September window.
- All data is synthetic and the "LLM" is a deterministic keyword mock, so accuracies (85.7% and 100%) say little about a real model; real prompts would need fresh evaluation on real tickets.
- One train/test split of 232 customers: test F1 differences of 0.01 (forest 0.655 versus logistic regression 0.648) are within noise. Next steps: repeated cross-validation, probability calibration and a time-based split.
- The snapshot-based RFM assumes a six-month window; K-means on three skewed variables could be repeated on log-transformed values.
- TF-IDF retrieval depends on exact words; embeddings or hybrid retrieval would cope with paraphrases. The refund tools do not check the 7-day request window (all simulated requests are inside it), and the checkpoint resume restarts the wall-clock timer.
- Charts were intentionally not created or committed.
