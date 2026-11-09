"""Part D (tasks 5-8): tools, ReAct loop, safety limits, checkpointing and resume."""
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.getcwd())
import mock_llm
from rag import retrieve

DB = "data/instamart.db"
LEDGER = "outputs/refund_ledger.jsonl"
RUNS = "outputs/agent_runs.jsonl"
DEMOS = "outputs/agent_demos.jsonl"
STATE_DIR = "state"
AGENT_SYSTEM = ("You are a refund AGENT for a grocery delivery company. Use the available tools to look up the order, "
                "check the policy, compute the refund and then issue or escalate it. "
                "Finish with one line starting REFUND, ESCALATE, DENIED or NOT_FOUND.")
_CURRENT_RUN = {"id": None}          # so ledger lines can be traced back to the run that wrote them


# ------------------------------------------------------------------ task 5: tools
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = None if row[3] is None or (isinstance(row[3], float) and row[3] != row[3]) else row[3]
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": dm}


def search_policy(query):
    return {"results": [{"source": h["source"], "score": round(h["score"], 4), "text": h["text"]}
                        for h in retrieve(query, k=3)]}


def calculate_refund(order_amount, delivery_minutes):
    """Delivered in more than 45 minutes -> 100% of the order amount; otherwise 0."""
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0}


def _ledger(tool, order_id, refund_inr):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "run_id": _CURRENT_RUN["id"], "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order by its id in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the three best matching chunks.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                      "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ------------------------------------------------------------------ task 8: checkpoints
class SimulatedCrash(Exception):
    pass


def _checkpoint_path(run_id):
    return os.path.join(STATE_DIR, f"{run_id}.json")


def _save_checkpoint(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(_checkpoint_path(st["run_id"]), "w", encoding="utf-8") as f:
        json.dump({k: st[k] for k in ("run_id", "scenario_id", "messages", "tokens_used", "step", "tools_executed",
                                      "llm_calls", "steps", "max_iterations", "token_limit", "time_limit_seconds")},
                  f, indent=1)


# ------------------------------------------------------------------ tasks 6-7: the ReAct loop
def _fallback(st, status):
    done = [s["tool"] for s in st["steps"]]
    summary = (f"HANDOFF to human agent: run stopped with {status} for scenario {st['scenario_id']} after "
               f"{st['llm_calls']} model calls and {len(done)} tool steps ({', '.join(done) or 'none'}). "
               f"Please review the request manually.")
    return _result(st, status, summary)


def _result(st, status, final_answer):
    refund = 0
    if status in ("REFUND", "ESCALATE") and st["steps"]:
        refund = st["steps"][-1]["observation"].get("refund_inr", 0)
    return {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": status, "final_answer": final_answer,
            "steps": st["steps"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"], "refund_inr": refund,
            "tools_executed": list(st["tools_executed"])}


def _loop(st, crash_after_tools=None):
    start = time.monotonic()
    _CURRENT_RUN["id"] = st["run_id"]
    while True:
        if time.monotonic() - start >= st["time_limit_seconds"]:              # (c) wall-clock limit
            return _fallback(st, "FALLBACK_TIMEOUT")
        if st["llm_calls"] >= st["max_iterations"]:                            # (a) iteration limit
            return _fallback(st, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                              # (b) token limit, tool NOT executed
            return _fallback(st, "FALLBACK_TOKEN_LIMIT")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            return _result(st, text.split(":")[0].strip(), text)
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        observation = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(observation)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": observation})
        st["tools_executed"].append(tc.name)
        st["step"] += 1
        _save_checkpoint(st)                                                   # after the observation is appended
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}_run", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "step": 0, "tools_executed": [], "llm_calls": 0, "steps": [],
          "max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    return _loop(st, crash_after_tools)


def resume_agent(run_id):
    with open(_checkpoint_path(run_id), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st)


def _append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _record(res):
    return {k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}


if __name__ == "__main__":
    for p in (LEDGER, RUNS, DEMOS):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    sc = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)
    print("=== Seven scenarios (default limits) ===")
    print("id  status          refund_inr  expected  model_calls  tokens")
    for r in sc.itertuples(index=False):
        res = run_agent(r.message, r.scenario_id)
        _append(RUNS, _record(res))
        exp = float(r.expected_refund_inr)
        assert abs(res["refund_inr"] - exp) <= 0.01, (r.scenario_id, res["refund_inr"], exp)
        want = r.expected_outcome if r.expected_outcome != "FALLBACK" else "FALLBACK_MAX_ITERATIONS"
        assert res["status"] == want, (r.scenario_id, res["status"], want)
        print(f"{r.scenario_id:<3} {res['status']:<23} {res['refund_inr']:>8.2f} {exp:>9.2f} "
              f"{res['llm_calls']:>8} {res['tokens_used']:>9}")

    s1 = sc.iloc[0]
    print("\n=== Stop-safely demos ===")
    d1 = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_demo")
    print(f"S1 token_limit=200       -> {d1['status']} | model calls {d1['llm_calls']} | tokens {d1['tokens_used']} "
          f"| tools executed {d1['tools_executed']}")
    print("   handoff:", d1["final_answer"])
    d2 = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
    print(f"S1 time_limit_seconds=0  -> {d2['status']} | model calls {d2['llm_calls']} | tokens {d2['tokens_used']}")
    print("   handoff:", d2["final_answer"])
    _append(DEMOS, {"demo": "token_limit=200", **_record(d1)})
    _append(DEMOS, {"demo": "time_limit_seconds=0", **_record(d2)})

    print("\n=== Crash and resume (S1) ===")
    try:
        run_agent(s1.message, "S1", run_id="S1_crash_demo", crash_after_tools=2)
    except SimulatedCrash as e:
        print("crashed:", e)
    ck = json.load(open(_checkpoint_path("S1_crash_demo")))
    print(f"checkpoint: step {ck['step']}, tools_executed {ck['tools_executed']}, tokens_used {ck['tokens_used']}")
    res = resume_agent("S1_crash_demo")
    print(f"resumed -> {res['status']} {res['refund_inr']:.2f} | model calls (this part) {res['llm_calls']}")
    print("tools executed across crash + resume:", res["tools_executed"])
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) <= 0.01
    _append(DEMOS, {"demo": "crash_and_resume", **_record(res), "tools_executed": res["tools_executed"]})
    print("ledger lines:", sum(1 for _ in open(LEDGER)))
