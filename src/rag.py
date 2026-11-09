"""Part D (tasks 1-4): chunking, TF-IDF retrieval, grounded answering and evaluation.
The agent (src/agent.py) imports retrieve() from here, so retrieval is built once."""
import glob
import os
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.getcwd())
import mock_llm

POLICY_DIR = "data/policies"
CHUNK_WORDS = 40


# ------------------------------------------------------------------ task 1: chunking
def chunk_words(text, size=CHUNK_WORDS, overlap=10):
    """Chunks of `size` words; each next chunk starts size-overlap words later; stop once a chunk reaches the end."""
    words, step, chunks, start = text.split(), size - overlap, [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks


def load_chunks(overlap=10):
    out = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        src = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            for c in chunk_words(f.read(), overlap=overlap):
                out.append({"source": src, "text": c})
    return out


# ------------------------------------------------------------------ task 2: vectorise + retrieve
class Index:
    def __init__(self, overlap=10):
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]}
                for i in order]


_DEFAULT = Index(overlap=10)


def retrieve(query, k=3, index=None):
    return (index or _DEFAULT).retrieve(query, k)


# ------------------------------------------------------------------ task 3: augment + answer
RAG_SYSTEM = ("You are a POLICY ASSISTANT for a grocery delivery company. Answer only from the supplied CONTEXT. "
              "If the answer is not in the context, say that you could not find the information in the policy documents.")


def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": RAG_SYSTEM}, {"role": "user", "content": user}])
    return {"answer": resp.choices[0].message.content, "sources": [h["source"] for h in hits]}


# ------------------------------------------------------------------ task 4: evaluation
def evaluate(overlap, verbose=True):
    idx = Index(overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fb_ok = 0
    n_ans = n_unans = 0
    failures = []
    for q in ev.itertuples(index=False):
        res = answer(q.question, idx)
        if q.expected_doc == "NONE":
            n_unans += 1
            fb_ok += res["answer"].startswith("I could not find")
        else:
            n_ans += 1
            d = q.expected_doc in res["sources"]
            f = q.expected_fact in res["answer"]
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((q.question, d, f, res["answer"]))
    if verbose:
        print(f"chunks (overlap={overlap}): {len(idx.chunks)} -> "
              f"{pd.Series([c['source'] for c in idx.chunks]).value_counts().sort_index().to_dict()}")
        print(f"answerable: expected_doc in top-3 {doc_ok}/{n_ans} | expected_fact in answer {fact_ok}/{n_ans} | "
              f"unanswerable fallback {fb_ok}/{n_unans}")
        for q, d, f, a in failures:
            print(f"  FAIL: {q!r} doc_ok={d} fact_ok={f}\n        answer: {a}")
    return doc_ok, fact_ok, fb_ok, failures


if __name__ == "__main__":
    print("=== Chunking and retrieval ===")
    print("chunk counts (overlap 10):", len(_DEFAULT.chunks))
    top = retrieve("refund policy late delivery")
    print("top-3 sources for 'refund policy late delivery':", [(h["source"], round(h["score"], 3)) for h in top])
    print("\n=== Evaluation, overlap 10 ===")
    evaluate(10)
    print("\n=== Evaluation, overlap 0 ===")
    evaluate(0)
