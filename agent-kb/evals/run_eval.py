import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import build_agent
from knowledge_base.vector_store import search

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

REFUSAL_MARKERS = [
    "not in the knowledge base",
    "no relevant information",
    "does not contain",
    "doesn't contain",
    "not specified",
    "cannot find",
    "could not find",
    "unable to find",
    "not mentioned",
    "not available",
]


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def looks_like_refusal(answer: str) -> bool:
    """
    Heuristic: did the agent decline to answer rather than
    inventing something? Used to score unanswerable questions.
    """
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score_keywords(answer: str, must_contain: list[str]) -> bool:
    """
    Cheap deterministic check: does the answer mention at least
    one of the required keywords? Case-insensitive.
    """
    if not must_contain:
        return True

    lowered = answer.lower()
    return any(kw.lower() in lowered for kw in must_contain)


def retrieval_hit(question: str, must_contain: list[str], k: int = 4) -> bool:
    """
    Diagnostic: did the vector store surface a chunk containing
    the expected keyword at all? Separates retrieval failures
    from generation failures.
    """
    if not must_contain:
        return True

    docs = search(question, k=k)
    combined = " ".join(d.page_content for d in docs).lower()
    return any(kw.lower() in combined for kw in must_contain)


def run_one(agent, case: dict) -> dict:
    """
    Runs a single eval case and returns a result record.
    """
    start = time.time()

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": case["question"]}]
        })
        answer = result["messages"][-1].content
        error = None
    except Exception as e:
        answer = ""
        error = str(e)

    latency = time.time() - start

    if error:
        passed = False
        reason = f"error: {error}"
    elif case["answerable"]:
        retrieved = retrieval_hit(case["question"], case["must_contain"])
        keywords_ok = score_keywords(answer, case["must_contain"])
        passed = keywords_ok
        if not retrieved:
            reason = "retrieval miss - expected content not in top-k"
        elif not keywords_ok:
            reason = "retrieved but answer missing expected keywords"
        else:
            reason = "ok"
    else:
        refused = looks_like_refusal(answer)
        passed = refused
        reason = "ok - correctly declined" if refused else "HALLUCINATION - answered an unanswerable question"

    return {
        "id": case["id"],
        "category": case["category"],
        "answerable": case["answerable"],
        "question": case["question"],
        "answer": answer,
        "passed": passed,
        "reason": reason,
        "latency": round(latency, 2),
    }


def report(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    print("\n" + "=" * 70)
    print(f"OVERALL: {passed}/{total} passed ({passed/total*100:.0f}%)")
    print("=" * 70)

    if answerable:
        a_passed = sum(1 for r in answerable if r["passed"])
        print(f"Answerable questions:   {a_passed}/{len(answerable)}")

    if unanswerable:
        u_passed = sum(1 for r in unanswerable if r["passed"])
        hallucinations = len(unanswerable) - u_passed
        print(f"Unanswerable questions: {u_passed}/{len(unanswerable)}"
              f"  ({hallucinations} hallucinated)")

    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    print("\nBy category:")
    for cat, rows in sorted(by_cat.items()):
        c_passed = sum(1 for r in rows if r["passed"])
        print(f"  {cat:<16} {c_passed}/{len(rows)}")

    avg_latency = sum(r["latency"] for r in results) / total
    print(f"\nAvg latency: {avg_latency:.2f}s")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n" + "-" * 70)
        print("FAILURES")
        print("-" * 70)
        for r in failures:
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  reason: {r['reason']}")
            print(f"  answer: {r['answer'][:200]}")


def main():
    dataset = load_dataset()
    agent = build_agent()

    results = []
    for case in dataset:
        print(f"running {case['id']}...", flush=True)
        results.append(run_one(agent, case))

    report(results)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()