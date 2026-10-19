"""Benchmark the assistant on eval/questions.json.

Usage:
  python eval.py                                   # default model
  python eval.py --models llama3.1:8b qwen2.5:7b   # compare models
  python eval.py --no-hybrid                       # vector-only, to show hybrid's gain

A question passes if the answer contains ALL of its expected keywords. Matching ignores case
and treats hyphens, underscores and spaces alike ("Self-Review" matches "self review").
Questions marked "unanswerable" pass only if the assistant says it doesn't know.
"""
import argparse
import json
import re
from pathlib import Path

import rag


def _norm(text):
    return re.sub(r"[-_\s]+", " ", text.lower())


def passed(item, text):
    if item.get("unanswerable"):
        return text.lower().startswith("i don't know")
    answer = _norm(text)
    return all(_norm(kw) in answer for kw in item["keywords"])


def run(model, hybrid, items):
    ok, total_seconds, failures = 0, 0.0, []
    for item in items:
        out = rag.answer(item["question"], hybrid=hybrid, model=model)
        total_seconds += out["seconds"]
        if passed(item, out["answer"]):
            ok += 1
        else:
            failures.append((item["question"], out["answer"][:200]))
    return ok, total_seconds / len(items), failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[rag.LLM_MODEL])
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--file", default="eval/questions.json")
    args = parser.parse_args()

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    mode = "vector-only" if args.no_hybrid else "hybrid"
    print(f"{len(items)} questions, retrieval: {mode}\n")
    for model in args.models:
        ok, avg_s, failures = run(model, not args.no_hybrid, items)
        print(f"{model}: {ok}/{len(items)} = {100 * ok / len(items):.0f}% accuracy, {avg_s:.1f}s avg")
        for q, a in failures:
            print(f"   FAILED: {q}\n     got: {a}")
        print()
