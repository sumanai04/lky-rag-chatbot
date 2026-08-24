"""Evaluation suite for the LKY RAG chatbot.

Two layers of evaluation:
1. Retrieval (mechanical, no LLM): for each gold question, check whether the
   expected fact phrases appear in the top-k retrieved chunks.
   Reports fact recall and question-level hit rate.
2. Generation (LLM-as-judge): a judge prompt scores each generated answer on
   faithfulness, persona, and relevance (1-5 each).

Run after ingest.py:
    python eval.py
Writes eval/eval_results.json and prints a summary table.
"""

import json
import re
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

from rag import LKYChatbot  # noqa: E402

JUDGE_PROMPT = """You are grading a chatbot that impersonates Lee Kuan Yew using retrieved source material.

Score the ANSWER on three dimensions, each an integer from 1 to 5:
1. faithfulness: every factual claim in the answer is supported by the CONTEXT (5 = fully supported, 1 = fabricated claims)
2. persona: sounds like Lee Kuan Yew, first person, direct and pragmatic (5 = convincing LKY voice, 1 = generic assistant)
3. relevance: directly answers the QUESTION (5 = fully addressed, 1 = off topic)

Output ONLY a JSON object like {{"faithfulness": 5, "persona": 4, "relevance": 5, "note": "short line"}} and nothing else.

CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}"""


def judge(bot, question, answer, context):
    prompt = JUDGE_PROMPT.format(context=context[:6000], question=question, answer=answer)
    response = bot.client.chat.completions.create(
        model=bot.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {
            "faithfulness": int(data["faithfulness"]),
            "persona": int(data["persona"]),
            "relevance": int(data["relevance"]),
            "note": str(data.get("note", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def main():
    bot = LKYChatbot()
    with open("eval/gold_qa.json", encoding="utf-8") as f:
        gold = json.load(f)

    rows = []
    for item in gold:
        result = bot.retrieve(item["question"])
        chunks = result["documents"][0]
        metadatas = result["metadatas"][0]
        blob = "\n".join(chunks).lower()

        found = [fact for fact in item["expected_facts"] if fact.lower() in blob]
        missing = [fact for fact in item["expected_facts"] if fact.lower() not in blob]
        fact_recall = len(found) / len(item["expected_facts"]) if item["expected_facts"] else 1.0

        answer, scores = None, None
        if bot.client is not None:
            answer, sources = bot.answer(item["question"], return_sources=True)
            scores = judge(bot, item["question"], answer, "\n\n---\n\n".join(chunks))

        rows.append({
            "question": item["question"],
            "category": item.get("category", "general"),
            "retrieved_sources": [m.get("title") for m in metadatas],
            "expected_facts_found": found,
            "expected_facts_missing": missing,
            "expected_facts": item["expected_facts"],
            "answer": answer,
            "judge_scores": scores,
        })
        status = "OK" if fact_recall == 1.0 else "PARTIAL" if found else "MISS"
        print(f"[{status}] {item['question'][:60]}...  facts {len(found)}/{len(item['expected_facts'])}")

    scored = [r for r in rows if r["judge_scores"]]
    hit_rate = mean(1.0 if not r["expected_facts_missing"] else 0.0 for r in rows)
    total_found = sum(len(r["expected_facts_found"]) for r in rows)
    total_expected = sum(len(r["expected_facts"]) for r in rows)
    fact_recall = total_found / total_expected if total_expected else 1.0
    aggregate = {
        "questions": len(rows),
        "judged": len(scored),
        "retrieval_question_hit_rate": round(hit_rate, 3),
        "retrieval_fact_recall": round(fact_recall, 3),
    }
    if not scored:
        aggregate["note"] = (
            "generation layer skipped: DEEPSEEK_API_KEY is not set. "
            "Fill .env and rerun to get judge scores."
        )
    for dim in ("faithfulness", "persona", "relevance"):
        if scored:
            aggregate[f"mean_{dim}"] = round(mean(r["judge_scores"][dim] for r in scored), 2)

    output = {
        "config": {
            "model": bot.model,
            "top_k": bot.top_k,
            "temperature": bot.temperature,
            "judge": bot.model,
        },
        "aggregate": aggregate,
        "questions": rows,
    }
    with open("eval/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    for key, value in aggregate.items():
        print(f"  {key}: {value}")
    print("Wrote eval/eval_results.json")


if __name__ == "__main__":
    main()
