"""Core RAG logic for the Lee Kuan Yew persona chatbot.

Usage:
    python rag.py ask "What did you feel when Singapore left Malaysia?"
    python rag.py            # interactive loop
"""

import os
import re
import sys
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

import chromadb  # noqa: E402
from openai import OpenAI  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

CHROMA_DIR = "chroma_db"
COLLECTION = "lky"
RRF_K = 60          # reciprocal rank fusion constant
CANDIDATE_N = 200   # candidates per retriever before fusion

SYSTEM_PROMPT = """You are Lee Kuan Yew, the founding father and first Prime Minister of Singapore. You speak in the first person, in your own voice: direct, pragmatic, unsentimental, blunt about hard truths, precise about facts. You care above all about Singapore's survival, meritocracy, discipline, clean government, and the real conditions of power.

Rules:
1. Ground every answer in the CONTEXT below. Never invent quotes, dates, policies, or events that are not in it.
2. If the context does not cover the question, say so in your own voice, plainly, and add the closest thing you have on record about the topic if there is one.
3. Keep answers under 150 words unless the question genuinely requires more.
4. Speak like a person giving a frank answer, not like an encyclopaedia. Do not use bullet lists unless asked. Do not use em dashes.
5. You are answering as the historical Lee Kuan Yew for a learning exercise. State your views firmly, including uncomfortable ones, as long as the context supports them.

CONTEXT:
{context}"""


class LKYChatbot:
    def __init__(self):
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)
        self.chroma = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma.get_collection(COLLECTION)
        self.client = None
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
            )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.top_k = int(os.getenv("TOP_K", "8"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.3"))
        self.rerank_pool = int(os.getenv("RERANK_POOL_SIZE", "300"))
        self.use_expansion = (
            self.client is not None
            and os.getenv("QUERY_EXPANSION", "true").lower() in {"1", "true", "yes"}
        )
        self.reranker = CrossEncoder(
            os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )

        # Load the whole corpus into memory for the BM25 lexical retriever.
        corpus = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = corpus["ids"]
        self.corpus_docs = corpus["documents"]
        self.corpus_metas = corpus["metadatas"]
        tokenized = [re.findall(r"\w+", doc.lower()) for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized)

    def _expand_query(self, question):
        """Rewrite the question in LKY's own vocabulary before retrieval."""
        if self.client is None:
            return question
        messages = [
            {
                "role": "system",
                "content": "Rewrite the question into one short, keyword-rich search query phrased the way Lee Kuan Yew would phrase it, using his own vocabulary. Output only the query, nothing else.",
            },
            {"role": "user", "content": question},
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0
        )
        return response.choices[0].message.content.strip() or question

    def retrieve(self, question, k=None):
        """Hybrid retrieval (BM25 + dense, RRF fused) with cross-encoder reranking."""
        k = k or self.top_k
        if self.use_expansion:
            question = self._expand_query(question)

        query_embedding = self.embedder.encode(question).tolist()
        dense_n = min(CANDIDATE_N, len(self.corpus_ids))
        dense = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=dense_n,
            include=["documents", "metadatas"],
        )
        docs_by_id = {
            cid: (doc, meta)
            for cid, doc, meta in zip(dense["ids"][0], dense["documents"][0], dense["metadatas"][0])
        }

        tokens = re.findall(r"\w+", question.lower())
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = sorted(range(len(self.corpus_ids)), key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATE_N]

        fused = defaultdict(float)
        for rank, cid in enumerate(dense["ids"][0]):
            fused[cid] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_top):
            cid = self.corpus_ids[idx]
            fused[cid] += 1.0 / (RRF_K + rank + 1)
            if cid not in docs_by_id:
                docs_by_id[cid] = (self.corpus_docs[idx], self.corpus_metas[idx])

        pool = sorted(fused, key=fused.get, reverse=True)[:self.rerank_pool]
        pairs = [(question, docs_by_id[cid][0]) for cid in pool]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(pool, rerank_scores), key=lambda item: -item[1])
        top_ids = [cid for cid, _ in ranked[:k]]
        return {
            "ids": [top_ids],
            "documents": [[docs_by_id[cid][0] for cid in top_ids]],
            "metadatas": [[docs_by_id[cid][1] for cid in top_ids]],
            "distances": [[0.0] * len(top_ids)],
        }

    def answer(self, question, return_sources=False):
        if self.client is None:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env, fill the "
                "key, and retry. Retrieval works without a key; generation does not."
            )
        result = self.retrieve(question)
        chunks = result["documents"][0]
        metadatas = result["metadatas"][0]
        context = "\n\n---\n\n".join(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        answer = response.choices[0].message.content
        if return_sources:
            return answer, list(zip(chunks, metadatas))
        return answer


def main():
    bot = LKYChatbot()
    if len(sys.argv) >= 3 and sys.argv[1] == "ask":
        question = " ".join(sys.argv[2:])
        answer, sources = bot.answer(question, return_sources=True)
        print(answer)
        print("\nSOURCES:")
        for i, (_, meta) in enumerate(sources, 1):
            print(f"  [{i}] {meta.get('title')} | {meta.get('date')}")
        return
    print("Ask Lee Kuan Yew. Type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue
        print("\nLKY:", bot.answer(question))


if __name__ == "__main__":
    main()
