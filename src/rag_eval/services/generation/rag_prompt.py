from __future__ import annotations

from rag_eval.db.vector_store import RetrievedChunk


class RAGPrompt:
    """Builds RAG system + user messages. Single template, fixed across pipelines."""

    SYSTEM = (
        "You are a precise financial domain assistant. Answer the user's question using ONLY the "
        "information provided in the context passages. If the context does not contain enough "
        'information to answer, reply exactly: "I cannot answer based on the provided context." '
        "Be concise and cite the most relevant context numbers in square brackets like [1], [2]."
    )

    USER_TEMPLATE = "Context passages:\n{contexts}\n\nQuestion: {question}\n\nAnswer:"

    def build(self, question: str, retrieved: list[RetrievedChunk]) -> list[dict]:
        contexts = self._format_contexts(retrieved)
        return [
            {"role": "system", "content": self.SYSTEM},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(contexts=contexts, question=question),
            },
        ]

    @staticmethod
    def _format_contexts(retrieved: list[RetrievedChunk]) -> str:
        lines = []
        for i, r in enumerate(retrieved, start=1):
            text = (r.chunk.metadata or {}).get("parent_text") or r.chunk.text
            lines.append(f"[{i}] {text.strip()}")
        return "\n\n".join(lines) if lines else "(no context retrieved)"
