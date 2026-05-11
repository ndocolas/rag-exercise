from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass
class FiQAData:
    corpus: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    def relevant_doc_ids(self, query_id: str) -> set[str]:
        return {doc_id for doc_id, rel in self.qrels.get(query_id, {}).items() if rel > 0}


class FiQADataset:
    """Loads FiQA-2018 from BEIR and exposes corpus/queries/qrels.

    Subsampling preserves all queries that have at least one relevant doc still
    present in the sampled corpus, so retrieval metrics stay meaningful.
    """

    BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"

    def __init__(self, data_dir: Path, subsample_size: int | None = None, seed: int = 42):
        self._data_dir = Path(data_dir)
        self._subsample_size = subsample_size
        self._seed = seed
        self._data: FiQAData | None = None

    async def load(self) -> FiQAData:
        if self._data is not None:
            return self._data
        self._data = await asyncio.to_thread(self._load_sync)
        return self._data

    def _load_sync(self) -> FiQAData:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        self._data_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = self._data_dir / "fiqa"
        if not dataset_path.exists():
            util.download_and_unzip(self.BEIR_URL, str(self._data_dir))

        corpus_raw, queries_raw, qrels_raw = GenericDataLoader(data_folder=str(dataset_path)).load(
            split="test"
        )

        corpus = {
            doc_id: Document(doc_id=doc_id, text=item.get("text", ""), title=item.get("title", ""))
            for doc_id, item in corpus_raw.items()
        }

        if self._subsample_size and self._subsample_size < len(corpus):
            corpus = self._subsample(corpus, qrels_raw)

        queries = {
            qid: Query(query_id=qid, text=text)
            for qid, text in queries_raw.items()
            if any(doc_id in corpus and rel > 0 for doc_id, rel in qrels_raw.get(qid, {}).items())
        }

        qrels = {
            qid: {doc_id: rel for doc_id, rel in rels.items() if doc_id in corpus}
            for qid, rels in qrels_raw.items()
            if qid in queries
        }

        return FiQAData(corpus=corpus, queries=queries, qrels=qrels)

    def _subsample(
        self, corpus: dict[str, Document], qrels: dict[str, dict[str, int]]
    ) -> dict[str, Document]:
        relevant_ids: set[str] = set()
        for rels in qrels.values():
            relevant_ids.update(doc_id for doc_id, rel in rels.items() if rel > 0)

        relevant_kept = {doc_id: corpus[doc_id] for doc_id in relevant_ids if doc_id in corpus}

        remaining_budget = max(0, (self._subsample_size or 0) - len(relevant_kept))
        rng = random.Random(self._seed)
        non_relevant = [doc_id for doc_id in corpus if doc_id not in relevant_kept]
        rng.shuffle(non_relevant)
        sampled_non_relevant = non_relevant[:remaining_budget]

        sampled = dict(relevant_kept)
        for doc_id in sampled_non_relevant:
            sampled[doc_id] = corpus[doc_id]
        return sampled
