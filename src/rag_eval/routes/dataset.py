from __future__ import annotations

import random

from fastapi import APIRouter, Query
from pydantic import BaseModel

from rag_eval.services.data.fiqa_dataset import FiQADataset
from rag_eval.utils.settings import Settings


class DocumentPreview(BaseModel):
    doc_id: str
    title: str
    text: str
    chars: int


class QueryPreview(BaseModel):
    query_id: str
    text: str
    relevant_doc_ids: list[str]


class DatasetStats(BaseModel):
    corpus_size: int
    queries_size: int
    avg_doc_chars: float
    avg_query_chars: float
    avg_relevant_docs_per_query: float


class DatasetPreviewResponse(BaseModel):
    stats: DatasetStats
    sample_queries: list[QueryPreview]
    sample_documents: list[DocumentPreview]


class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    text: str
    queries_pointing_here: list[QueryPreview]


class FullDocument(BaseModel):
    doc_id: str
    title: str
    text: str


class FullQuery(BaseModel):
    query_id: str
    text: str
    relevant_doc_ids: list[str]


class FullDatasetResponse(BaseModel):
    stats: DatasetStats
    documents: list[FullDocument]
    queries: list[FullQuery]


class ExpectedDocument(BaseModel):
    doc_id: str
    title: str
    text: str
    chars: int


class HeadQuery(BaseModel):
    query_id: str
    question: str
    expected_documents: list[ExpectedDocument]


class DatasetHeadResponse(BaseModel):
    n: int
    queries: list[HeadQuery]


class DatasetRouter:
    """Inspect the loaded FiQA dataset: stats + sample queries + sample docs."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._dataset = FiQADataset(
            data_dir=settings.fiqa_data_dir,
            subsample_size=settings.subsample_size,
            seed=settings.seed,
        )
        self._router = APIRouter(prefix="/dataset", tags=["dataset"])
        self._register_routes()

    @property
    def router(self) -> APIRouter:
        return self._router

    def _register_routes(self) -> None:
        @self._router.get("/preview", response_model=DatasetPreviewResponse)
        async def preview(
            n_queries: int = Query(default=10, ge=1, le=50),
            n_docs: int = Query(default=5, ge=1, le=20),
            doc_preview_chars: int = Query(default=300, ge=50, le=2000),
        ) -> DatasetPreviewResponse:
            data = await self._dataset.load()
            rng = random.Random(self._settings.seed)

            query_ids = list(data.queries.keys())
            rng.shuffle(query_ids)
            sample_qids = query_ids[:n_queries]
            sample_queries = [
                QueryPreview(
                    query_id=qid,
                    text=data.queries[qid].text,
                    relevant_doc_ids=sorted(data.relevant_doc_ids(qid)),
                )
                for qid in sample_qids
            ]

            doc_ids = list(data.corpus.keys())
            rng.shuffle(doc_ids)
            sample_dids = doc_ids[:n_docs]
            sample_documents = [
                DocumentPreview(
                    doc_id=did,
                    title=data.corpus[did].title,
                    text=data.corpus[did].text[:doc_preview_chars]
                    + ("…" if len(data.corpus[did].text) > doc_preview_chars else ""),
                    chars=len(data.corpus[did].text),
                )
                for did in sample_dids
            ]

            total_doc_chars = sum(len(d.text) for d in data.corpus.values())
            total_query_chars = sum(len(q.text) for q in data.queries.values())
            total_relevant = sum(len(rels) for rels in data.qrels.values())
            stats = DatasetStats(
                corpus_size=len(data.corpus),
                queries_size=len(data.queries),
                avg_doc_chars=round(total_doc_chars / max(1, len(data.corpus)), 1),
                avg_query_chars=round(total_query_chars / max(1, len(data.queries)), 1),
                avg_relevant_docs_per_query=round(total_relevant / max(1, len(data.queries)), 2),
            )
            return DatasetPreviewResponse(
                stats=stats,
                sample_queries=sample_queries,
                sample_documents=sample_documents,
            )

        @self._router.get("/head", response_model=DatasetHeadResponse)
        async def head(
            n: int = Query(default=10, ge=1, le=50),
            preview_chars: int = Query(default=0, ge=0, le=5000),
            seed: int | None = Query(default=None),
        ) -> DatasetHeadResponse:
            data = await self._dataset.load()
            shuffle_seed = seed if seed is not None else self._settings.seed
            query_ids = list(data.queries.keys())
            random.Random(shuffle_seed).shuffle(query_ids)
            picked = query_ids[: min(n, len(query_ids))]

            head_queries: list[HeadQuery] = []
            for qid in picked:
                expected: list[ExpectedDocument] = []
                for did in sorted(data.relevant_doc_ids(qid)):
                    doc = data.corpus.get(did)
                    if doc is None:
                        continue
                    full_text = doc.text
                    text = (
                        full_text
                        if preview_chars == 0 or len(full_text) <= preview_chars
                        else full_text[:preview_chars] + "..."
                    )
                    expected.append(
                        ExpectedDocument(
                            doc_id=doc.doc_id,
                            title=doc.title,
                            text=text,
                            chars=len(full_text),
                        )
                    )
                head_queries.append(
                    HeadQuery(
                        query_id=qid,
                        question=data.queries[qid].text,
                        expected_documents=expected,
                    )
                )
            return DatasetHeadResponse(n=len(head_queries), queries=head_queries)

        @self._router.get("/queries", response_model=list[QueryPreview])
        async def list_queries(
            limit: int = Query(default=50, ge=1, le=500),
            offset: int = Query(default=0, ge=0),
            search: str | None = None,
        ) -> list[QueryPreview]:
            data = await self._dataset.load()
            items = list(data.queries.items())
            if search:
                s = search.lower()
                items = [(qid, q) for qid, q in items if s in q.text.lower()]
            sliced = items[offset : offset + limit]
            return [
                QueryPreview(
                    query_id=qid,
                    text=q.text,
                    relevant_doc_ids=sorted(data.relevant_doc_ids(qid)),
                )
                for qid, q in sliced
            ]

        @self._router.get("/all", response_model=FullDatasetResponse)
        async def get_all() -> FullDatasetResponse:
            data = await self._dataset.load()
            documents = [
                FullDocument(doc_id=d.doc_id, title=d.title, text=d.text)
                for d in data.corpus.values()
            ]
            queries = [
                FullQuery(
                    query_id=qid,
                    text=q.text,
                    relevant_doc_ids=sorted(data.relevant_doc_ids(qid)),
                )
                for qid, q in data.queries.items()
            ]
            total_doc_chars = sum(len(d.text) for d in data.corpus.values())
            total_query_chars = sum(len(q.text) for q in data.queries.values())
            total_relevant = sum(len(rels) for rels in data.qrels.values())
            stats = DatasetStats(
                corpus_size=len(data.corpus),
                queries_size=len(data.queries),
                avg_doc_chars=round(total_doc_chars / max(1, len(data.corpus)), 1),
                avg_query_chars=round(total_query_chars / max(1, len(data.queries)), 1),
                avg_relevant_docs_per_query=round(total_relevant / max(1, len(data.queries)), 2),
            )
            return FullDatasetResponse(stats=stats, documents=documents, queries=queries)

        @self._router.get("/documents/{doc_id}", response_model=DocumentResponse)
        async def get_document(doc_id: str) -> DocumentResponse:
            data = await self._dataset.load()
            if doc_id not in data.corpus:
                from fastapi import HTTPException

                raise HTTPException(404, f"doc_id {doc_id} not in subsampled corpus")
            doc = data.corpus[doc_id]
            pointing = [
                QueryPreview(
                    query_id=qid,
                    text=data.queries[qid].text,
                    relevant_doc_ids=sorted(data.relevant_doc_ids(qid)),
                )
                for qid, rels in data.qrels.items()
                if doc_id in rels and rels[doc_id] > 0 and qid in data.queries
            ]
            return DocumentResponse(
                doc_id=doc.doc_id,
                title=doc.title,
                text=doc.text,
                queries_pointing_here=pointing,
            )
