# RAG Evaluation Framework

Benchmark sistemático que compara estratégias de RAG sobre FiQA-2018 e quantifica impacto de cada decisão de design.

## Matriz experimental (3×3 = 9 pipelines)

- **Chunking:** fixed (512/64), semantic (percentile 95), hierarchical (1024/256)
- **Embeddings:** `text-embedding-3-small` (fuelix), `bge-small-en-v1.5` (local), `bge-large-en-v1.5` (local)

LLM gerador fixo: `claude-sonnet-4-5` via fuelix. Judge: `gpt-4o`. Dataset: FiQA-2018 via BEIR.

### Pipelines

| ID | Chunking | Embedding |
|----|----------|-----------|
| P1 | fixed | text-embedding-3-small |
| P2 | fixed | bge-small-en-v1.5 |
| P3 | fixed | bge-large-en-v1.5 |
| P4 | semantic | text-embedding-3-small |
| P5 | semantic | bge-small-en-v1.5 |
| P6 | semantic | bge-large-en-v1.5 |
| P7 | hierarchical | text-embedding-3-small |
| P8 | hierarchical | bge-small-en-v1.5 |
| P9 | hierarchical | bge-large-en-v1.5 |

## Arquitetura

```
corpus FiQA → chunker → embedder → Qdrant
                                     ↓
query → embedder → retriever (top-k) → prompt → LLM → resposta
                                     ↓
                          evaluator (retrieval + RAGAS + DeepEval) → report
```

Módulos em `src/rag_eval/`:

- `pipeline.py` — orquestra retrieve → generate, mede latência
- `chunking/` — chunkers fixed, semantic, hierarchical (IDs determinísticos)
- `embeddings/` — fuelix + local (fastembed), cache SQLite sha256
- `generation/` — cliente fuelix async com cache em disco; prompt RAG com citações `[1][2]`
- `retrieval/` — wrapper Qdrant async + retriever (dedup por doc-id)
- `data/fiqa_dataset.py` — loader FiQA via BEIR, subsample preservando queries relevantes
- `evaluation/` — métricas retrieval, RAGAS, DeepEval, failure analyzer, composite score
- `benchmark/` — `PipelineMatrix` (P1–P9), `BenchmarkRunner`, gerador de relatório markdown
- `api/` — FastAPI (rotas query, experiments, health)
- `storage/` — registro SQLite de experimentos + results store (JSON/parquet)

## Setup

Requer Python 3.11+, [uv](https://docs.astral.sh/uv/) e Docker.

```bash
uv sync
docker compose up -d qdrant
cp .env.example .env  # preencher FUELIX_API_KEY
```

## Variáveis de ambiente

| Variável | Default | Função |
|----------|---------|--------|
| `FUELIX_API_KEY` | — | obrigatória; acessa Claude + embeddings + judge |
| `FUELIX_BASE_URL` | `https://api.fuelix.ai/v1` | endpoint OpenAI-compatível |
| `QDRANT_URL` | `http://localhost:6333` | URL do Qdrant |
| `GENERATOR_MODEL` | `claude-sonnet-4-5` | LLM gerador |
| `JUDGE_MODEL` | `gpt-4o` | LLM judge (RAGAS/DeepEval) |
| `EMBEDDING_CACHE_PATH` | `data/cache/embeddings.sqlite` | cache de embeddings |
| `LLM_CACHE_PATH` | `data/cache/llm.sqlite` | cache de geração |
| `EXPERIMENT_STORE_PATH` | `data/experiments.sqlite` | registry de runs |
| `RESULTS_DIR` | `data/results` | saídas por experimento |
| `SUBSAMPLE_SIZE` | `10000` | tamanho do corpus subsampled |
| `TOP_K` | `10` | chunks recuperados por query |
| `SEED` | `42` | reprodutibilidade |
| `LOG_LEVEL` | `INFO` | nível de log |

## Uso via CLI

### Benchmark completo

```bash
uv run scripts/run_benchmark.py --name full_v1
```

Flags:
- `--pipelines P1 P5` — filtrar IDs (default: todos os 9)
- `--subsample N` — limitar corpus
- `--queries N` — limitar nº de queries avaliadas
- `--top-k 10` — chunks por query
- `--judge gpt-4o` — override judge model
- `--force-reindex` — reindexar Qdrant
- `--force-regen` — invalidar cache de geração
- `--skip-judges` — pular RAGAS + DeepEval (só métricas retrieval)

Smoke test rápido:

```bash
uv run scripts/run_benchmark.py --pipelines P1 --subsample 500 --queries 50 --skip-judges
```

### Só indexar (popular Qdrant)

```bash
uv run scripts/ingest.py --pipelines P1 P5 --subsample 10000 --force
```

## Uso via API

```bash
uv run uvicorn rag_eval.main:app --reload
```

### Fluxo didático (recomendado)

5 arquivos `.rest` numerados em [`requests/`](requests/README.md). Siga `1 → 2 → 3 → 4 → 5`:

| # | Arquivo | Endpoint principal | O que mostra |
|---|---|---|---|
| 1 | [`requests/1-health.rest`](requests/1-health.rest) | `GET /health` | API + Qdrant + coleções existentes |
| 2 | [`requests/2-dataset.rest`](requests/2-dataset.rest) | `GET /dataset/*` | conteúdo do FiQA (preview, queries, docs) |
| 3 | [`requests/3-index.rest`](requests/3-index.rest) | `POST /index` | popular Qdrant com 3 embedders (idempotente) |
| 4 | [`requests/4-ask.rest`](requests/4-ask.rest) | `POST /ask` | RAG: pergunta + chunks + resposta com citações + controle sem-RAG |
| 5 | [`requests/5-compare.rest`](requests/5-compare.rest) | `POST /compare` | mesma pergunta × 3 embedders lado a lado |
| 6 | [`requests/6-evaluate.rest`](requests/6-evaluate.rest) | `POST /evaluate` | benchmark de retrieval × ground truth FiQA (6 métricas + nota 0-100) |

Exemplo:

```bash
# 1. Popular o Qdrant uma vez (3 embedders, idempotente)
curl -X POST localhost:8000/index -H "Content-Type: application/json" -d '{}'

# 2. Perguntar (com controle sem-RAG ativo)
curl -X POST localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a business expense?","embedder":"openai","top_k":5}'

# 3. Comparar embedders na mesma pergunta
curl -X POST localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a business expense?","top_k":5}'
```

### Endpoints didáticos

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | health check + Qdrant ping |
| GET | `/dataset/preview` | stats + sample queries/docs do FiQA |
| GET | `/dataset/head` | head do ground truth: N queries (random determinístico) + texto dos docs esperados inline |
| GET | `/dataset/queries?search=...` | busca queries por palavra-chave |
| GET | `/dataset/documents/{doc_id}` | doc cru + queries que apontam pra ele |
| POST | `/index` | popula Qdrant com 3 embedders × chunking fixed (idempotente) |
| POST | `/ask` | RAG: pergunta → JSON `{ question, answer, retrieved_documents, response_without_rag? }` |
| POST | `/compare` | mesma pergunta × 3 embedders → JSON `{ question, results: { openai, bge-small, bge-large } }` |
| POST | `/evaluate` | retrieval × ground truth FiQA: 1 embedder × N queries → JSON `{ embedder, queries_avaliadas, hit_rate, precision@10, recall@10, ndcg@10, score }` |

`POST /ask` body:

```json
{ "question": "...", "embedder": "openai" | "bge-small" | "bge-large",
  "top_k": 5, "with_control": true }
```

`POST /compare` body:

```json
{ "question": "...", "top_k": 5 }
```

Resposta padrão = JSON minimal `{ question, answer, retrieved_documents, response_without_rag? }`. `?format=markdown` para versão humana renderizada.

### Endpoints avançados (análise quantitativa)

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/experiments` | lista os 9 pipelines da matriz (P1..P9) |
| POST | `/benchmark` | roda matriz inline e devolve digest + relatório markdown |
| POST | `/experiments/run` | dispara benchmark em background (202) — runs longos com judges |
| GET | `/experiments/{id}` | status + progresso + métricas |
| GET | `/experiments/{id}/report` | markdown do relatório |

Use Swagger (`/docs`) ou `curl` direto. Não estão nos `.rest` didáticos pra não poluir o fluxo.

## Avaliação

**Retrieval (doc-level):** nDCG@10, Recall@10, Precision@10, MRR, Hit@{1,3,5,10}.

**Geração (judge-based):** RAGAS (faithfulness, answer_relevancy, context_precision) + DeepEval.

**Composite score:** soma ponderada (`evaluation/statistics_calculator.py`).

**Failure analysis:** `evaluation/failure_analyzer.py` categoriza queries com baixo score.

## Saídas

Por experimento, em `data/results/<experiment_id>/`:
- `report.md` — comparativo dos pipelines
- `aggregates.json` / `.parquet` — métricas agregadas
- `queries.parquet` — resultados por query

## Testes

```bash
uv run pytest tests/
```

Cobre: chunkers, matriz de pipelines, cache de embeddings, prompt, evaluator retrieval, statistics, dedup do retriever, failure analyzer, smoke da API.

## Estrutura

```
src/rag_eval/
├── api/              # FastAPI app + routers (query, experiments)
├── benchmark/        # PipelineMatrix, BenchmarkRunner, ReportGenerator
├── chunking/         # fixed, semantic, hierarchical chunkers
├── data/             # FiQA loader (BEIR)
├── embeddings/       # fuelix + local + cache SQLite
├── evaluation/       # retrieval, RAGAS, DeepEval, failure, stats
├── generation/       # LLM client + prompt RAG
├── retrieval/        # Qdrant store + retriever
├── storage/          # experiment store + results store
├── config.py         # Settings (env)
└── pipeline.py       # RAGPipeline (retrieve+generate)
scripts/
├── run_benchmark.py  # CLI benchmark
└── ingest.py         # CLI só indexação
tests/                # unit + smoke
docker-compose.yml    # Qdrant
```
