# `requests/` — fluxo didático ponta-a-ponta

5 arquivos numerados. Siga `1 → 2 → 3 → 4 → 5`. Cada um responde uma pergunta sobre o sistema.

| # | Arquivo | O que mostra |
|---|---|---|
| 1 | `1-health.rest` | API up? Qdrant up? Quais coleções já existem? |
| 2 | `2-dataset.rest` | O que tem dentro do FiQA? (queries, docs, stats) |
| 3 | `3-index.rest` | Popular Qdrant com 3 embedders × chunking fixed. **Rodar 1×.** Idempotente. |
| 4 | `4-ask.rest` | RAG didático: pergunta → JSON `{ question, answer, retrieved_documents, response_without_rag? }` |
| 5 | `5-compare.rest` | Mesma pergunta × 3 embedders → JSON `{ question, results: { openai, bge-small, bge-large } }` |

## Pré-requisitos

- `docker compose up -d` (API em `:8000`, Qdrant em `:6333`)
- `.env` com `FUELIX_API_KEY` válida
- Rodar `3-index.rest` ao menos uma vez antes de `4-ask.rest` e `5-compare.rest`

## Conceito

Default das duas rotas é JSON minimal. `?format=markdown` opcional para humano.

- **`POST /ask`** → `{ question, answer, retrieved_documents: { "1": "...", "2": "..." }, response_without_rag? }`. `response_without_rag` aparece quando `with_control: true` (default): mesma pergunta, mesma LLM, sem ver o dataset — útil para comparar com a resposta grounded.
- **`POST /compare`** → `{ question, results: { openai: { answer, retrieved_documents }, bge-small: {...}, bge-large: {...} } }`. Embedder que falhou/sem coleção retorna `{ error: "..." }`.
- **`POST /index`** → bootstrap idempotente. Substitui rodar `/benchmark` só pra popular o Qdrant.

## Avançado (não está nos `.rest` numerados)

- `POST /benchmark` — métricas objetivas (nDCG, Recall, MRR, RAGAS faithfulness). Para análise quantitativa.
- `POST /experiments/run` — runs longos com judges em background.

Use via `curl` ou Swagger (`/docs`) quando precisar.
