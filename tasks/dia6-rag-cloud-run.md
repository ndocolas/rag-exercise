# Dia 6 — Deploy do RAG no Cloud Run

## Objetivo
Adaptar o retriever para suportar Vertex AI Vector Search e fazer deploy
do servidor RAG como endpoint público no Cloud Run.

## Pré-requisitos
- Dia 5 concluído: Vertex AI Index com status `DEPLOYED` e `INDEX_ENDPOINT_ID` anotado
- Docker funcionando localmente
- gcloud autenticado no projeto GCP

## Passo 1 — Criar `src/rag_eval/db/vertex_vector_store.py`

Implementar a mesma interface do `vector_store.py` atual (Qdrant),
mas usando o Vertex AI `IndexEndpoint` como backend.

Interface mínima a implementar:
```python
class VertexVectorStore:
    async def search(self, vector: list[float], top_k: int) -> list[SearchResult]:
        # usa index_endpoint.find_neighbors()
        ...
```

Variáveis de ambiente necessárias:
```
VECTOR_BACKEND=vertex         # ou "qdrant" (default)
INDEX_ENDPOINT_ID=...         # ID do endpoint do Dia 5
DEPLOYED_INDEX_ID=nicolas_rag_deployed
```

## Passo 2 — Atualizar retriever para suportar os dois backends

Em `src/rag_eval/services/retrieval/retriever.py`:
- Verificar `VECTOR_BACKEND` env var
- Instanciar `VertexVectorStore` se `vertex`, `QdrantVectorStore` se `qdrant`
- Qdrant continua funcionando localmente para desenvolvimento

Em `src/rag_eval/utils/settings.py`:
- Adicionar `vector_backend: str = "qdrant"`
- Adicionar `index_endpoint_id: str = ""`
- Adicionar `deployed_index_id: str = "nicolas_rag_deployed"`

## Passo 3 — Verificar Dockerfile existente

```bash
docker build -t rag-eval . && docker run --rm rag-eval python -c "import rag_eval; print('ok')"
```

Corrigir se necessário antes de prosseguir.

## Passo 4 — Build e push da imagem

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/nicolas-rag-eval
```

## Passo 5 — Deploy no Cloud Run

```bash
gcloud run deploy nicolas-rag \
  --image gcr.io/PROJECT_ID/nicolas-rag-eval \
  --region us-central1 \
  --set-env-vars VECTOR_BACKEND=vertex,\
    INDEX_ENDPOINT_ID=SEU_ID_AQUI,\
    DEPLOYED_INDEX_ID=nicolas_rag_deployed,\
    FUELIX_API_KEY=SEU_KEY_AQUI,\
    FUELIX_BASE_URL=https://api.fuelix.ai/v1,\
    GENERATOR_MODEL=claude-sonnet-4-5
```

## Passo 6 — Testar o endpoint público

```bash
URL=$(gcloud run services describe nicolas-rag --region us-central1 --format 'value(status.url)')

# Health check
curl -X GET $URL/health

# Pergunta via RAG com Vertex AI
curl -X POST $URL/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is a bond?","embedder":"openai"}'
```

## Arquivos a criar/modificar

| Arquivo | Ação |
|---------|------|
| `src/rag_eval/db/vertex_vector_store.py` | Novo |
| `src/rag_eval/services/retrieval/retriever.py` | Adaptar para VECTOR_BACKEND |
| `src/rag_eval/utils/settings.py` | Adicionar 3 novas vars |
| `.env.example` | Documentar vars novas |

## Definição de feito
- [ ] URL pública do endpoint RAG funcionando (documentar no README)
- [ ] `GET /health` retorna 200 com status do Vertex AI Vector Search
- [ ] `POST /ask` retorna resposta com citações usando Vertex AI como retriever
- [ ] Qdrant continua funcionando localmente (VECTOR_BACKEND=qdrant)

## Dependências
- Dia 5: Vertex AI Index com `DEPLOYED` status e `INDEX_ENDPOINT_ID` anotado
- Dockerfile existente funcionando (`docker build` sem erros)
