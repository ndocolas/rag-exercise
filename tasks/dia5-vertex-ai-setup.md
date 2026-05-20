# Dia 5 — Setup Vertex AI + Vector Search

## Contexto GCP
Projeto GCP compartilhado dos estagiários.
**Confirmar com Ester antes de começar:** ID do projeto GCP a usar.
Todos os recursos com prefixo `nicolas-` (ex: `nicolas-rag-index`) para não conflitar.

## Pré-requisitos
- gcloud CLI instalado e autenticado (`gcloud auth login`)
- Permissão Editor no projeto GCP (confirmar com Ester)
- Python instalado (`google-cloud-aiplatform` a instalar)

## Passo 1 — Habilitar APIs necessárias

```bash
gcloud services enable aiplatform.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com
```

Verificar:
```bash
gcloud services list --enabled | grep -E "aiplatform|storage|run"
```

## Passo 2 — Criar bucket GCS para artefatos

```bash
# Nome deve ser único globalmente — ajustar se necessário
gsutil mb -l us-central1 gs://nicolas-rag-artifacts
```

Região `us-central1`: mesma do Vertex AI para evitar custos de egress.

## Passo 3 — Exportar embeddings do corpus FiQA

Criar `scripts/export_embeddings_vertex.py` que:
1. Conecta no Qdrant local
2. Lê os vetores da coleção `openai` (text-embedding-3-small, 1536 dim)
3. Exporta em formato JSONL: `{"id": "doc_001", "embedding": [0.1, 0.2, ...]}`
4. Salva em `data/vertex_embeddings.jsonl`
5. Faz upload para `gs://nicolas-rag-artifacts/embeddings/`

Subset recomendado: 1000 documentos para teste inicial.

```bash
gsutil cp data/vertex_embeddings.jsonl gs://nicolas-rag-artifacts/embeddings/
```

## Passo 4 — Criar Vertex AI Vector Search Index

```python
from google.cloud import aiplatform

aiplatform.init(project="PROJECT_ID", location="us-central1")

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="nicolas-rag-index",
    contents_delta_uri="gs://nicolas-rag-artifacts/embeddings/",
    dimensions=1536,  # text-embedding-3-small
    approximate_neighbors_count=10,
)
```

⚠️ A criação demora **30–60 minutos**. Monitorar status no Console GCP.

## Passo 5 — Criar IndexEndpoint e fazer deploy

```python
index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="nicolas-rag-endpoint",
    public_endpoint_enabled=True,
)

index_endpoint.deploy_index(
    index=index,
    deployed_index_id="nicolas_rag_deployed",
)
```

Anotar o `INDEX_ENDPOINT_ID` gerado — necessário nos dias seguintes.

## Passo 6 — Testar query de similaridade

```python
response = index_endpoint.find_neighbors(
    deployed_index_id="nicolas_rag_deployed",
    queries=[embedding_vector],  # vetor de 1536 dim
    num_neighbors=5,
)
print(response)
```

Deve retornar 5 IDs de documentos vizinhos.

## Atenção — custos GCP
- Não deixar o index endpoint deployado sem uso
- Um index parado ainda tem custo mínimo de armazenamento
- Confirmar com Ester qualquer recurso antes de criar

## Definição de feito
- [ ] APIs habilitadas: `gcloud services list --enabled | grep aiplatform`
- [ ] Bucket GCS criado: `gsutil ls gs://nicolas-rag-artifacts`
- [ ] Index criado no Vertex AI com status `DEPLOYED` (Console GCP)
- [ ] Query de teste retornando os k vizinhos com sucesso
- [ ] `INDEX_ENDPOINT_ID` anotado para uso nos dias 6 e 7

## Dependências
- Qdrant local com dados indexados (servidor RAG rodando e `/index` já executado)
- Permissões GCP confirmadas com Ester
- `google-cloud-aiplatform` instalado: `pip install google-cloud-aiplatform`
