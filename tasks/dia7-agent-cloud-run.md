# Dia 7 — Deploy do Agente no Cloud Run

## Objetivo
Empacotar o agente LangGraph como servidor FastAPI e fazer deploy
como serviço público no Cloud Run, conectado ao RAG do Dia 6.

## Pré-requisitos
- Dia 6 concluído: URL pública do RAG funcionando (`POST /ask` retornando resposta)
- `agent.py` funcionando localmente

## Passo 1 — Criar `src/agent/main.py`

Servidor FastAPI simples com 1 endpoint:

```python
# POST /agent/ask
# Body: { "question": string }
# Response: { "answer": string, "tool_used": string }
```

O agente usa a `RAGTool` via HTTP apontando para `RAG_ENDPOINT_URL`
(URL do Cloud Run do Dia 6 — não mais localhost).

```
RAG_ENDPOINT_URL=https://nicolas-rag-xxxx.run.app
```

Também criar `src/agent/__init__.py` (vazio).

## Passo 2 — Criar Dockerfile para o agente

Opção A: Dockerfile separado em `src/agent/Dockerfile`
Opção B: Adaptar o Dockerfile existente com `ARG ENTRYPOINT`

O container deve executar `uvicorn src.agent.main:app`.

## Passo 3 — Build e push

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/nicolas-agent
```

## Passo 4 — Deploy no Cloud Run

```bash
gcloud run deploy nicolas-agent \
  --image gcr.io/PROJECT_ID/nicolas-agent \
  --region us-central1 \
  --set-env-vars \
    RAG_ENDPOINT_URL=https://nicolas-rag-xxxx.run.app,\
    FUELIX_API_KEY=SEU_KEY_AQUI,\
    FUELIX_BASE_URL=https://api.fuelix.ai/v1,\
    GENERATOR_MODEL=claude-sonnet-4-5,\
    LANGCHAIN_TRACING_V2=true,\
    LANGCHAIN_API_KEY=SEU_LANGSMITH_KEY,\
    LANGCHAIN_PROJECT=rag-exercise
```

## Passo 5 — Testar fluxo completo end-to-end

```bash
AGENT_URL=$(gcloud run services describe nicolas-agent --region us-central1 --format 'value(status.url)')

curl -X POST $AGENT_URL/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the risks of bond investing?"}'
```

Verificar a cadeia completa:
```
pergunta → Agente (Cloud Run) → RAGTool → RAG (Cloud Run) → Vertex AI → resposta
```

## Passo 6 — Verificar traces no LangSmith

Após o curl acima, verificar no LangSmith que o trace aparece com o fluxo completo
incluindo a chamada HTTP para o RAG.

## Arquivos a criar

| Arquivo | Ação |
|---------|------|
| `src/agent/__init__.py` | Novo (vazio) |
| `src/agent/main.py` | Novo (FastAPI wrapper do agent.py) |
| Dockerfile do agente | Novo ou adaptar existente |

## Definição de feito
- [ ] URL pública do endpoint do agente funcionando (documentar no README)
- [ ] `POST /agent/ask` retorna resposta com indicação da ferramenta usada
- [ ] Fluxo end-to-end funcionando: agente → RAG → Vertex AI
- [ ] Traces visíveis no LangSmith para chamadas em produção

## Dependências
- Dia 6: `nicolas-rag` URL pública funcionando
- `agent.py` funcionando localmente
