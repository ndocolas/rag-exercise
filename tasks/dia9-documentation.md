# Dia 9 — Documentação Final

## Objetivo
Atualizar o README do repositório e publicar 2 guias no Confluence.

## Pré-requisitos
- Todos os dias 1–8 concluídos
- Sistema rodando em produção (Cloud Run + Vertex AI)
- Screenshots do LangSmith e Cloud Monitoring capturados

## Parte 1 — Atualizar README do repositório

### Seção nova: 'Arquitetura em produção'

Adicionar diagrama ASCII:
```
┌─────────────────────────────────────────────────────────┐
│                    Arquitetura em Produção              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   Usuário                                               │
│      │                                                  │
│      ▼                                                  │
│   Cloud Run (Agente)  nicolas-agent-xxx.run.app         │
│      │                                                  │
│      ├── RAGTool ──► Cloud Run (RAG)                    │
│      │                nicolas-rag-xxx.run.app            │
│      │                      │                           │
│      │               Vertex AI Vector Search             │
│      │               nicolas-rag-index                  │
│      │                                                  │
│   LangSmith (traces) │ Cloud Monitoring (métricas)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Seção nova: 'Endpoints em produção'
```
RAG:    https://nicolas-rag-xxxx.run.app
Agente: https://nicolas-agent-xxxx.run.app
```

### Atualizar seção 'Setup'
- Incluir instruções para Vertex AI (além do Qdrant local)
- Documentar variáveis de ambiente novas:
  - `VECTOR_BACKEND`
  - `INDEX_ENDPOINT_ID`
  - `DEPLOYED_INDEX_ID`
  - `RAG_ENDPOINT_URL`

## Parte 2 — Guia Confluence 1: 'Setup do LangSmith para observabilidade de agentes'

(Escrito no Dia 2 — revisar e publicar como página final, não draft)

Checklist de revisão:
- [ ] Screenshots reais do projeto no LangSmith incluídos
- [ ] Todos os passos testados por outra pessoa (ou simulado)
- [ ] Publicado como página final (não draft)
- [ ] Link adicionado ao README

## Parte 3 — Guia Confluence 2: 'Deploy de RAG e agente no Vertex AI'

Seções obrigatórias:
1. **Pré-requisitos**: APIs habilitadas, permissões, bucket GCS
2. **Criação do Vertex AI Vector Search Index**: comandos completos + tempo estimado
3. **Deploy no Cloud Run**: comandos completos, variáveis de ambiente, troubleshooting comum
4. **Verificação do deploy**: como testar que está funcionando (`curl` de exemplo)
5. **Monitoramento**: como acessar o dashboard e interpretar as métricas

O guia deve ser reutilizável por qualquer pessoa da equipe em projetos futuros.

## Definição de feito
- [ ] README atualizado com diagrama de arquitetura e URLs em produção
- [ ] Seção de setup do README cobre Vertex AI além do Qdrant local
- [ ] Guia LangSmith publicado no Confluence (link adicionado ao README)
- [ ] Guia Vertex AI publicado no Confluence (link adicionado ao README)

## Dependências
- Todos os serviços em produção funcionando (Dias 5–8)
- Screenshots do LangSmith capturados (Dia 2)
- Screenshots do Cloud Monitoring capturados (Dia 8)
