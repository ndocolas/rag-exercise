# Dia 8 — Monitoramento com Cloud Monitoring

## Objetivo
Adicionar logging estruturado nos endpoints e criar dashboard de
monitoramento com alertas no Cloud Monitoring.

## Pré-requisitos
- Dias 6 e 7 concluídos: ambos os serviços no Cloud Run funcionando
- Acesso ao Console GCP

## Passo 1 — Logging estruturado nos endpoints

`structlog` já está instalado no projeto. Substituir `print()` por logs estruturados.

Cada request deve logar:
- `timestamp`
- `endpoint` (ex: `/ask`, `/agent/ask`)
- `latency_ms`
- `status_code`
- `vector_backend` (qdrant ou vertex)

Os logs aparecem automaticamente no Cloud Logging quando o serviço roda no Cloud Run.

Arquivos a modificar:
- `src/rag_eval/routes/query.py` — rota `/ask`
- `src/agent/main.py` — rota `/agent/ask`

Verificar: acessar **Cloud Logging** no Console GCP e confirmar que os logs chegam após uma request.

## Passo 2 — Criar dashboard no Cloud Monitoring

```
Console GCP → Cloud Monitoring → Dashboards → Create Dashboard
Nome: 'Nicolas — RAG + Agent Monitoring'
```

### Widget 1 — Latência (p50 e p95)
```
Fonte: Cloud Run > Request Latencies
Filtrar: resource.labels.service_name = nicolas-rag OR nicolas-agent
```

### Widget 2 — Taxa de erro (5xx)
```
Fonte: Cloud Run > Request Count
Filtrar: response_code_class = 5xx
```

### Widget 3 — Volume de requests
```
Fonte: Cloud Run > Request Count total
Granularidade: por minuto
Filtrar: nicolas-rag e nicolas-agent
```

## Passo 3 — Configurar alerta

```
Console GCP → Cloud Monitoring → Alerting → Create Policy
```

Condição: taxa de erro (5xx) > 5% por 5 minutos consecutivos

Notificação: email (nicolas.docolas@poatek.com)

Documentar no README:
- Nome da policy
- Condição exata
- Canal de notificação

## Definição de feito
- [ ] Logs estruturados chegando no Cloud Logging (tirar screenshot para o README)
- [ ] Dashboard `'Nicolas — RAG + Agent Monitoring'` criado com 3 widgets
- [ ] 1 alerta configurado e ativo (tirar screenshot da configuração)
- [ ] Alerta documentado no README (nome, condição, canal)

## Dependências
- `nicolas-rag` e `nicolas-agent` rodando no Cloud Run (Dias 6 e 7)
- `structlog` já instalado (está em `pyproject.toml`)
