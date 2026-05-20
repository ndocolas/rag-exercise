# Sprint Overview — LangGraph Agent + Vertex AI em Produção

Duração: 10 dias úteis (dias 6–15 do plano)
Estagiário: Nicolas | Mentora: Ester Figini

## Fases

### FASE 1 — Agentes LangGraph + Avaliação (Dias 1–4)

| Dia | Task | Arquivo | Status |
|-----|------|---------|--------|
| 1 | Agente LangGraph com 3 ferramentas | tasks/fase1-agent-langgraph.md | ✅ código existe |
| 2 | LangSmith — observabilidade | tasks/dia2-langsmith.md | ⬜ pendente |
| 3 | Suite de avaliação (tool correctness + G-Eval) | tasks/fase1-evaluation.md | ✅ código existe |
| 4 | Relatório de avaliação do agente | tasks/dia4-report-agent.md | ⬜ pendente |

### FASE 2 — Vertex AI em Produção (Dias 5–10)

| Dia | Task | Arquivo | Status |
|-----|------|---------|--------|
| 5 | Setup Vertex AI + Vector Search | tasks/dia5-vertex-ai-setup.md | ⬜ pendente |
| 6 | Deploy RAG no Cloud Run | tasks/dia6-rag-cloud-run.md | ⬜ pendente |
| 7 | Deploy agente no Cloud Run | tasks/dia7-agent-cloud-run.md | ⬜ pendente |
| 8 | Monitoramento com Cloud Monitoring | tasks/dia8-monitoring.md | ⬜ pendente |
| 9 | Documentação final | tasks/dia9-documentation.md | ⬜ pendente |
| 10 | Preparação da demo | tasks/dia10-demo.md | ⬜ pendente |

## Entregáveis finais (checklist de entrega)

### Fase 1 — Agentes
- [ ] agent.py funcionando com 3 ferramentas (RAGTool, CalculatorTool, DateTool)
- [ ] 5+ execuções documentadas no README
- [ ] 10+ traces visíveis no LangSmith
- [ ] Guia LangSmith publicado no Confluence
- [ ] 20 casos de teste de tool correctness (tool_correctness_cases.json)
- [ ] Script eval_tool_correctness.py rodando e calculando accuracy
- [ ] 3 critérios G-Eval implementados e documentados
- [ ] Script eval_geval.py rodando e salvando resultados
- [ ] report_agent.md com métricas, análise de falhas e conclusão

### Fase 2 — Vertex AI em Produção
- [ ] Vertex AI Vector Search Index com status DEPLOYED
- [ ] URL pública do RAG no Cloud Run funcionando
- [ ] GET /health retorna status do Vertex AI
- [ ] POST /ask retorna resposta usando Vertex AI como retriever
- [ ] URL pública do agente no Cloud Run funcionando
- [ ] Fluxo end-to-end: agente → RAG → Vertex AI funcionando
- [ ] Logs estruturados visíveis no Cloud Logging
- [ ] Dashboard 'Nicolas — RAG + Agent Monitoring' com 3 widgets de métricas
- [ ] 1 alerta configurado e ativo
- [ ] README atualizado com diagrama de arquitetura e URLs de produção
- [ ] Guia LangSmith publicado no Confluence (link no README)
- [ ] Guia Vertex AI publicado no Confluence (link no README)

### Demo Final
- [ ] Demo de 30 min ensaiada e cronometrada
- [ ] Todos os sistemas em produção testados no dia anterior

## Regras do sprint
- Se travar por mais de 1 hora: comunicar à Ester
- Cada dia tem definição de feito clara — comunicar proativamente se não estiver concluído
- Todos os recursos GCP com prefixo 'nicolas-'
- Não deixar índices ou endpoints rodando sem necessidade (custo)
