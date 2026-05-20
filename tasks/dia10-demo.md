# Dia 10 — Preparação e Ensaio da Demo Final

## Objetivo
Preparar e ensaiar a demo de 30 minutos que apresenta o sistema completo
funcionando em produção.

## Pré-requisitos
- Todos os dias 1–9 concluídos
- Todos os sistemas em produção testados no dia anterior (Dia 9)

## Estrutura da demo (30 minutos)

| Tempo | Seção | Conteúdo |
|-------|-------|----------|
| 5 min | Contexto | O que foi o sprint, o que foi construído, qual era o objetivo |
| 8 min | Agente ao vivo | 3 perguntas (1 por ferramenta), mostrar trace no LangSmith em tempo real |
| 7 min | Avaliação | Resultados de tool correctness e G-Eval, analisar 1 caso de falha ao vivo |
| 7 min | Produção | Abrir endpoint no browser/curl, mostrar dashboard Cloud Monitoring com métricas reais |
| 3 min | Decisões técnicas | Por que Qdrant → Vertex AI? Por que Cloud Run? O que faria diferente? |

## Checklist de preparação

### No dia anterior (Dia 9):
- [ ] Testar todos os endpoints em produção com os curls do Dia 6 e 7
- [ ] Confirmar que LangSmith está recebendo traces
- [ ] Confirmar que Cloud Monitoring tem dados (fazer algumas requests para gerar métricas)

### No dia da demo, antes de começar:
- [ ] Abrir previamente: smith.langchain.com, Console GCP → Cloud Monitoring
- [ ] Abrir terminal com os curls prontos (não digitar ao vivo)
- [ ] Ter o README aberto com as URLs de produção

## As 3 perguntas de exemplo (testar antes)

```bash
AGENT_URL=https://nicolas-agent-xxxx.run.app

# 1. RAGTool
curl -X POST $AGENT_URL/agent/ask \
  -d '{"question":"What are the risks of bond investing?"}'

# 2. CalculatorTool
curl -X POST $AGENT_URL/agent/ask \
  -d '{"question":"What is 15% of 3000?"}'

# 3. DateTool
curl -X POST $AGENT_URL/agent/ask \
  -d '{"question":"How many years ago was 2010?"}'
```

## Decisões técnicas para defender

| Decisão | Por que |
|---------|---------|
| `create_react_agent` vs StateGraph customizado | Mínimo de código, cobertura completa, trace automático |
| Qdrant local → Vertex AI em produção | Qdrant não é gerenciado, Vertex AI é escalável e integrado ao GCP |
| Cloud Run vs Vertex AI Endpoints | Cloud Run é mais simples para APIs HTTP, Vertex AI Endpoints é para modelos ML |
| LangSmith para observabilidade | Nativo ao LangGraph, zero instrumentação extra |

## Resposta preparada

**"O que você aprendeu sobre avaliação de IA que não sabia antes?"**

Sugestão de estrutura:
- Diferença entre avaliar RAG vs avaliar agentes (decisões de ferramenta, multi-step)
- Por que tool correctness é necessário além de qualidade de resposta
- O que os critérios G-Eval revelam que métricas simples não capturam

## Definição de feito
- [ ] Demo ensaiada pelo menos 1 vez completa (cronometrar os 30 min)
- [ ] Todos os sistemas em produção confirmados no dia anterior
- [ ] 3 perguntas de exemplo testadas end-to-end em produção
- [ ] Respostas preparadas para perguntas técnicas sobre decisões de design
- [ ] Resposta preparada para "o que você aprendeu"

## Dependências
- Todos os dias 1–9 concluídos
- Sistema completo em produção funcionando
