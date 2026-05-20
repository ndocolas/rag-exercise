# Fase 1 — Agente LangGraph (Dias 1–2)

## Objetivo

Construir um agente LangGraph com 3 ferramentas financeiras e instrumentar com LangSmith.

## Arquitetura

```
agent.py                  ← CLI entry point
  ├── RAGTool             ← HTTP POST /ask no servidor RAG existente
  ├── CalculatorTool      ← sympy.sympify().evalf()
  └── DateTool            ← datetime.now() + dateutil.relativedelta
```

Agente: `langgraph.prebuilt.create_react_agent(llm, tools)`

LLM: `ChatOpenAI` com FUELIX_BASE_URL + FUELIX_API_KEY (mesmas vars do projeto).

## Uso

```bash
python agent.py --question "What are the risks of bond investing?"
python agent.py --question "What is 15% of 3000?"
python agent.py --question "How many years ago was 2010?"
python agent.py --question "What is the annual return if I invest 1000 at 5% for 3 years?"
```

Output: imprime qual ferramenta usou em cada passo + resposta final.

## Env vars

| Var | Default | Descrição |
|-----|---------|-----------|
| `FUELIX_API_KEY` | (obrigatório) | Auth Fuelix |
| `FUELIX_BASE_URL` | `https://api.fuelix.ai/v1` | OpenAI-compatible endpoint |
| `GENERATOR_MODEL` | `claude-sonnet-4-5` | Modelo LLM |
| `RAG_ENDPOINT_URL` | `http://localhost:8000` | URL do servidor RAG |
| `LANGCHAIN_TRACING_V2` | `false` | Ativa traces no LangSmith |
| `LANGCHAIN_API_KEY` | — | Chave LangSmith |
| `LANGCHAIN_PROJECT` | `rag-exercise` | Projeto no LangSmith |

## Ferramentas

### RAGTool
- Trigger: perguntas sobre conceitos financeiros, definições, análise do dataset FiQA
- Exemplo: "What are the risks of bond investing?"
- Implementação: `httpx.AsyncClient.post(RAG_ENDPOINT_URL + "/ask")`

### CalculatorTool
- Trigger: cálculos numéricos explícitos (porcentagem, juros, diferença)
- Exemplo: "What is 15% of 3000?"
- Implementação: `sympy.sympify(expression).evalf()`

### DateTool
- Trigger: data atual, diferenças de tempo
- Exemplo: "How many years ago was 2010?"
- Implementação: `datetime.now()` + `dateutil.relativedelta`

## Definição de feito (Dia 1)

- [ ] `agent.py` executável com `python agent.py --question "..."`
- [ ] Pelo menos 5 execuções documentadas no README (1 por ferramenta + 2 multi-step)
- [ ] Agente imprime qual ferramenta usou em cada passo

## Definição de feito (Dia 2 — LangSmith)

- [ ] `.env.example` com vars LangSmith documentadas
- [ ] 10+ traces visíveis no LangSmith com inputs, tool calls e outputs
- [ ] Cada trace mostra: entrada, ferramentas usadas com args, saída, latência
