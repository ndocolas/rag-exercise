# Fase 1 — Suite de Avaliação do Agente (Dias 3–4)

## Objetivo

Avaliar o agente em dois eixos:
1. **Tool correctness**: o agente escolhe a ferramenta certa?
2. **G-Eval**: as respostas atendem critérios de qualidade para o domínio financeiro?

## Arquivos

```
tests/eval/
  tool_correctness_cases.json   ← 20 casos {question, expected_tool}
  eval_tool_correctness.py      ← roda agente, compara ferramenta, imprime accuracy
  eval_geval.py                 ← G-Eval com DeepEval, salva geval_results.json
  geval_results.json            ← (gerado por eval_geval.py)
report_agent.md                 ← relatório final (Dia 4)
```

## Tool Correctness

### Distribuição dos casos (20 total)
| Ferramenta | Casos |
|-----------|-------|
| RAGTool | 8 |
| CalculatorTool | 7 |
| DateTool | 5 |

### Formato `tool_correctness_cases.json`
```json
[
  {"question": "What are the risks of bond investing?", "expected_tool": "RAGTool"},
  {"question": "What is 15% of 3000?", "expected_tool": "CalculatorTool"},
  {"question": "How many years ago was 2010?", "expected_tool": "DateTool"}
]
```

### Script `eval_tool_correctness.py`
- Carrega os 20 casos
- Para cada caso: roda agente, extrai nome da primeira tool chamada do primeiro `ToolMessage`
- Compara com `expected_tool` (case-insensitive)
- Output: tabela por ferramenta (acertos/total) + accuracy geral + casos errados

## G-Eval

### 3 Critérios (domínio financeiro)

| # | Nome | Critério |
|---|------|----------|
| 1 | Factual accuracy | A resposta contém apenas afirmações financeiras corretas |
| 2 | Source citation | A resposta cita fontes como [1], [2] quando usa RAG |
| 3 | Appropriate confidence | A resposta evita afirmações absolutas sobre movimentos de mercado |

### Script `eval_geval.py`
- 10 perguntas hardcoded (mix das 3 ferramentas)
- Para cada: roda agente → `LLMTestCase(input=q, actual_output=answer)`
- Avalia nos 3 critérios com `GEval` do DeepEval
- Salva `tests/eval/geval_results.json`: `{question, scores, passed}`

## Relatório `report_agent.md` (Dia 4)

### Seções obrigatórias
1. **Tool Correctness**: accuracy geral + tabela acertos/erros por ferramenta
2. **G-Eval**: score médio por critério + 2 exemplos de resposta com análise
3. **Análise de falhas**: em quais tipos de pergunta o agente erra mais? Por quê?
4. **Conclusão**: o agente está pronto para produção? O que precisaria melhorar?

### Definição de feito (Dia 3)
- [ ] `tool_correctness_cases.json` com 20 casos (8/7/5)
- [ ] `eval_tool_correctness.py` rodando sem erros e imprimindo accuracy
- [ ] 3 critérios G-Eval implementados e documentados no código
- [ ] `eval_geval.py` rodando e salvando `geval_results.json`

### Definição de feito (Dia 4)
- [ ] `report_agent.md` commitado no repositório
- [ ] Todas as métricas numéricas presentes (accuracy, scores G-Eval)
- [ ] Pelo menos 2 exemplos de falha analisados com explicação
- [ ] Seção de conclusão com recomendação clara (sim/não está pronto e por quê)
