# Dia 4 — Relatório de Avaliação do Agente

## Objetivo
Executar os scripts de avaliação, analisar os resultados e escrever
`report_agent.md` com conclusão sobre prontidão para produção.

## Pré-requisitos
- Dia 3 concluído (eval_tool_correctness.py e eval_geval.py rodando sem erros)
- Servidor RAG rodando localmente

## O que executar

### 1. Tool Correctness
```bash
# Com servidor RAG rodando em outro terminal
python tests/eval/eval_tool_correctness.py
```
Registrar: accuracy geral + acertos/erros por ferramenta.

### 2. G-Eval
```bash
python tests/eval/eval_geval.py
```
Registrar: score médio por critério salvo em `tests/eval/geval_results.json`.

Identificar: pelo menos 2 casos onde o agente errou e entender o motivo.

## Conteúdo do report_agent.md

### Seção 1 — Tool Correctness
- Accuracy geral (ex: 17/20 = 85%)
- Tabela com acertos/erros por ferramenta:

| Ferramenta      | Acertos | Total | Accuracy |
|-----------------|---------|-------|----------|
| rag_tool        |         |       |          |
| calculator_tool |         |       |          |
| date_tool       |         |       |          |

### Seção 2 — G-Eval
- Score médio por critério (Factual accuracy, Source citation, Appropriate confidence)
- 2 exemplos de resposta com análise (1 boa, 1 com problemas)

### Seção 3 — Análise de falhas
- Em quais tipos de pergunta o agente erra mais?
- Por que erra? (ex: pergunta ambígua, ferramenta errada, LLM confuso)
- Padrões observados

### Seção 4 — Conclusão
- O agente está pronto para produção? Sim/Não e por quê
- O que precisaria melhorar antes do deploy?
- Prioridades de melhoria

## Definição de feito
- [ ] `report_agent.md` commitado no repositório
- [ ] Todas as métricas numéricas presentes (accuracy geral, por ferramenta, scores G-Eval)
- [ ] Pelo menos 2 exemplos de falha analisados com explicação do motivo
- [ ] Seção de conclusão com recomendação clara (sim/não está pronto e por quê)

## Dependências
- `eval_tool_correctness.py` executado com sucesso
- `eval_geval.py` executado com sucesso (`geval_results.json` gerado)
- Servidor RAG rodando durante os testes
