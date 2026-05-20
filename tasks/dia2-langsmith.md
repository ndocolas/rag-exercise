# Dia 2 — LangSmith: Observabilidade do Agente

## Objetivo
Instrumentar o agente LangGraph com rastreamento via LangSmith para ver
inputs, outputs, ferramentas usadas e latência de cada execução.

## Pré-requisitos
- Dia 1 concluído (agent.py funcionando)
- Conta criada em smith.langchain.com

## Setup

### Passo 1 — Criar conta e projeto
- Acessar smith.langchain.com
- Criar projeto chamado `rag-exercise`
- Copiar a API key gerada

### Passo 2 — Configurar variáveis de ambiente
Adicionar ao `.env` local (não commitar o valor real):
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=sua-chave-aqui
LANGCHAIN_PROJECT=rag-exercise
```
Já adicionado ao `.env.example` (sem o valor real).

### Passo 3 — Verificar instrumentação
O LangGraph instrumenta automaticamente quando as vars estão definidas.
**Nenhuma mudança de código necessária.**

### Passo 4 — Gerar 10+ traces
Rodar o agente pelo menos 10 vezes com perguntas variadas:
```bash
python agent.py --question "What are the risks of bond investing?"
python agent.py --question "What is 15% of 3000?"
python agent.py --question "How many years ago was 2010?"
python agent.py --question "What is a mutual fund?"
python agent.py --question "Calculate 8.5% of 42000"
python agent.py --question "What is the P/E ratio?"
python agent.py --question "How many days since 2020?"
python agent.py --question "What is compound interest?"
python agent.py --question "What is 1000 divided by 0.04?"
python agent.py --question "What year is it currently?"
```

Verificar no dashboard LangSmith que cada trace mostra:
- Input da pergunta
- Ferramenta chamada + argumentos usados
- Output da ferramenta
- Resposta final do agente
- Latência de cada nó

## Entregável Confluence — 'Setup do LangSmith para observabilidade de agentes'

Seções obrigatórias:
1. Criação da conta e do projeto no smith.langchain.com
2. Variáveis de ambiente necessárias e onde configurar
3. Como o LangGraph instrumenta automaticamente (ou como usar @traceable se necessário)
4. Como interpretar um trace: screenshot real do projeto + explicação de cada campo

O guia deve ser reproduzível por outra pessoa da equipe sem ajuda adicional.

## Definição de feito
- [ ] 10+ traces visíveis no LangSmith com inputs, tool calls e outputs
- [ ] Cada trace mostra: entrada, ferramentas com args, saída, latência por nó
- [ ] Screenshot do dashboard LangSmith capturado
- [ ] Guia publicado no Confluence (ou draft compartilhável)
- [ ] Screenshot incluído no guia

## Dependências
- agent.py funcionando (Dia 1)
- Servidor RAG rodando (`uv run uvicorn rag_eval.main:app --reload`)
- Conta LangSmith criada
