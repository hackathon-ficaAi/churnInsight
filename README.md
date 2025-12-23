
<h1 align="center">
  ChurnInsight — Churn Prediction API
</h1>

<div align="center">

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.125.0-009688)
![Docker](https://img.shields.io/badge/docker-ready-blue)
![Static Badge](https://img.shields.io/badge/status-em_desenvolvilmento-yellow)
![ML](https://img.shields.io/badge/machine%20learning-scikit--learn-orange)

</div>

## Previsão de Cancelamento de Clientes
Descrição do projeto

O desafio do ChurnInsight consiste em criar uma solução que preveja se um cliente está propenso a cancelar um serviço (churn).

O objetivo é que o time de Data Science desenvolva um modelo preditivo e que o time de Back-end construa uma API para disponibilizar essa previsão a outros sistemas, permitindo que o negócio aja antes que o cliente decida sair.

Exemplo: uma fintech quer saber, com base nos hábitos de uso e histórico de pagamento, quais clientes têm alta probabilidade de evasão. Com essa informação, o time de marketing pode oferecer serviços personalizados e o time de suporte pode agir preventivamente.

## Necessidade do cliente (explicação não técnica)

Toda empresa que vende por assinatura ou contrato recorrente sofre com cancelamentos. Manter clientes fiéis é mais barato do que conquistar novos.

O cliente (empresa) quer prever antecipadamente quem está prestes a cancelar, para poder agir e reter essas pessoas.

A solução esperada deve ajudar a:

identificar clientes com risco de churn (cancelamento);

priorizar ações de retenção (ofertas, contatos, bônus);

medir o impacto dessas ações ao longo do tempo.

## Validação de mercado

A previsão de churn é uma das aplicações mais comuns e valiosas da ciência de dados em negócios modernos.

Empresas de telecom, bancos digitais, academias, plataformas de streaming e provedores de software usam modelos de churn para:

reduzir perdas financeiras;

entender padrões de comportamento de clientes;

aumentar o tempo médio de relacionamento (lifetime value).

Mesmo modelos simples já trazem valor, pois ajudam as empresas a direcionar esforços onde há maior risco de perda.

## Tecnologias utilizadas

- **Python 3.11** — Linguagem principal
- **FastAPI** — API para disponibilização do modelo
- **Docker** — Containerização da aplicação
- **Scikit-learn** — Modelagem e pipelines de ML
- **Pandas / NumPy** — Manipulação e análise de dados
- **Matplotlib / Seaborn** — Gerar gráficos para visualização e análise de dados
- **Feature-engine** — Engenharia de features e pré-processamento

## Documentação do projeto

Este repositório é organizado de forma modular.  
Cada parte do projeto possui sua própria documentação detalhada.

- 📊 **Datasets e APIs**
  - [`churn_bancos/README.md`](./churn_bancos/README.md) — Caso de churn bancário
  - [`music_streaming/README.md`](./music_streaming/README.md) — Caso de churn em streaming

- 🤖 **Modelos de Machine Learning**
  - [`scripts/README.md`](./scripts/README.md) — Metodologia SEMMA, treino, validação e pipelines

- 🧪 **Análises exploratórias**
  - Documentadas diretamente nos notebooks em `notebooks/`

