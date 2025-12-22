# ChurnInsight — Previsão de Cancelamento de Clientes
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

## Estrutura do Projeto
```text
churn_bancos/
├── .gitignore
├── app.py                 # Ponto de entrada da API (FastAPI)
├── config.py              # Configurações globais do projeto
├── Dockerfile             # Build da imagem Docker da aplicação
├── README.md              # Documentação do projeto
├── requirements.txt       # Dependências do projeto
│
├── data/                  # Dados utilizados no projeto
│   └── churn_bancos.csv
│
├── models/                # Modelos treinados e pipelines serializados
│   ├── pipeline_churn_rf.joblib
│   └── pipeline_churn_reg.joblib
│
├── notebooks/             # Análises exploratórias (EDA)
│   └── eda_churn.ipynb
│
├── schemas/               # Esquemas de entrada/saída da API (Pydantic)
│   └── churn_schema.py
│
├── scripts/               # Scripts executáveis (CLI / batch)
│   ├── test/
│   │   └── test_pipeline.py
│   └── train/
│       └── train_model.py
│
├── services/              # Lógica principal da aplicação
│   ├── predict.py         # Inferência com modelo treinado
│   └── train.py           # Treino e validação reutilizáveis
│
└── utils/                 # Funções auxiliares (helpers, métricas, logs)
    ├── io.py
    ├── logger.py
    └── metrics.py
```
## FastAPI e como executar

Para fazer a comunicação com o back-end no projeto foi utilizado FastAPI, um moderno e rápido (alta performance) framework web para construção de APIs com Python, baseado nos type hints padrões do Python.

### Executando a API com Docker
Por enquanto o projeto só possui API disponível na pasta `music_streaming`, então utilize essa pasta como destino,
porém suportara múltiplos datasets (ex: music_streaming, churn_bancos),
cada um com sua própria API e modelo treinado.

Cada pasta contém:

- um Dockerfile

- um app.py

- um modelo treinado em models/

Estando na pasta raiz `churnInsight`, escolha a pasta do dataset desejado como contexto do build.

*Exemplo*: `music_streaming`
```bash
docker build --no-cache -t churn-ml-api ./music_streaming
```

Após isso basta executar o container com:

```bash
docker run -p 8000:8000 churn-ml-api
```
A API estará disponível em:

```text
http://localhost:8000/docs
```
### Exemplo de Requisição via POST
Em `POST /predict` pode fazer uma requisição com a estrutura exemplo abaixo:
```json
{
  "plano_pagamento": "anual",
  "chamados_suporte": "nenhum",
  "idade": 71,
  "horas_semanais": 23.510865,
  "tempo_medio_sessao": 3.436433,
  "taxa_skip_musica": 0.01,
  "taxa_musicas_unicas": 0.77,
  "notificacoes_clicadas": 129
} 
```