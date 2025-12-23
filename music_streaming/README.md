# Churn Music Streaming — API de Machine Learning

## Visão geral

Este módulo do projeto **ChurnInsight** implementa um caso de uso de **previsão de churn em plataformas de streaming de música**.

O objetivo é identificar usuários com alta probabilidade de cancelamento, utilizando métricas de uso da plataforma, engajamento e comportamento de consumo.

## Estrutura do Projeto

```text
music_streaming/
├── app.py                 # Ponto de entrada da API (FastAPI)
├── config.py              # Configurações globais do projeto
├── Dockerfile             # Build da imagem Docker
├── README.md              # Documentação do dataset music_streaming
├── requirements.txt       # Dependências do projeto
│
├── data/                  # Dados utilizados no projeto
│
├── models/                # Modelos treinados e pipelines serializados
│
├── notebooks/             # Análises exploratórias (EDA)
│
├── schema/                # Esquema de entrada/saída da API
│
├── scripts/               # Scripts de treino e avaliação
│
├── services/              # Lógica principal da aplicação
│
└── utils/                 # Funções auxiliares
```

## FastAPI e como executar
A API foi desenvolvida utilizando FastAPI, permitindo integração simples com sistemas de back-end e serviços externos.

### Executando com Docker
A partir da pasta raiz do projeto (`churnInsight`), execute:
```bash
docker build --no-cache -t churn-music-api ./music_streaming
```

Em seguida:
```bash
docker run -p 8000:8000 churn-music-api
```

A API ficará disponível em:
```text
http://localhost:8000/docs
```

### Exemplo de Requisição via POST
Endpoint:
```text
POST /predict
```

Payload de exemplo:
```json
{
  "plano_pagamento": "anual",
  "chamados_suporte": "nenhum",
  "idade": 29,
  "horas_semanais": 18.4,
  "tempo_medio_sessao": 2.9,
  "taxa_skip_musica": 0.12,
  "taxa_musicas_unicas": 0.68,
  "notificacoes_clicadas": 42
}
```