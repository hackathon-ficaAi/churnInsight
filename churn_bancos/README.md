# Churn Bancos — API de Machine Learning

## Visão geral

Este módulo do projeto **ChurnInsight** implementa um caso de uso de **previsão de churn no setor bancário**.

O objetivo é prever se um cliente irá cancelar o relacionamento com o banco (`churned`), utilizando dados demográficos e comportamentais, disponibilizando essa previsão via uma **API FastAPI**.

---

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
├── notebooks/             # Análises exploratórias (EDA) - Separado individualmente 
│   ├── alexandre_imai/
│   │   └── notebook.ipynb
│   ├── kevin/
│   │   └── notebook.ipynb
│   ├── marcos/
│   │   └── notebook.ipynb
│   ├── michele/
│   │   └── notebook.ipynb
│   └── ruthe/
│       └── notebook.ipynb
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
O projeto suporta múltiplos datasets (ex: music_streaming, churn_bancos),
cada um com sua própria API e modelo treinado.

Cada pasta contém:

- um Dockerfile

- um app.py

- um modelo treinado em models/

Estando na pasta raiz `churnInsight`, escolha a pasta do dataset desejado como contexto do build.

*Exemplo*: `churn_bancos`
```bash
docker build --no-cache -t churn-ml-api ./churn_bancos
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
Endpoint:
```text
POST /predict
```

Payload de exemplo:
```json
{
  "pais": "France",
  "genero": "Male",
  "idade": 45,
  "num_produtos": 2,
  "membro_ativo": 1,
  "saldo": 75432.50,
  "salario_estimado": 62000.00
}
```