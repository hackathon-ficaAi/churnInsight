#%%
import pandas as pd
import numpy as np
# Bibliotecas para gerar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# Encode e preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from feature_engine import discretisation
# Modelos de classificação 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
# Métricas
from sklearn import metrics
# serialização
import joblib
# %%
# Colocando os dados traduzidos em um DataFrame
url = 'https://raw.githubusercontent.com/hackathon-ficaAi/churnInsight/refs/heads/main/data/treino.csv'
df = pd.read_csv(url)
df.head()
# %%
# Juntando data de inscrição para inscrição Anual
df['data_inscricao'] = pd.to_datetime(df['data_inscricao'])
df['ano_inscricao'] = df['data_inscricao'].dt.to_period('A')
df['ano_inscricao'].value_counts().sort_index()
# %%
# Criando uma série temporal - com os anos de 2022 e 2021
ultimo_ano = df['ano_inscricao'].max()
oot = df[df['ano_inscricao'] >= (ultimo_ano - 1)].copy()
oot
# %%
# Criando uma base de treino sem os anos de 2022 e 2021
df_train = df[df['ano_inscricao'] < (ultimo_ano - 1)].copy()
df_train
# %%
# Separando target de features
target = 'churned'

categoricals = [
    'tipo_assinatura',
    'plano_pagamento',
    'metodo_pagamento',
    'chamados_suporte',
    'localizacao'
]

numericals = [
    col for col in df_train.columns
    if col not in categoricals + [target, 'data_inscricao',
                                   'ano_inscricao','id_cliente']
]

X,y= df_train[categoricals + numericals], df_train[target]
# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    random_state=42,
                                                    test_size=0.2,
                                                    stratify=y)
# %%
# EXPLORE
def criar_features(df:pd.DataFrame):
    df = df.copy()
    # Cria uma coluna que indica se usuarios clicou em pelo menos uma notificação ou não
    df['clicou_notificacao'] = (df['notificacoes_clicadas'] > 0).astype(int)
    # Aplicando log para tratar inflação de zeros
    df['log_notificacoes'] = np.log1p(df['notificacoes_clicadas'])
    # Cria uma coluna com a relação de taxas de musicas unicas por semana
    df['taxa_musicas_unicas'] = df['musicas_unicas_semana']/df['musicas_tocadas_semana']
    return df

# %%
# Análise de comportamento por churn com variáveis numéricas
X_train.isna().sum().sort_values(ascending=False)

df_analise = X_train.copy()
df_analise = criar_features(df_analise)
feat_num = list(set(df_analise.columns) - set(categoricals))

df_analise[target] = y_train
sumario = df_analise[feat_num + [target]].groupby(by=target).agg(["mean","median"]).T
# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0]/sumario[1]
sumario.sort_values(by=['diff_rel'], ascending=False)
# %%
X_train = criar_features(X_train)
arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train[feat_num],y_train)
# %%
plt.figure(dpi=700)
tree.plot_tree(arvore,
               feature_names=X_train[feat_num].columns,
               filled=True,
               class_names=[str(i) for i in arvore.classes_])
# %%
pd.Series(arvore.feature_importances_, index=X_train[feat_num].columns).sort_values(ascending=False)

#%%
# Análise de comportamento por churn com variáveis categóricas
df_analise = X_train[['tipo_assinatura','plano_pagamento','metodo_pagamento','chamados_suporte']].copy()
df_analise[target] = y_train

# Função que cria tabeças cruzadas por variável categórica
def resumo_categorica(var,df=df_analise, target=target):
    tabela = pd.crosstab(df[var], df[target], normalize='columns')
    
    resumo = tabela.copy()
    
    resumo['diff_abs'] = resumo[0] - resumo[1]
    resumo['diff_rel'] = resumo[0] / resumo[1]
    
    return resumo
# %%
resumo_categorica('tipo_assinatura')
# %%
resumo_categorica('plano_pagamento')
# %%
resumo_categorica('metodo_pagamento')
# %%
resumo_categorica('chamados_suporte')
# %%
from scipy.stats import chi2_contingency

def resumo_categorica_global(var, df=df_analise, target=target):
    # Crosstab absoluta (para chi²)
    tabela_abs = pd.crosstab(df[var], df[target])
    
    # Crosstab relativa (para diff)
    tabela_rel = pd.crosstab(df[var], df[target], normalize='columns')
    
    diff_abs = (tabela_rel[0] - tabela_rel[1]).abs().max()
    diff_rel = (tabela_rel[0] / tabela_rel[1]).apply(lambda x: max(x, 1/x)).max()
    
    chi2, p, _, _ = chi2_contingency(tabela_abs)
    
    return pd.Series({
        'max_diff_abs': diff_abs,
        'max_diff_rel': diff_rel,
        'chi2': chi2,
        'p_value': p
    })

# %%
resumo_final = (
    pd.DataFrame(
        {var: resumo_categorica_global(var) for var in df_analise.columns[:-1]}
    )
    .T
    .sort_values('chi2', ascending=False)
)

resumo_final
# %%
df_analise_ohe = pd.get_dummies(df_analise[df_analise.columns[:-1]], drop_first=True)

arvore_cat = tree.DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_leaf=50
)

arvore_cat.fit(df_analise_ohe, y_train)
# %%
imp_cat = (
    pd.Series(arvore_cat.feature_importances_, index=df_analise_ohe.columns)
    .sort_values(ascending=False)
)

imp_cat.head(15)
# %%
imp_cat_grouped = (
    imp_cat
    .groupby(lambda x: x.split('_')[0])
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

imp_cat_grouped['acumulada'] = imp_cat_grouped[0].cumsum()
imp_cat_grouped
# %%
