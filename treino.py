#%%
import pandas as pd
import numpy as np
# Bibliotecas para gerar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# Encode e preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Modelos de classificação 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
# Métricas
from sklearn import metrics
# %%
# %%
df = pd.read_csv("data/treino.csv")
df.head()
# %%
categoricals = ['tipo_assinatura','plano_pagamento','metodo_pagamento','chamados_suporte','data_inscricao','localizacao']
numericals = list(set(df.columns) - set(categoricals))
# %%

# %%
df.info()
#%%

# %%
df['data_inscricao'].value_counts().sort_index()
# %%
df['data_inscricao'] = pd.to_datetime(df['data_inscricao'])
df['mes_ano_inscricao'] = df['data_inscricao'].dt.to_period('M')
df['mes_ano_inscricao'].value_counts().sort_index()
#%%
# Criando uma variável nova de taxa de músicas únicas
df.isna().sum().sort_values(ascending=False)
df['taxa_musicas_unicas'] = df['musicas_unicas_semana']/df['musicas_tocadas_semana']
df.head()
# %%
ultimo_mes = df['mes_ano_inscricao'].max()
oot = df[df['mes_ano_inscricao'] >= (ultimo_mes - 3)].copy()
oot
# %%
df_train = df[df['mes_ano_inscricao'] < (ultimo_mes - 3)].copy()
df_train
# %%
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
                                   'mes_ano_inscricao','id_cliente',
                                   'musicas_unicas_semana','musicas_tocadas_semana']
]

X,y= df_train[categoricals + numericals], df_train[target]

# %%

# %%

# %%

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    random_state=42,
                                                    test_size=0.2,
                                                    stratify=y)
y_train.mean(), y_test.mean()
# %%
X_train.isna().sum().sort_values(ascending=False)
# análise de variáveis 

df_analise = X_train[numericals].copy()
df_analise[target] = y_train

#%%
df_analise
#%%
sumario = df_analise.groupby(by=target).agg(["mean","median"]).T
sumario


# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0]/sumario[1]
sumario.sort_values(by=['diff_rel'],ascending=False)


# %%
arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
arvore.fit(X_train[numericals],y_train)

plt.figure(dpi=800,figsize=(12,12))
tree.plot_tree(arvore, 
               feature_names=X_train[numericals].columns, 
               filled=True, 
               class_names=[str(i) for i in arvore.classes_])

# %%
features_importances = (pd.Series(arvore.feature_importances_, index=numericals)
                        .sort_values(ascending=False)
                        .reset_index())
features_importances['acumulada'] = features_importances[0].cumsum()
# %%
features_importances




# %%
categoricas = X_train.select_dtypes(include='object').columns

df_cat = X_train[categoricas].copy()
df_cat[target] = y_train
df_cat
# %%
pd.crosstab(df_cat['tipo_assinatura'], df_cat[target],normalize='columns')

# %%
def resumo_categorica(var,df=df_cat, target=target):
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


# %%
from scipy.stats import chi2_contingency

def resumo_categorica_global(var, df=df_cat, target=target):
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
        {var: resumo_categorica_global(var) for var in categoricas}
    )
    .T
    .sort_values('chi2', ascending=False)
)
# %%
resumo_final
X_cat_ohe = pd.get_dummies(X_train[categoricas], drop_first=True)

arvore_cat = tree.DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_leaf=50
)

arvore_cat.fit(X_cat_ohe, y_train)

# %%
imp_cat = (
    pd.Series(arvore_cat.feature_importances_, index=X_cat_ohe.columns)
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
num_features = ['taxa_skip_musica','taxa_musicas_unicas',
                 'notificacoes_clicadas','horas_semanais',
                 'tempo_medio_sessao','idade']
cat_features = ['plano_pagamento','chamados_suporte']
best_features = num_features + cat_features

# %%
## One Hot -- categóricas para numéricas
colunas = X_train[best_features].columns
one_hot = make_column_transformer((
    OneHotEncoder(drop = 'if_binary'),
    cat_features
    ),
    remainder = 'passthrough',
    sparse_threshold=0)
X_train = one_hot.fit_transform(X_train[best_features])

X_train= pd.DataFrame(X_train, columns=one_hot.get_feature_names_out(colunas))
X_train.head()

# %%
# %%
normalizacao = MinMaxScaler()
X_train = normalizacao.fit_transform(X_train)
# %%
pd.DataFrame(X_train)
# %%
## Fazendo na base de teste
X_test = one_hot.transform(X_test[best_features])
X_test = pd.DataFrame(X_test, columns=one_hot.get_feature_names_out(colunas))
X_test.head()
# %%
X_test = normalizacao.transform(X_test)
# %%
reg = linear_model.LogisticRegression(penalty=None,
                                      random_state=42,
                                      max_iter=1000000)
reg.fit(X_train,y_train)

# %%
rf = RandomForestClassifier(random_state=42,
                            n_jobs=2)
rf.fit(X_train,y_train)
# %%
y_train_predict = reg.predict(X_train)
y_train_proba = reg.predict_proba(X_train)[:,1]

acc_train = metrics.accuracy_score(y_train,y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print("Acurácia Treino",acc_train)
print("AUC Treino",auc_train)
# %%
y_train_predict = rf.predict(X_train)
y_train_proba = rf.predict_proba(X_train)[:,1]

acc_train = metrics.accuracy_score(y_train,y_train_predict)
auc_train = metrics.roc_auc_score(y_train, y_train_proba)

print("Acurácia Treino",acc_train)
print("AUC Treino",auc_train)
# %%
y_test_predict = rf.predict(X_test)
y_test_proba = rf.predict_proba(X_test)[:,1]

acc_test = metrics.accuracy_score(y_test,y_test_predict)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)

print("Acurácia Treino",acc_test)
print("AUC Treino",auc_test)
# %%
y_test_predict.sum()
y_train_predict.sum()
# %%
df.head()
# %%
