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
# %%
url = 'https://raw.githubusercontent.com/hackathon-ficaAi/churnInsight/refs/heads/main/data/treino.csv'
df = pd.read_csv(url)
df.head()
# %%
categoricals = ['tipo_assinatura','plano_pagamento','metodo_pagamento','chamados_suporte','data_inscricao','localizacao']
numericals = list(set(df.columns) - set(categoricals))
# %%
df.info()
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
# Criando uma coluna onde se o cliente clicou em pelo menos 1 notificação retorna 1, caso contrario 0
df_train['clicou_notificacao'] = (df_train['notificacoes_clicadas'] > 0).astype(int)
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
num_features = [
    'taxa_skip_musica',
    'taxa_musicas_unicas',
    'notificacoes_clicadas',
    'horas_semanais',
    'tempo_medio_sessao',
    'idade'
]

cat_features = [
    'plano_pagamento',
    'chamados_suporte',
    'tem_notificacao'  # nova flag
]
best_features = num_features + cat_features

# %%
# Criando Pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), cat_features),
        ('num', MinMaxScaler(), num_features)
    ],
    remainder='drop'
)
# %%
from feature_engine import encoding
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.preprocessing import StandardScaler

# %%
# Agrupa categorias raras em 'Rare', reduzindo dimensionalidade
rare_encoder = RareLabelEncoder(
    variables=['plano_pagamento', 'chamados_suporte'],
    tol=0.05,
    n_categories=10
)
# %%
# transformação que reduz assimetria dos dados, deixando uma distribuição mais simétrica
yeo_johnson = YeoJohnsonTransformer(variables=num_features)

# %%
# substitui outliers por percentis
winsorizer = Winsorizer(
    variables=num_features,
    capping_method='quantiles',
    tail='both',
    fold=0.05
)

onehot = OneHotEncoder(
    variables=cat_features,
    ignore_format=True,
)
# %%
tree_discretization = discretisation.DecisionTreeDiscretiser(variables=num_features,
                                                             cv=3,
                                                             regression=False,
                                                             bin_output='bin_number',
                                                             )
# %%
# Pipeline Regressão Logística
pipeline_reg = Pipeline(steps=[
#    ('preprocessor', preprocessor),
#    ('discretização', tree_discretization),
    ('rare_labels', rare_encoder),
    ('yeo_johnson', yeo_johnson),
    ('winsorizer', winsorizer),    
    ('OneHot', onehot),
    ('scaler', StandardScaler()),
    ('modelo', linear_model.LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])
# %%
# Pipeline Random Forest
pipeline_rf = Pipeline([
#    ('preprocessor', preprocessor),
#    ('discretização', tree_discretization),
    ('rare_labels', rare_encoder),
    ('yeo_johnson', yeo_johnson),
    ('winsorizer', winsorizer),    
    ('OneHot', onehot),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# %%
# Treinamento - Regressão Logistica
pipeline_reg.fit(X_train[best_features], y_train)
# %%
# Predições
y_train_predict_reg = pipeline_reg.predict(X_train[best_features])
y_train_proba_reg = pipeline_reg.predict_proba(X_train[best_features])[:,1]

y_test_predict_reg = pipeline_reg.predict(X_test[best_features])
y_test_proba_reg = pipeline_reg.predict_proba(X_test[best_features])[:,1]
# %%
# Métricas
acc_train_reg = metrics.accuracy_score(y_train, y_train_predict_reg)
auc_train_reg = metrics.roc_auc_score(y_train, y_train_proba_reg)

acc_test_reg = metrics.accuracy_score(y_test, y_test_predict_reg)
auc_test_reg = metrics.roc_auc_score(y_test, y_test_proba_reg)
# %%
print(f"Acurácia Treino: {acc_train_reg:.4f}")
print(f"AUC Treino: {auc_train_reg:.4f}")
print(f"Predições positivas treino: {y_train_predict_reg.sum()} de {len(y_train)} ({y_train_predict_reg.mean():.2%})")
print(f"Taxa real churn treino: {y_train.mean():.2%}")
# %%
print(f"Acurácia Teste: {acc_test_reg:.4f}")
print(f"AUC Teste: {auc_test_reg:.4f}")
print(f"Predições positivas teste: {y_test_predict_reg.sum()} de {len(y_test)} ({y_test_predict_reg.mean():.2%})")
print(f"Taxa real churn teste: {y_test.mean():.2%}")

# %%
# Treinamento - Random Forest
pipeline_rf.fit(X_train[best_features], y_train)

# Predições
y_train_predict_rf = pipeline_rf.predict(X_train[best_features])
y_train_proba_rf = pipeline_rf.predict_proba(X_train[best_features])[:,1]

y_test_predict_rf = pipeline_rf.predict(X_test[best_features])
y_test_proba_rf = pipeline_rf.predict_proba(X_test[best_features])[:,1]
# %%
# Métricas
acc_train_rf = metrics.accuracy_score(y_train, y_train_predict_rf)
auc_train_rf = metrics.roc_auc_score(y_train, y_train_proba_rf)

acc_test_rf = metrics.accuracy_score(y_test, y_test_predict_rf)
auc_test_rf = metrics.roc_auc_score(y_test, y_test_proba_rf)
# %%
print(f"Acurácia Treino: {acc_train_rf:.4f}")
print(f"AUC Treino: {auc_train_rf:.4f}")
print(f"Predições positivas treino: {y_train_predict_rf.sum()} de {len(y_train)} ({y_train_predict_rf.mean():.2%})")
print(f"Taxa real churn treino: {y_train.mean():.2%}")
# %%
print(f"Acurácia Teste: {acc_test_rf:.4f}")
print(f"AUC Teste: {auc_test_rf:.4f}")
print(f"Predições positivas teste: {y_test_predict_rf.sum()} de {len(y_test)} ({y_test_predict_rf.mean():.2%})")
print(f"Taxa real churn teste: {y_test.mean():.2%}")

# %%
# Export
joblib.dump(pipeline_reg, "pipeline_churn_reg.joblib")
print("\n✓ Pipeline Regressão Logística salvo em: pipeline_churn_reg.joblib")

joblib.dump(pipeline_rf, "pipeline_churn_rf.joblib")
print("✓ Pipeline Random Forest salvo em: pipeline_churn_rf.joblib")
# %%
