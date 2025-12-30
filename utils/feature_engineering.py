# utils/feature_engineering.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    Gera:
      - pais_enc, genero_enc (internos)
      - alemao_mulher
      - idade_x_produtos
    """

    def __init__(self, pais_col='pais', genero_col='genero'):
        self.pais_col = pais_col
        self.genero_col = genero_col
        self.le_pais = LabelEncoder()
        self.le_genero = LabelEncoder()

    def fit(self, X, y=None):
        X = X.copy()
        # fit apenas nas colunas originais (assume pd.DataFrame)
        self.le_pais.fit(X[self.pais_col].astype(str))
        self.le_genero.fit(X[self.genero_col].astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        # garante strings (evita erros)
        X[self.pais_col] = X[self.pais_col].astype(str)
        X[self.genero_col] = X[self.genero_col].astype(str)

        # encodings (mantemos as originais também)
        X['pais_enc'] = self.le_pais.transform(X[self.pais_col])
        X['genero_enc'] = self.le_genero.transform(X[self.genero_col])

        # features derivadas
        # atenção: o valor exato 'alemanha' e 'feminino' deve bater com seu dataset (case)
        try:
            alemanha_code = self.le_pais.transform(['alemanha'])[0]
        except ValueError:
            alemanha_code = None

        try:
            feminino_code = self.le_genero.transform(['feminino'])[0]
        except ValueError:
            feminino_code = None

        if alemanha_code is not None and feminino_code is not None:
            X['alemao_mulher'] = (
                (X['pais_enc'] == alemanha_code) &
                (X['genero_enc'] == feminino_code)
            ).astype(int)
        else:
            X['alemao_mulher'] = 0

        # membro ativo binário (preserva valor original)
        X['membro_ativo_bin'] = X['membro_ativo'].astype(int)

        # interação
        X['idade_x_produtos'] = X['idade'] * X['num_produtos']

        return X