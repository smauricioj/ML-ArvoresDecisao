# -*- coding: utf-8 -*-

# Autor: Sergio P
# Data: 10/09/2022

# ---------------------------------------------------------------
# IMPORTS

from sklearn import tree
from pandas import get_dummies, DataFrame, Series

# ---------------------------------------------------------------
# CLASSES

class CART:
    '''
    Classe para calculo de Arvore de Decisão usando algoritmo CART
    
    A implementação desse algoritmo foi feita pelo módulo sklearn
    e, até o momento, não aceita atributos categóricos.
    
    Por isso, foi necessário transformar todas as colunas em valores
    numéricos (get_dummies)
    '''

    preditores: DataFrame
    ''' Tabela com os preditores, exemplos '''

    rotulos: Series
    ''' Vetor com os rotulos, resultados dos exemplos'''

    nomes_atributos: list
    ''' Lista com os nomes de todos os atributos '''

    arvore: tree.DecisionTreeClassifier
    ''' Classe de sklearn que faz os calculos pra nós :) '''

    def __init__(self, df, nome_rotulo):
        self.preditores = get_dummies(df.drop(nome_rotulo, axis=1))
        self.rotulos = df[nome_rotulo]
        self.nomes_atributos = list(self.preditores.columns)
        self.arvore = tree.DecisionTreeClassifier(criterion='entropy')

    def __repr__(self):
        return tree.export_text(
                    self.arvore,
                    feature_names=list(self.nomes_atributos)
                ).replace('<= 0.50', '== False').\
                  replace('>  0.50', '== True')

    def run(self):
        self.arvore = self.arvore.fit(self.preditores, self.rotulos)