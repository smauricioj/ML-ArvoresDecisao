# -*- coding: utf-8 -*-

# Autor: Sergio P
# Data: 10/09/2022

# ---------------------------------------------------------------
# IMPORTS

from math import log
from numpy import ndarray
from pandas import Series

# ---------------------------------------------------------------
# CLASSES

class Nodo:
    """Contém as informações da Arvore de Decisão"""

    def __init__(self):
        self.value = None # atributo para tomar decisões
        self.next = None # próximo nodo
        self.childs = None # galhos a partir do nodo

    def __repr__(self, nivel=0, bunch=None):
        # bunch acumula linhas para representação
        if bunch == None:
            bunch = list()
        # se nodo tem filhos, então é nó
        if self.childs:
            for child in self.childs:
                bunch.append(f"{'|   '*nivel}|--- {self.value} == {child.value}\n")
                nivel += 1
                child.next.__repr__(nivel, bunch)
                nivel -= 1
        # se não, é folha!
        else:
            bunch.append(f"{'|   '*nivel}|--- class: {self.value}\n")
        return bunch

class ID3:
    '''
    Classificador de Arvore de Decisão usando o algoritmo ID3-ish

    Baseado em um tutorial que achei na net.

    Funciona bem para atributos categóricos com pouco tamamnho vetorial.
    '''

    preditores: ndarray
    ''' Tabela com os preditores, exemplos '''

    rotulos: Series
    ''' Vetor com os rotulos, resultados dos exemplos'''

    nomes_atributos: list
    ''' Lista com os nomes de todos os atributos '''

    categorias: list
    ''' Lista com todos os rótulos únicos '''

    nCategorias: list
    ''' Lista com o número de cada rótulo '''

    arvore: Nodo
    ''' Raiz da Árvore de Decisão criada '''

    def __init__(self, df, nome_rotulo):
        self.preditores = df.drop(nome_rotulo, axis=1).values
        self.nomes_atributos = list(df.drop(nome_rotulo, axis=1).columns)
        self.rotulos = df[nome_rotulo]
        self.categorias = list(set(self.rotulos))
        self.nCategorias = [
            list(self.rotulos).count(x)
            for x in self.categorias
        ]
        self.arvore = None

    def __repr__(self):
        return self.arvore.__repr__()

    def run(self):
        self.arvore = self._id3_recv(
            [x for x in range(len(self.preditores))],
            [x for x in range(len(self.nomes_atributos))],
            self.arvore
        )

    # ---------------------------------------------------------------
    # Métodos internos
    # ---------------------------------------------------------------

    def _get_entropia(self, x_ids: list) -> float:
        """ Calcula a entropia.
        Parametros
            x_ids : list. Lista contendo os ID's dos exemplos

        Retorno
            entropia : float: Entropia do sistema
        """

        # rotulos ordenados pelos ID's dos exemplos
        rotulos = [self.rotulos[i] for i in x_ids]
        # contagem do número de exemplos em cada categoria
        conta_rotulos = [rotulos.count(x) for x in self.categorias]
        # calcula a entropia para cada categoria e soma
        return sum([
            -count / len(x_ids) * log(count / len(x_ids), 2)
            if count else 0
            for count in conta_rotulos
        ])

    def _get_ganho(self, x_ids: list, id_atributo: int) -> float:
        """ Calcula o ganho de informação para um dado atributo
            baseado na entropia
        Parametros
            x_ids : list. Lista contendo os ID's dos exemplos
            id_atributo: int. ID do atributo

        Retorno
            ganho : float. Ganho de informação para o dado atributo
        """

        # calcula a entropia inicial
        entropia_inicial = self._get_entropia(x_ids)
        # armazena todos os valores do atributo escolhido
        x_atributos = [self.preditores[x][id_atributo] for x in x_ids]
        # valores únicos
        val_atributos = list(set(x_atributos))
        # frequência de presença
        f_val_atributos = [x_atributos.count(x) for x in val_atributos]
        # ID's dos exemplos que contém os valores únicos do atributo
        id_val_atributos = [
            [
                x_ids[i]
                for i, x in enumerate(x_atributos)
                if x == y
            ]
            for y in val_atributos
        ]
        # calcula o ganho de informçao para o atributo escolhido
        ganho_atributo = sum([
                val_count / len(x_ids) * self._get_entropia(v_ids)
                for val_count, v_ids in zip(f_val_atributos, id_val_atributos)
        ])
        return entropia_inicial - ganho_atributo

    def _get_atributo_max_ganho(self, x_ids: list, atributos_ids: list):
        """ Encontra o atributo que maximiza o ganho de informação
        Parametros
            x_ids : list. Lista contendo os ID's dos exemplos
            atributos_ids : list. Lista contendo os ID's dos atributos

        Retorno
            (nome,i d) : (string, int). Nome e ID do atributo encontrado
        """

        # calcula o ganho de cada atributo
        entropias_atributos = [
            self._get_ganho(x_ids, id_atributo)
            for id_atributo in atributos_ids
        ]
        # encontra o maior ganho
        max_id = atributos_ids[entropias_atributos.index(
            max(entropias_atributos))]

        return self.nomes_atributos[max_id], max_id

    def _id3_recv(self, x_ids, atributos_ids, nodo):
        """ Algoritmo ID3. Chamado de forma recursiva até
            encontrar algum critério de parada.

        Parametros
            x_ids : list. Lista contendo os ID's dos exemplos no galho
            atributos_ids : list. Lista contendo os ID's dos atributos no galho
            nodo : objeto. Instância da classe Nodo

        Retorno
            nodo : objeto. Instância da classe Nodo
        """

        # inicializa a árvore, nodo raiz
        if not nodo:
            nodo = Nodo()

        # busca todos os rotulos no galho atual
        r_atributo = [self.rotulos[x] for x in x_ids]

        # se todos os rotulos são iguais, é folha
        if len(set(r_atributo)) == 1:
            nodo.value = self.rotulos[x_ids[0]]
            return nodo

        # Isso aqui não é ID3... mas é fácil fazer :D
        # se não existem mais atributos para computar,
        # retorna o nodo com o rótulo mais provável
        if len(atributos_ids) == 0:
            nodo.value = max(set(r_atributo), key=r_atributo.count)
            return nodo

        # se não...
        # escolhe o atributo que maximiza o ganho
        melhor_nome, melhor_id = self._get_atributo_max_ganho(
            x_ids, atributos_ids)

        # o nodo atual representa a decisão realizada
        nodo.value = melhor_nome

        # os filhos do nodo atual representam seus possíveis valores
        nodo.childs = list()

        # loop em todos os valores únicos do atributo escolhido
        for val in list(set([self.preditores[x][melhor_id] for x in x_ids])):
            # adiciona um galho do nodo para o valor
            child = Nodo()
            child.value = val
            nodo.childs.append(child)

            # partição dos exemplos cujo valor do atributo pertence ao galho
            child_x_ids = [
                x
                for x in x_ids
                if self.preditores[x][melhor_id] == val
            ]

            # se não houver exemplos com essa escolha (é estranho, mas)
            # então a folha seguinte é o atributo mais comum dos exemplos
            if not child_x_ids:
                child.next = max(set(r_atributo), key=r_atributo.count)
            else:
                # remove o atributo escolhido da lista de atributos
                # para próxima iteração
                if (atributos_ids) and (melhor_id in atributos_ids):
                    remover = atributos_ids.index(melhor_id)
                    removed = atributos_ids.pop(remover)

                # repete o algoritmo para a nova partição
                child.next = self._id3_recv(
                    child_x_ids,
                    atributos_ids,
                    child.next
                )

                # Reinsere o atributo removido para a próxima análise
                atributos_ids.insert(remover, removed)

        # retorna o nodo atual para continuar o método recursivo
        return nodo
