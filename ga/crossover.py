"""
Operadores de cruzamento para o TSP com representação por caminho.

Este módulo implementa três operadores de cruzamento conforme descrito no artigo:
1. PMX (Partially Mapped Crossover) - Goldberg e Lingle, 1985
2. OX (Order Crossover) - Davis, 1985
3. CX2 (Modified Cycle Crossover) - Proposto por Hussain et al., 2017

Todos os operadores funcionam com representação por caminho e produzem percursos válidos.
"""

import numpy as np
from typing import Tuple, List
from .chromosome import Chromosome


class PMX:
    """
    Cruzamento Parcialmente Mapeado (PMX).

    Após escolher dois pontos de corte aleatórios nos pais para construir os filhos,
    a porção entre os pontos de corte da string de um pai é mapeada na
    string do outro pai e as informações restantes são trocadas.

    Referência: Goldberg e Lingle, 1985
    """

    @staticmethod
    def crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Realiza o cruzamento PMX em dois pais.

        Args:
            parent1: Primeiro cromossomo pai.
            parent2: Segundo cromossomo pai.

        Returns:
            Tupla de dois cromossomos filhos.
        """
        n = len(parent1)
        p1 = parent1.genes.copy()
        p2 = parent2.genes.copy()

        # Inicializa filhos com -1 (vazio)
        o1 = np.full(n, -1, dtype=np.int32)
        o2 = np.full(n, -1, dtype=np.int32)

        # Seleciona dois pontos de corte aleatórios
        cut1, cut2 = sorted(np.random.choice(n, 2, replace=False))

        # Copia o segmento entre os pontos de corte
        o1[cut1:cut2+1] = p2[cut1:cut2+1]
        o2[cut1:cut2+1] = p1[cut1:cut2+1]

        # Cria mapeamentos dos segmentos trocados
        mapping1 = {}  # Mapeia valores no segmento de o1 para valores no segmento de p1
        mapping2 = {}  # Mapeia valores no segmento de o2 para valores no segmento de p2

        for i in range(cut1, cut2 + 1):
            mapping1[p2[i]] = p1[i]
            mapping2[p1[i]] = p2[i]

        # Preenche posições restantes para o filho 1
        for i in list(range(0, cut1)) + list(range(cut2 + 1, n)):
            val = p1[i]
            while val in o1[cut1:cut2+1]:
                val = mapping1[val]
            o1[i] = val

        # Preenche posições restantes para o filho 2
        for i in list(range(0, cut1)) + list(range(cut2 + 1, n)):
            val = p2[i]
            while val in o2[cut1:cut2+1]:
                val = mapping2[val]
            o2[i] = val

        return Chromosome(genes=o1), Chromosome(genes=o2)


class OX:
    """
    Cruzamento de Ordem (OX).

    Constrói filhos escolhendo um sub-percurso de um pai e preservando
    a ordem relativa dos bits do outro pai.

    Referência: Davis, 1985
    """

    @staticmethod
    def crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Realiza o cruzamento OX em dois pais.

        Args:
            parent1: Primeiro cromossomo pai.
            parent2: Segundo cromossomo pai.

        Returns:
            Tupla de dois cromossomos filhos.
        """
        n = len(parent1)
        p1 = parent1.genes.copy()
        p2 = parent2.genes.copy()

        # Inicializa filhos com -1 (vazio)
        o1 = np.full(n, -1, dtype=np.int32)
        o2 = np.full(n, -1, dtype=np.int32)

        # Seleciona dois pontos de corte aleatórios
        cut1, cut2 = sorted(np.random.choice(n, 2, replace=False))

        # Copia o segmento entre os pontos de corte dos respectivos pais
        o1[cut1:cut2+1] = p1[cut1:cut2+1]
        o2[cut1:cut2+1] = p2[cut1:cut2+1]

        # Obtém valores nos segmentos copiados
        segment1 = set(o1[cut1:cut2+1])
        segment2 = set(o2[cut1:cut2+1])

        # Preenche filho 1 com valores do pai 2 (em ordem)
        # Começando de cut2+1, com wraparound
        p2_order = list(p2[cut2+1:]) + list(p2[:cut2+1])
        p2_filtered = [x for x in p2_order if x not in segment1]

        pos = (cut2 + 1) % n
        for val in p2_filtered:
            o1[pos] = val
            pos = (pos + 1) % n

        # Preenche filho 2 com valores do pai 1 (em ordem)
        p1_order = list(p1[cut2+1:]) + list(p1[:cut2+1])
        p1_filtered = [x for x in p1_order if x not in segment2]

        pos = (cut2 + 1) % n
        for val in p1_filtered:
            o2[pos] = val
            pos = (pos + 1) % n

        return Chromosome(genes=o1), Chromosome(genes=o2)


class CX2:
    """
    Cruzamento de Ciclo Modificado (CX2).

    Este é o operador de cruzamento proposto no artigo.
    Funciona de forma similar ao CX mas gera ambos os filhos simultaneamente
    usando ciclos, com um movimento para o primeiro filho e dois movimentos para o segundo filho.

    Referência: Hussain et al., 2017

    Passos do Algoritmo:
    Passo 1. Escolher dois pais para o cruzamento.
    Passo 2. Selecionar o 1º bit do segundo pai como 1º bit do primeiro filho.
    Passo 3. O bit selecionado no Passo 2 seria encontrado no primeiro pai e escolher
            o bit na mesma posição exata que está no segundo pai e esse bit
            seria encontrado novamente no primeiro pai e, finalmente, o bit na
            mesma posição exata que está no segundo pai será selecionado para o 1º bit
            do segundo filho.
    Passo 4. O bit selecionado no Passo 3 seria encontrado no primeiro pai e escolher
            o bit na mesma posição exata que está no segundo pai como o próximo bit
            para o primeiro filho. (Nota: para o primeiro filho, escolhemos bits
            com apenas um movimento e dois movimentos para os bits do segundo filho.)
    Passo 5. Repetir os Passos 3 e 4 até que o 1º bit do primeiro pai não venha no
            segundo filho (completar um ciclo) e o processo pode ser encerrado.
    Passo 6. Se alguns bits sobrarem, então os mesmos bits no primeiro pai e no
            segundo filho até agora e vice-versa são excluídos de ambos os pais.
            Para os bits restantes repetir os Passos 2, 3 e 4 para completar o processo.
    """

    @staticmethod
    def crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Realiza o cruzamento CX2 seguindo o algoritmo do artigo exatamente.

        Seguindo o exemplo do artigo:
        P1 = (3 4 8 2 7 1 6 5)
        P2 = (4 2 5 1 6 8 3 7)
        Resultado:
        O1 = (4 8 6 2 5 3 1 7)
        O2 = (1 7 4 8 6 2 5 3)
        """
        n = len(parent1)
        p1 = parent1.genes.tolist()
        p2 = parent2.genes.tolist()

        o1 = []
        o2 = []

        # Cria lookup de posição para P1: onde está cada valor?
        pos_in_p1 = {val: idx for idx, val in enumerate(p1)}

        used_in_o1 = set()
        used_in_o2 = set()

        # Rastreia o último valor adicionado a O2 (necessário para o Passo 4)
        last_o2_val = None

        while len(o1) < n:
            # Passo 2: Selecionar o primeiro bit não utilizado de P2 para O1
            start_val = None
            for v in p2:
                if v not in used_in_o1:
                    start_val = v
                    break

            if start_val is None:
                break

            # Adiciona start_val a O1
            o1.append(start_val)
            used_in_o1.add(start_val)

            # Passo 3: Dois movimentos para encontrar o primeiro bit para O2
            # Movimento 1: Encontrar start_val em P1, obter o valor da posição correspondente em P2
            pos1 = pos_in_p1[start_val]
            val1 = p2[pos1]

            # Movimento 2: Encontrar val1 em P1, obter o valor da posição correspondente em P2
            pos2 = pos_in_p1[val1]
            val2 = p2[pos2]

            # Adiciona val2 a O2
            if val2 not in used_in_o2:
                o2.append(val2)
                used_in_o2.add(val2)
                last_o2_val = val2

            # Continua o ciclo
            while len(o1) < n:
                # Passo 4: Um movimento para O1 (a partir do último valor de O2)
                if last_o2_val is None:
                    break

                pos_last = pos_in_p1[last_o2_val]
                next_o1_val = p2[pos_last]

                if next_o1_val in used_in_o1:
                    # Ciclo completo para esta rodada
                    break

                o1.append(next_o1_val)
                used_in_o1.add(next_o1_val)

                # Passo 4 continuação: Dois movimentos para O2
                # Movimento 1
                pos_next = pos_in_p1[next_o1_val]
                temp_val = p2[pos_next]

                # Movimento 2
                pos_temp = pos_in_p1[temp_val]
                next_o2_val = p2[pos_temp]

                if next_o2_val not in used_in_o2 and len(o2) < n:
                    o2.append(next_o2_val)
                    used_in_o2.add(next_o2_val)
                    last_o2_val = next_o2_val
                else:
                    # Não pode adicionar a O2, mas continua com O1
                    # Encontra um novo last_o2_val do que já foi adicionado
                    if o2:
                        last_o2_val = o2[-1]

                # Passo 5: Verifica se o ciclo está completo
                if p1[0] in used_in_o2:
                    break

        # Passo 6: Preenche posições restantes com valores não utilizados
        for v in p2:
            if v not in used_in_o1 and len(o1) < n:
                o1.append(v)
                used_in_o1.add(v)

        for v in p1:
            if v not in used_in_o2 and len(o2) < n:
                o2.append(v)
                used_in_o2.add(v)

        return Chromosome(genes=np.array(o1, dtype=np.int32)), \
               Chromosome(genes=np.array(o2, dtype=np.int32))

    @staticmethod
    def crossover_simple(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        CX2 alternativo mais simples que segue diretamente a estrutura de ciclos.
        """
        return CX2.crossover(parent1, parent2)


class CrossoverFactory:
    """
    Classe fábrica para criar operadores de cruzamento por nome.
    """

    OPERATORS = {
        'PMX': PMX.crossover,
        'OX': OX.crossover,
        'CX2': CX2.crossover,
    }

    @staticmethod
    def get_operator(name: str):
        """
        Obtém um operador de cruzamento pelo nome.

        Args:
            name: Nome do operador ('PMX', 'OX', 'CX2').

        Returns:
            Função de cruzamento.
        """
        name = name.upper()
        if name not in CrossoverFactory.OPERATORS:
            raise ValueError(f"Unknown crossover operator: {name}. "
                           f"Available: {list(CrossoverFactory.OPERATORS.keys())}")
        return CrossoverFactory.OPERATORS[name]
