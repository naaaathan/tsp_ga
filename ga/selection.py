"""
Operadores de seleção para o Algoritmo Genético.

O artigo utiliza seleção proporcional (roleta) para escolher P/2 pais
da população atual.
"""

import numpy as np
from typing import List, Tuple
from .chromosome import Chromosome, Population


class Selection:
    """
    Operadores de seleção para escolher pais para o cruzamento.
    """

    @staticmethod
    def roulette_wheel(population: Population, num_parents: int) -> List[Chromosome]:
        """
        Seleção por roleta (seleção proporcional).

        Para problemas de minimização, invertemos o fitness para que
        cromossomos com menor fitness tenham maior probabilidade de seleção.

        Args:
            population: A população de onde selecionar.
            num_parents: Número de pais a selecionar.

        Returns:
            Lista de cromossomos pais selecionados.
        """
        # Obtém valores de fitness (estamos minimizando, então inverte para seleção)
        fitness_values = np.array([c.fitness for c in population.chromosomes])

        # Trata minimização: usa fitness inverso
        # Adiciona pequeno epsilon para evitar divisão por zero
        epsilon = 1e-10
        inverse_fitness = 1.0 / (fitness_values + epsilon)

        # Calcula probabilidades de seleção
        total_inverse = np.sum(inverse_fitness)
        probabilities = inverse_fitness / total_inverse

        # Seleciona pais usando probabilidades
        indices = np.random.choice(
            len(population),
            size=num_parents,
            replace=True,
            p=probabilities
        )

        return [population[i].copy() for i in indices]

    @staticmethod
    def tournament(population: Population, num_parents: int,
                   tournament_size: int = 3) -> List[Chromosome]:
        """
        Seleção por torneio.

        Seleciona aleatoriamente tournament_size indivíduos e escolhe o melhor.
        Repete até que num_parents sejam selecionados.

        Args:
            population: A população de onde selecionar.
            num_parents: Número de pais a selecionar.
            tournament_size: Número de indivíduos em cada torneio.

        Returns:
            Lista de cromossomos pais selecionados.
        """
        parents = []

        for _ in range(num_parents):
            # Seleciona aleatoriamente tournament_size indivíduos
            tournament_indices = np.random.choice(
                len(population),
                size=tournament_size,
                replace=False
            )

            # Obtém o melhor (menor fitness) do torneio
            tournament = [population[i] for i in tournament_indices]
            winner = min(tournament, key=lambda c: c.fitness)
            parents.append(winner.copy())

        return parents

    @staticmethod
    def rank_based(population: Population, num_parents: int) -> List[Chromosome]:
        """
        Seleção baseada em ranking.

        A probabilidade de seleção é baseada no ranking ao invés dos valores brutos de fitness.
        Melhor para quando os valores de fitness têm alta variância.

        Args:
            population: A população de onde selecionar.
            num_parents: Número de pais a selecionar.

        Returns:
            Lista de cromossomos pais selecionados.
        """
        # Ordena população por fitness (melhor primeiro para minimização)
        sorted_pop = sorted(population.chromosomes, key=lambda c: c.fitness)
        n = len(sorted_pop)

        # Atribui rankings (melhor recebe maior ranking)
        ranks = np.arange(n, 0, -1)  # [n, n-1, ..., 2, 1]

        # Calcula probabilidades de seleção baseadas no ranking
        total_rank = np.sum(ranks)
        probabilities = ranks / total_rank

        # Seleciona pais
        indices = np.random.choice(
            n,
            size=num_parents,
            replace=True,
            p=probabilities
        )

        return [sorted_pop[i].copy() for i in indices]

    @staticmethod
    def random_selection(population: Population, num_parents: int) -> List[Chromosome]:
        """
        Seleção aleatória (probabilidade uniforme).

        Args:
            population: A população de onde selecionar.
            num_parents: Número de pais a selecionar.

        Returns:
            Lista de cromossomos pais selecionados.
        """
        indices = np.random.choice(len(population), size=num_parents, replace=True)
        return [population[i].copy() for i in indices]

    @staticmethod
    def elitism(population: Population, num_elite: int) -> List[Chromosome]:
        """
        Seleciona os melhores indivíduos (elitismo).

        Utilizado para preservar as melhores soluções entre gerações.

        Args:
            population: A população de onde selecionar.
            num_elite: Número de indivíduos elite a selecionar.

        Returns:
            Lista de cromossomos elite (cópias).
        """
        sorted_pop = sorted(population.chromosomes, key=lambda c: c.fitness)
        return [c.copy() for c in sorted_pop[:num_elite]]
