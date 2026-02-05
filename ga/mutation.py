"""
Operadores de mutação para o TSP.

A mutação introduz pequenas alterações aleatórias para manter a diversidade genética
e ajudar a escapar de ótimos locais.

O artigo utiliza mutação por troca (swap) com probabilidade Pm = 0.10.
"""

import numpy as np
from typing import Optional
from .chromosome import Chromosome


class Mutation:
    """
    Operadores de mutação para cromossomos do TSP.
    """

    @staticmethod
    def swap(chromosome: Chromosome, probability: float = 0.1) -> Chromosome:
        """
        Mutação por troca: troca aleatoriamente duas cidades no percurso.

        Este é o operador de mutação utilizado no artigo.

        Args:
            chromosome: O cromossomo a ser mutado.
            probability: Probabilidade de ocorrência da mutação.

        Returns:
            Cromossomo mutado (modificado no local e retornado).
        """
        if np.random.random() < probability:
            n = len(chromosome)
            # Seleciona duas posições aleatórias
            i, j = np.random.choice(n, 2, replace=False)
            # Troca os valores
            chromosome.genes[i], chromosome.genes[j] = \
                chromosome.genes[j], chromosome.genes[i]
            # Invalida fitness
            chromosome.fitness = None

        return chromosome

    @staticmethod
    def insert(chromosome: Chromosome, probability: float = 0.1) -> Chromosome:
        """
        Mutação por inserção: remove uma cidade e a insere em uma posição aleatória.

        Args:
            chromosome: O cromossomo a ser mutado.
            probability: Probabilidade de ocorrência da mutação.

        Returns:
            Cromossomo mutado.
        """
        if np.random.random() < probability:
            n = len(chromosome)
            # Seleciona posição para remover
            i = np.random.randint(n)
            # Seleciona posição para inserir
            j = np.random.randint(n)

            if i != j:
                # Remove cidade na posição i
                city = chromosome.genes[i]
                genes = np.delete(chromosome.genes, i)
                # Insere na posição j
                chromosome.genes = np.insert(genes, j, city)
                chromosome.fitness = None

        return chromosome

    @staticmethod
    def inversion(chromosome: Chromosome, probability: float = 0.1) -> Chromosome:
        """
        Mutação por inversão: inverte um segmento aleatório do percurso.

        Também conhecida como movimento 2-opt quando os extremos do segmento são adjacentes
        ao restante do percurso.

        Args:
            chromosome: O cromossomo a ser mutado.
            probability: Probabilidade de ocorrência da mutação.

        Returns:
            Cromossomo mutado.
        """
        if np.random.random() < probability:
            n = len(chromosome)
            # Seleciona duas posições aleatórias
            i, j = sorted(np.random.choice(n, 2, replace=False))
            # Inverte o segmento entre i e j (inclusive)
            chromosome.genes[i:j+1] = chromosome.genes[i:j+1][::-1]
            chromosome.fitness = None

        return chromosome

    @staticmethod
    def scramble(chromosome: Chromosome, probability: float = 0.1) -> Chromosome:
        """
        Mutação por embaralhamento: embaralha aleatoriamente um segmento do percurso.

        Args:
            chromosome: O cromossomo a ser mutado.
            probability: Probabilidade de ocorrência da mutação.

        Returns:
            Cromossomo mutado.
        """
        if np.random.random() < probability:
            n = len(chromosome)
            # Seleciona duas posições aleatórias
            i, j = sorted(np.random.choice(n, 2, replace=False))
            # Embaralha o segmento
            segment = chromosome.genes[i:j+1].copy()
            np.random.shuffle(segment)
            chromosome.genes[i:j+1] = segment
            chromosome.fitness = None

        return chromosome

    @staticmethod
    def apply_mutation(chromosome: Chromosome,
                       mutation_type: str = 'swap',
                       probability: float = 0.1) -> Chromosome:
        """
        Aplica o tipo de mutação especificado ao cromossomo.

        Args:
            chromosome: O cromossomo a ser mutado.
            mutation_type: Tipo de mutação ('swap', 'insert', 'inversion', 'scramble').
            probability: Probabilidade de ocorrência da mutação.

        Returns:
            Cromossomo mutado.
        """
        mutation_functions = {
            'swap': Mutation.swap,
            'insert': Mutation.insert,
            'inversion': Mutation.inversion,
            'scramble': Mutation.scramble,
        }

        if mutation_type not in mutation_functions:
            raise ValueError(f"Unknown mutation type: {mutation_type}. "
                           f"Available: {list(mutation_functions.keys())}")

        return mutation_functions[mutation_type](chromosome, probability)
