"""
Implementação principal do Algoritmo Genético para o TSP.

Implementa o AG conforme descrito em Hussain et al., 2017.

Passos do Algoritmo:
1. Criar uma população inicial de P cromossomos
2. Avaliar o fitness de cada cromossomo
3. Escolher P/2 pais da população atual via seleção proporcional
4. Selecionar aleatoriamente dois pais para criar filhos usando cruzamento
5. Aplicar operadores de mutação para pequenas alterações
6. Repetir os Passos 4 e 5 até que todos os pais sejam selecionados e cruzados
7. Substituir a população antiga pela nova
8. Avaliar o fitness de cada cromossomo na nova população
9. Encerrar se as gerações atingirem o limite superior; caso contrário, ir para o Passo 3
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time

from .chromosome import Chromosome, Population
from .fitness import FitnessCalculator
from .selection import Selection
from .crossover import CrossoverFactory, PMX, OX, CX2
from .mutation import Mutation


@dataclass
class GAConfig:
    """Parâmetros de configuração para o Algoritmo Genético."""
    population_size: int = 150
    max_generations: int = 500
    crossover_probability: float = 0.80
    mutation_probability: float = 0.10
    crossover_type: str = 'CX2'
    mutation_type: str = 'swap'
    selection_type: str = 'roulette'
    elitism_count: int = 2  # Manter os N melhores indivíduos
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Valida os parâmetros de configuração."""
        assert self.population_size > 0, "Population size must be positive"
        assert self.max_generations > 0, "Max generations must be positive"
        assert 0 <= self.crossover_probability <= 1, "Crossover probability must be in [0, 1]"
        assert 0 <= self.mutation_probability <= 1, "Mutation probability must be in [0, 1]"
        assert self.crossover_type.upper() in ['PMX', 'OX', 'CX2'], \
            f"Invalid crossover type: {self.crossover_type}"


@dataclass
class GAResult:
    """Resultados de uma execução do AG."""
    best_tour: List[int] = field(default_factory=list)
    best_fitness: float = float('inf')
    fitness_history: List[float] = field(default_factory=list)
    avg_fitness_history: List[float] = field(default_factory=list)
    worst_fitness_history: List[float] = field(default_factory=list)
    generations_run: int = 0
    execution_time: float = 0.0
    config: Optional[GAConfig] = None

    def get_gap_percentage(self, optimal: float) -> float:
        """Calcula o gap percentual em relação à solução ótima."""
        if optimal == 0:
            return 0.0
        return ((self.best_fitness - optimal) / optimal) * 100


class GeneticAlgorithm:
    """
    Algoritmo Genético para o Problema do Caixeiro Viajante.

    Implementa o algoritmo de:
    Hussain et al., "Genetic Algorithm for Traveling Salesman Problem
    with Modified Cycle Crossover Operator", 2017
    """

    def __init__(self, fitness_calculator: FitnessCalculator, config: GAConfig = None):
        """
        Inicializa o Algoritmo Genético.

        Args:
            fitness_calculator: Instância de FitnessCalculator com a matriz de distâncias.
            config: GAConfig com os parâmetros do algoritmo.
        """
        self.fitness_calculator = fitness_calculator
        self.config = config or GAConfig()
        self.n_cities = fitness_calculator.n_cities

        # Define semente aleatória se especificada
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Obtém operador de cruzamento
        self.crossover_fn = CrossoverFactory.get_operator(self.config.crossover_type)

        # Inicializa rastreamento de resultados
        self.result = GAResult(config=self.config)

    def run(self, verbose: bool = False) -> GAResult:
        """
        Executa o algoritmo genético.

        Args:
            verbose: Se True, imprime informações de progresso.

        Returns:
            GAResult com a melhor solução e estatísticas.
        """
        start_time = time.time()

        # Passo 1: Criar população inicial
        population = Population(self.config.population_size, self.n_cities)

        # Passo 2: Avaliar população inicial
        self.fitness_calculator.evaluate_population(population)
        population.sort_by_fitness()

        # Rastreia melhor solução
        best_chromosome = population.get_best().copy()
        self.result.best_fitness = best_chromosome.fitness
        self.result.best_tour = best_chromosome.get_tour()

        # Registra estatísticas iniciais
        self._record_generation_stats(population)

        if verbose:
            print(f"Geração 0: Melhor = {best_chromosome.fitness:.2f}, "
                  f"Média = {population.get_average_fitness():.2f}")

        # Loop principal de evolução
        for generation in range(1, self.config.max_generations + 1):
            # Passo 3: Selecionar pais (P/2 pares)
            num_parents = self.config.population_size
            parents = self._select_parents(population, num_parents)

            # Passos 4-6: Criar filhos por cruzamento e mutação
            offspring = self._create_offspring(parents)

            # Passo 7: Criar nova população (elitismo + filhos)
            elite = Selection.elitism(population, self.config.elitism_count)
            new_chromosomes = elite + offspring[:self.config.population_size - len(elite)]

            population = Population(
                self.config.population_size,
                self.n_cities,
                chromosomes=new_chromosomes[:self.config.population_size]
            )

            # Passo 8: Avaliar nova população
            self.fitness_calculator.evaluate_population(population)
            population.sort_by_fitness()

            # Atualiza melhor solução se houve melhoria
            current_best = population.get_best()
            if current_best.fitness < self.result.best_fitness:
                self.result.best_fitness = current_best.fitness
                self.result.best_tour = current_best.get_tour()

            # Registra estatísticas
            self._record_generation_stats(population)

            if verbose and generation % 50 == 0:
                print(f"Geração {generation}: Melhor = {self.result.best_fitness:.2f}, "
                      f"Média = {population.get_average_fitness():.2f}")

        # Finaliza resultado
        self.result.generations_run = self.config.max_generations
        self.result.execution_time = time.time() - start_time

        if verbose:
            print(f"\nFinal: Melhor fitness = {self.result.best_fitness:.2f}")
            print(f"Tempo de execução: {self.result.execution_time:.2f}s")

        return self.result

    def _select_parents(self, population: Population, num_parents: int) -> List[Chromosome]:
        """Seleciona pais para cruzamento com base no tipo de seleção."""
        if self.config.selection_type == 'roulette':
            return Selection.roulette_wheel(population, num_parents)
        elif self.config.selection_type == 'tournament':
            return Selection.tournament(population, num_parents)
        elif self.config.selection_type == 'rank':
            return Selection.rank_based(population, num_parents)
        else:
            return Selection.roulette_wheel(population, num_parents)

    def _create_offspring(self, parents: List[Chromosome]) -> List[Chromosome]:
        """Cria filhos por cruzamento e mutação."""
        offspring = []

        # Emparelha pais e aplica cruzamento
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Aplica cruzamento com probabilidade
            if np.random.random() < self.config.crossover_probability:
                child1, child2 = self.crossover_fn(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Aplica mutação
            Mutation.apply_mutation(child1, self.config.mutation_type,
                                   self.config.mutation_probability)
            Mutation.apply_mutation(child2, self.config.mutation_type,
                                   self.config.mutation_probability)

            offspring.extend([child1, child2])

        return offspring

    def _record_generation_stats(self, population: Population):
        """Registra estatísticas da geração atual."""
        self.result.fitness_history.append(population.get_best().fitness)
        self.result.avg_fitness_history.append(population.get_average_fitness())
        self.result.worst_fitness_history.append(population.get_worst().fitness)


def run_ga_experiment(
    distance_matrix: np.ndarray,
    config: GAConfig = None,
    num_runs: int = 30,
    verbose: bool = False
) -> Dict:
    """
    Executa múltiplos experimentos do AG e coleta estatísticas.

    Args:
        distance_matrix: Matriz de distâncias para a instância do TSP.
        config: Configuração do AG.
        num_runs: Número de execuções independentes.
        verbose: Imprimir progresso.

    Returns:
        Dicionário com resultados dos experimentos.
    """
    config = config or GAConfig()
    fitness_calc = FitnessCalculator(distance_matrix)

    results = []
    best_values = []
    worst_values = []

    for run in range(num_runs):
        if config.random_seed is not None:
            run_config = GAConfig(
                population_size=config.population_size,
                max_generations=config.max_generations,
                crossover_probability=config.crossover_probability,
                mutation_probability=config.mutation_probability,
                crossover_type=config.crossover_type,
                mutation_type=config.mutation_type,
                selection_type=config.selection_type,
                elitism_count=config.elitism_count,
                random_seed=config.random_seed + run
            )
        else:
            run_config = config

        ga = GeneticAlgorithm(fitness_calc, run_config)
        result = ga.run(verbose=False)
        results.append(result)
        best_values.append(result.best_fitness)

        if verbose:
            print(f"Execução {run + 1}/{num_runs}: Melhor = {result.best_fitness:.2f}")

    return {
        'results': results,
        'best': min(best_values),
        'worst': max(best_values),
        'average': np.mean(best_values),
        'std': np.std(best_values),
        'all_best_values': best_values,
    }
