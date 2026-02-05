"""
Executor de experimentos para benchmarks do AG aplicado ao PCV.

Executa experimentos conforme a metodologia do artigo:
- Múltiplas instâncias da TSPLIB
- 30 execuções por configuração
- Três operadores de cruzamento (PMX, OX, CX2)
- Coleta de estatísticas (melhor, pior, média)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig, GAResult
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser, TSPInstance


@dataclass
class ExperimentConfig:
    """Configuração para execução de experimentos."""
    num_runs: int = 30
    crossover_operators: Tuple[str, ...] = ('PMX', 'OX', 'CX2')

    # Parâmetros para instâncias pequenas/médias (< 100 cidades)
    small_population_size: int = 150
    small_max_generations: int = 500

    # Parâmetros para instâncias grandes (>= 100 cidades)
    large_population_size: int = 200
    large_max_generations: int = 1000

    # Parâmetros comuns
    crossover_probability: float = 0.80
    mutation_probability: float = 0.10
    mutation_type: str = 'swap'
    selection_type: str = 'roulette'
    elitism_count: int = 2


@dataclass
class InstanceResult:
    """Resultados para uma única instância com um operador de cruzamento."""
    instance_name: str
    crossover_type: str
    num_cities: int
    optimal_value: Optional[float]
    best: float
    worst: float
    average: float
    std: float
    all_values: List[float]
    best_tour: List[int]
    avg_time: float
    num_runs: int
    gap_percentage: Optional[float] = None

    def __post_init__(self):
        if self.optimal_value and self.optimal_value > 0:
            self.gap_percentage = ((self.best - self.optimal_value) / self.optimal_value) * 100


class ExperimentRunner:
    """
    Executa experimentos do AG em instâncias da TSPLIB.
    """

    def __init__(self, instances_dir: str, results_dir: str, config: ExperimentConfig = None):
        """
        Inicializa o executor de experimentos.

        Args:
            instances_dir: Diretório contendo os arquivos de instâncias da TSPLIB.
            results_dir: Diretório para salvar os resultados.
            config: Configuração do experimento.
        """
        self.instances_dir = Path(instances_dir)
        self.results_dir = Path(results_dir)
        self.config = config or ExperimentConfig()
        self.parser = TSPLibParser()

        # Cria o diretório de resultados se necessário
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_instance(self, filename: str) -> TSPInstance:
        """Carrega uma instância da TSPLIB a partir de um arquivo."""
        filepath = self.instances_dir / filename
        return self.parser.parse(str(filepath))

    def run_instance(self, instance: TSPInstance, crossover_type: str,
                     verbose: bool = False) -> InstanceResult:
        """
        Executa experimentos em uma única instância com um tipo de cruzamento.

        Args:
            instance: Instância do PCV a ser resolvida.
            crossover_type: Operador de cruzamento a ser utilizado.
            verbose: Exibir progresso.

        Returns:
            InstanceResult com estatísticas.
        """
        # Determina parâmetros com base no tamanho da instância
        if instance.dimension < 100:
            pop_size = self.config.small_population_size
            max_gen = self.config.small_max_generations
        else:
            pop_size = self.config.large_population_size
            max_gen = self.config.large_max_generations

        # Obtém a matriz de distâncias
        distance_matrix = instance.get_distance_matrix()
        fitness_calc = FitnessCalculator(distance_matrix)

        # Executa múltiplos experimentos
        all_values = []
        all_times = []
        best_result = None

        for run in range(self.config.num_runs):
            config = GAConfig(
                population_size=pop_size,
                max_generations=max_gen,
                crossover_probability=self.config.crossover_probability,
                mutation_probability=self.config.mutation_probability,
                crossover_type=crossover_type,
                mutation_type=self.config.mutation_type,
                selection_type=self.config.selection_type,
                elitism_count=self.config.elitism_count,
                random_seed=run  # Semente diferente para cada execução
            )

            ga = GeneticAlgorithm(fitness_calc, config)
            result = ga.run(verbose=False)

            all_values.append(result.best_fitness)
            all_times.append(result.execution_time)

            if best_result is None or result.best_fitness < best_result.best_fitness:
                best_result = result

            if verbose:
                print(f"  Execução {run + 1}/{self.config.num_runs}: {result.best_fitness:.2f}")

        return InstanceResult(
            instance_name=instance.name,
            crossover_type=crossover_type,
            num_cities=instance.dimension,
            optimal_value=instance.optimal_value,
            best=min(all_values),
            worst=max(all_values),
            average=np.mean(all_values),
            std=np.std(all_values),
            all_values=all_values,
            best_tour=best_result.best_tour,
            avg_time=np.mean(all_times),
            num_runs=self.config.num_runs
        )

    def run_all_operators(self, instance: TSPInstance,
                          verbose: bool = True) -> Dict[str, InstanceResult]:
        """
        Executa experimentos com todos os operadores de cruzamento em uma instância.

        Args:
            instance: Instância do PCV a ser resolvida.
            verbose: Exibir progresso.

        Returns:
            Dicionário mapeando nome do operador para resultados.
        """
        results = {}

        for operator in self.config.crossover_operators:
            if verbose:
                print(f"\n  Executando {operator}...")

            result = self.run_instance(instance, operator, verbose=False)
            results[operator] = result

            if verbose:
                print(f"    Melhor: {result.best:.2f}, Média: {result.average:.2f}, "
                      f"Pior: {result.worst:.2f}")

        return results

    def run_benchmark_suite(self, instance_files: List[str],
                           verbose: bool = True) -> Dict[str, Dict[str, InstanceResult]]:
        """
        Executa experimentos em múltiplas instâncias.

        Args:
            instance_files: Lista de nomes de arquivos de instâncias.
            verbose: Exibir progresso.

        Returns:
            Dicionário aninhado: nome_instância -> operador -> resultados
        """
        all_results = {}

        for i, filename in enumerate(instance_files):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Instância {i+1}/{len(instance_files)}: {filename}")
                print('='*60)

            try:
                instance = self.load_instance(filename)
                results = self.run_all_operators(instance, verbose)
                all_results[instance.name] = results

                # Salva resultados intermediários
                self._save_results(all_results)

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")
                continue

        return all_results

    def _save_results(self, results: Dict[str, Dict[str, InstanceResult]]):
        """Salva resultados em arquivo JSON."""
        # Converte para formato serializável
        data = {}
        for instance_name, operator_results in results.items():
            data[instance_name] = {}
            for operator, result in operator_results.items():
                data[instance_name][operator] = {
                    'instance_name': result.instance_name,
                    'crossover_type': result.crossover_type,
                    'num_cities': result.num_cities,
                    'optimal_value': result.optimal_value,
                    'best': result.best,
                    'worst': result.worst,
                    'average': result.average,
                    'std': result.std,
                    'gap_percentage': result.gap_percentage,
                    'avg_time': result.avg_time,
                    'num_runs': result.num_runs,
                    'best_tour': result.best_tour,
                    'all_values': result.all_values,
                }

        # Salva no arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.results_dir / f'results_{timestamp}.json'

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        # Também salva como mais recente
        latest_path = self.results_dir / 'results_latest.json'
        with open(latest_path, 'w') as f:
            json.dump(data, f, indent=2)

    def print_comparison_table(self, results: Dict[str, Dict[str, InstanceResult]]):
        """Exibe resultados em formato de tabela semelhante ao artigo."""
        print("\n" + "="*80)
        print("TABELA DE COMPARAÇÃO DE RESULTADOS")
        print("="*80)

        # Cabeçalho
        print(f"{'Instância':<12} {'N':<5} {'Ótimo':<8} {'Op':<5} "
              f"{'Melhor':<10} {'Pior':<10} {'Média':<10} {'Gap%':<8}")
        print("-"*80)

        for instance_name, operator_results in results.items():
            first_row = True
            for operator, result in operator_results.items():
                if first_row:
                    opt_str = f"{result.optimal_value:.0f}" if result.optimal_value else "N/D"
                    print(f"{instance_name:<12} {result.num_cities:<5} {opt_str:<8} "
                          f"{operator:<5} {result.best:<10.2f} {result.worst:<10.2f} "
                          f"{result.average:<10.2f} "
                          f"{result.gap_percentage:.2f}%" if result.gap_percentage else "N/D")
                    first_row = False
                else:
                    print(f"{'':<12} {'':<5} {'':<8} "
                          f"{operator:<5} {result.best:<10.2f} {result.worst:<10.2f} "
                          f"{result.average:<10.2f} "
                          f"{result.gap_percentage:.2f}%" if result.gap_percentage else "N/D")
            print("-"*80)


def create_manual_instances() -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Cria matrizes de distância manuais para instâncias cujos arquivos não estão disponíveis.
    Retorna dicionário: nome_instância -> (matriz_distância, valor_ótimo)
    """
    instances = {}

    # Instância de 7 cidades do artigo (Tabela 2)
    instances['paper7'] = (
        np.array([
            [0,  34, 36, 37, 31, 33, 35],
            [34,  0, 29, 23, 22, 25, 24],
            [36, 29,  0, 17, 12, 18, 17],
            [37, 23, 17,  0, 32, 30, 29],
            [31, 22, 12, 17,  0, 26, 24],
            [33, 25, 18, 30, 26,  0, 19],
            [35, 24, 17, 29, 24, 19,  0],
        ], dtype=np.float64),
        159.0
    )

    return instances


if __name__ == "__main__":
    # Teste rápido com a instância de 7 cidades do artigo
    from tsplib.parser import TSPLibParser

    instances = create_manual_instances()
    matrix, optimal = instances['paper7']

    instance = TSPLibParser.create_manual_instance(matrix, "paper7")
    instance.optimal_value = optimal

    config = ExperimentConfig(num_runs=10)
    runner = ExperimentRunner(
        instances_dir="tsplib/instances",
        results_dir="results",
        config=config
    )

    print("Testando com a instância de 7 cidades do artigo...")
    results = runner.run_all_operators(instance, verbose=True)

    print("\n" + "="*60)
    print("Resultados do Teste Rápido:")
    print("="*60)
    for op, res in results.items():
        print(f"{op}: Melhor={res.best:.0f}, Média={res.average:.1f}, "
              f"Pior={res.worst:.0f}, Gap={res.gap_percentage:.1f}%")
