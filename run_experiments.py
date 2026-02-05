"""
Script principal para executar todos os experimentos conforme a metodologia do artigo.

Reproduz os experimentos de:
Hussain et al., "Genetic Algorithm for Traveling Salesman Problem
with Modified Cycle Crossover Operator", 2017
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Adicionar raiz do projeto ao caminho
sys.path.insert(0, str(Path(__file__).parent))

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser, TSPInstance
from experiments.runner import ExperimentRunner, ExperimentConfig, create_manual_instances
from experiments.statistics import ExperimentStatistics


# Resultados do artigo para comparacao (Tabelas 4 e 5)
PAPER_RESULTS = {
    'gr21': {
        'optimal': 2707,
        'PMX': {'best': 2962, 'worst': 3322, 'average': 3127},
        'OX': {'best': 3005, 'worst': 3693, 'average': 3208},
        'CX2': {'best': 2995, 'worst': 3576, 'average': 3145},
    },
    'fri26': {
        'optimal': 937,
        'PMX': {'best': 1056, 'worst': 1294, 'average': 1133},
        'OX': {'best': 1051, 'worst': 1323, 'average': 1158},
        'CX2': {'best': 1099, 'worst': 1278, 'average': 1128},
    },
    'dantzig42': {
        'optimal': 699,
        'PMX': {'best': 1298, 'worst': 1606, 'average': 1425},
        'OX': {'best': 1222, 'worst': 1562, 'average': 1301},
        'CX2': {'best': 699, 'worst': 920, 'average': 802},
    },
    'ftv33': {
        'optimal': 1286,
        'PMX': {'best': 1708, 'worst': 2399, 'average': 2012},
        'OX': {'best': 1804, 'worst': 2366, 'average': 2098},
        'CX2': {'best': 1811, 'worst': 2322, 'average': 2083},
    },
    'ftv38': {
        'optimal': 1530,
        'PMX': {'best': 2345, 'worst': 2726, 'average': 2578},
        'OX': {'best': 2371, 'worst': 2913, 'average': 2617},
        'CX2': {'best': 2252, 'worst': 2718, 'average': 2560},
    },
    'ft53': {
        'optimal': 6905,
        'PMX': {'best': 13445, 'worst': 16947, 'average': 14949},
        'OX': {'best': 13826, 'worst': 16279, 'average': 14724},
        'CX2': {'best': 10987, 'worst': 13055, 'average': 12243},
    },
}


def run_single_instance(instance: TSPInstance, num_runs: int = 30,
                        verbose: bool = True) -> dict:
    """Executa experimentos em uma unica instancia com todos os operadores."""
    results = {}

    # Determinar parametros com base no tamanho da instancia
    if instance.dimension < 100:
        pop_size = 150
        max_gen = 500
    else:
        pop_size = 200
        max_gen = 1000

    distance_matrix = instance.get_distance_matrix()
    fitness_calc = FitnessCalculator(distance_matrix)

    for operator in ['PMX', 'OX', 'CX2']:
        if verbose:
            print(f"  Executando {operator}...", end=' ', flush=True)

        all_values = []
        best_tour = None
        best_fitness = float('inf')

        for run in range(num_runs):
            config = GAConfig(
                population_size=pop_size,
                max_generations=max_gen,
                crossover_probability=0.80,
                mutation_probability=0.10,
                crossover_type=operator,
                random_seed=run
            )

            ga = GeneticAlgorithm(fitness_calc, config)
            result = ga.run(verbose=False)
            all_values.append(result.best_fitness)

            if result.best_fitness < best_fitness:
                best_fitness = result.best_fitness
                best_tour = result.best_tour

        results[operator] = {
            'best': min(all_values),
            'worst': max(all_values),
            'average': np.mean(all_values),
            'std': np.std(all_values),
            'all_values': all_values,
            'best_tour': best_tour,
        }

        if verbose:
            print(f"Melhor: {results[operator]['best']:.0f}, "
                  f"Media: {results[operator]['average']:.1f}")

    return results


def print_comparison_table(our_results: dict, paper_results: dict, instance_name: str):
    """Imprime comparacao entre nossos resultados e os resultados do artigo."""
    print(f"\n{'='*70}")
    print(f"COMPARACAO: {instance_name}")
    print(f"{'='*70}")

    optimal = paper_results.get('optimal', 'N/D')
    print(f"Valor otimo: {optimal}")

    print(f"\n{'Operador':<8} | {'Metrica':<8} | {'Nosso':<12} | {'Artigo':<12} | {'Diff':<10}")
    print("-"*58)

    for operator in ['PMX', 'OX', 'CX2']:
        if operator not in our_results or operator not in paper_results:
            continue

        our = our_results[operator]
        paper = paper_results[operator]

        for metric in ['best', 'worst', 'average']:
            our_val = our.get(metric, 0)
            paper_val = paper.get(metric, 0)
            diff = our_val - paper_val
            diff_str = f"{diff:+.1f}"

            op_label = operator if metric == 'best' else ''
            print(f"{op_label:<8} | {metric:<8} | {our_val:<12.1f} | {paper_val:<12.1f} | {diff_str:<10}")

        print("-"*58)


def run_available_benchmarks(instances_dir: str, num_runs: int = 30):
    """Executa experimentos nas instancias TSPLIB disponiveis."""
    parser = TSPLibParser()
    instances_path = Path(instances_dir)

    all_results = {}

    # Encontrar arquivos de instancias disponiveis
    available_files = list(instances_path.glob("*.tsp"))

    if not available_files:
        print(f"Nenhum arquivo .tsp encontrado em {instances_dir}")
        return all_results

    print(f"\nEncontrados {len(available_files)} arquivos de instancias")
    print("="*70)

    for filepath in sorted(available_files):
        instance_name = filepath.stem
        print(f"\n[{instance_name}] Carregando...")

        try:
            instance = parser.parse(str(filepath))
            print(f"[{instance_name}] {instance.dimension} cidades, "
                  f"Otimo: {instance.optimal_value or 'Desconhecido'}")

            results = run_single_instance(instance, num_runs=num_runs)
            all_results[instance_name] = {
                'dimension': instance.dimension,
                'optimal': instance.optimal_value,
                'results': results
            }

            # Imprimir comparacao com o artigo se disponivel
            if instance_name in PAPER_RESULTS:
                print_comparison_table(results, PAPER_RESULTS[instance_name], instance_name)

        except Exception as e:
            print(f"[{instance_name}] Erro: {e}")
            continue

    return all_results


def save_results(results: dict, output_dir: str):
    """Salva os resultados em um arquivo JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_path / f'experiment_results_{timestamp}.json'

    # Converter tipos numpy para tipos Python para serializacao JSON
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Tambem salvar como mais recente
    latest_path = output_path / 'results_latest.json'
    with open(latest_path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def print_final_summary(all_results: dict):
    """Imprime a tabela de resumo final."""
    print("\n" + "="*80)
    print("RESUMO FINAL DOS RESULTADOS")
    print("="*80)

    print(f"\n{'Instancia':<12} {'N':<5} {'Otm':<8} {'Op':<5} "
          f"{'Melhor':<10} {'Pior':<10} {'Media':<10} {'Gap%':<8}")
    print("-"*80)

    for instance_name, data in all_results.items():
        dimension = data['dimension']
        optimal = data.get('optimal')
        results = data['results']

        first = True
        for operator in ['PMX', 'OX', 'CX2']:
            if operator not in results:
                continue

            r = results[operator]
            gap = ((r['best'] - optimal) / optimal * 100) if optimal else 0

            if first:
                opt_str = f"{optimal:.0f}" if optimal else "N/D"
                print(f"{instance_name:<12} {dimension:<5} {opt_str:<8} "
                      f"{operator:<5} {r['best']:<10.0f} {r['worst']:<10.0f} "
                      f"{r['average']:<10.1f} {gap:<8.1f}%")
                first = False
            else:
                print(f"{'':<12} {'':<5} {'':<8} "
                      f"{operator:<5} {r['best']:<10.0f} {r['worst']:<10.0f} "
                      f"{r['average']:<10.1f} {gap:<8.1f}%")

        print("-"*80)


if __name__ == "__main__":
    print("="*70)
    print("EXPERIMENTOS DO ALGORITMO GENETICO PARA O TSP")
    print("Reproduzindo: Hussain et al., 2017")
    print("="*70)

    # Configuracao
    INSTANCES_DIR = "tsplib/instances"
    RESULTS_DIR = "results"
    NUM_RUNS = 30

    print(f"\nConfiguracao:")
    print(f"  Diretorio de instancias: {INSTANCES_DIR}")
    print(f"  Diretorio de resultados: {RESULTS_DIR}")
    print(f"  Numero de execucoes: {NUM_RUNS}")

    # Executar experimentos
    all_results = run_available_benchmarks(INSTANCES_DIR, NUM_RUNS)

    if all_results:
        # Imprimir resumo final
        print_final_summary(all_results)

        # Salvar resultados
        save_results(all_results, RESULTS_DIR)

    print("\nExperimentos concluidos!")
