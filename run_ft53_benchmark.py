"""
Benchmark do ft53 - Comparacao com o artigo Hussain et al. 2017

Este script executa benchmarks na instancia ft53 (53 cidades, ATSP)
comparando os operadores PMX, OX e CX2 com os resultados do artigo.

Parametros do artigo (Tabela 4):
- population_size: 150
- max_generations: 500
- crossover_probability: 0.80
- mutation_probability: 0.10
- 30 runs por operador

Resultados do artigo para ft53:
- PMX: Best = 13445, Worst = 16947, Average = 14949
- OX: Best = 13826, Worst = 16279, Average = 14724
- CX2: Best = 10987, Worst = 13055, Average = 12243

Valor otimo: 6905
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser


# Resultados do artigo para comparacao
PAPER_RESULTS = {
    'PMX': {'best': 13445, 'worst': 16947, 'average': 14949},
    'OX': {'best': 13826, 'worst': 16279, 'average': 14724},
    'CX2': {'best': 10987, 'worst': 13055, 'average': 12243}
}

OPTIMAL_VALUE = 6905


def run_experiment(fitness_calc, crossover_type, selection_type, num_runs=30):
    """Executa multiplos experimentos do AG e retorna estatisticas."""
    all_values = []
    best_tour = None
    best_fitness = float('inf')

    for run in range(num_runs):
        config = GAConfig(
            population_size=150,
            max_generations=500,
            crossover_probability=0.80,
            mutation_probability=0.10,
            crossover_type=crossover_type,
            mutation_type='swap',
            selection_type=selection_type,
            elitism_count=2,
            random_seed=run
        )

        ga = GeneticAlgorithm(fitness_calc, config)
        result = ga.run(verbose=False)
        all_values.append(result.best_fitness)

        if result.best_fitness < best_fitness:
            best_fitness = result.best_fitness
            best_tour = result.best_tour

        # Progress indicator
        print(f"    Run {run+1}/{num_runs}: {result.best_fitness:.0f}", end='\r')

    print()  # New line after progress

    return {
        'best': float(min(all_values)),
        'worst': float(max(all_values)),
        'average': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'median': float(np.median(all_values)),
        'all_values': [float(v) for v in all_values],
        'best_tour': [int(c) for c in best_tour] if best_tour else None,
    }


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("BENCHMARK FT53 - Comparacao com Hussain et al. 2017")
    print("=" * 80)

    # Carregar instancia ft53
    parser = TSPLibParser()
    instance_path = Path("tsplib/instances/ft53.atsp")

    if not instance_path.exists():
        print(f"Erro: {instance_path} nao encontrado")
        return

    instance = parser.parse(str(instance_path))
    distance_matrix = instance.get_distance_matrix()
    fitness_calc = FitnessCalculator(distance_matrix)

    print(f"\nInstancia: ft53 ({instance.dimension} cidades)")
    print(f"Tipo: ATSP (assimetrico)")
    print(f"Valor otimo: {OPTIMAL_VALUE}")
    print(f"\nParametros do artigo:")
    print(f"  - Population: 150")
    print(f"  - Generations: 500")
    print(f"  - Crossover prob: 0.80")
    print(f"  - Mutation prob: 0.10")
    print(f"  - Runs: 30")

    num_runs = 30
    operators = ['PMX', 'OX', 'CX2']

    # Usar selecao por torneio (descoberta do projeto)
    selection_type = 'tournament'

    print(f"\n{'='*80}")
    print(f"Executando benchmarks com selecao por torneio")
    print(f"{'='*80}")

    results = {}

    for op in operators:
        print(f"\n[{op}] Executando {num_runs} runs...")
        start = time.time()
        results[op] = run_experiment(fitness_calc, op, selection_type, num_runs)
        elapsed = time.time() - start

        paper = PAPER_RESULTS[op]
        gap = ((results[op]['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)

        print(f"    Tempo: {elapsed:.1f}s")
        print(f"    Nossa impl.: Best={results[op]['best']:.0f}, Avg={results[op]['average']:.1f}, Worst={results[op]['worst']:.0f}")
        print(f"    Artigo:       Best={paper['best']}, Avg={paper['average']}, Worst={paper['worst']}")
        print(f"    Gap ao otimo: {gap:.1f}%")

    # Imprimir tabela comparativa
    print("\n" + "=" * 80)
    print("TABELA COMPARATIVA - ft53")
    print("=" * 80)
    print(f"\n{'Operador':<10} {'Metrica':<10} {'Nossa Impl.':<15} {'Artigo':<15} {'Diferenca':<15}")
    print("-" * 70)

    for op in operators:
        our = results[op]
        paper = PAPER_RESULTS[op]

        print(f"{op:<10} {'Best':<10} {our['best']:<15.0f} {paper['best']:<15} {our['best']-paper['best']:<15.0f}")
        print(f"{'':<10} {'Average':<10} {our['average']:<15.1f} {paper['average']:<15} {our['average']-paper['average']:<15.1f}")
        print(f"{'':<10} {'Worst':<10} {our['worst']:<15.0f} {paper['worst']:<15} {our['worst']-paper['worst']:<15.0f}")
        print("-" * 70)

    # Gap ao otimo
    print(f"\n{'='*80}")
    print("GAP AO VALOR OTIMO (6905)")
    print("=" * 80)
    print(f"\n{'Operador':<10} {'Best':<10} {'Gap%':<10} {'Artigo Best':<15} {'Artigo Gap%':<15}")
    print("-" * 60)

    for op in operators:
        our_gap = ((results[op]['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        paper_gap = ((PAPER_RESULTS[op]['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"{op:<10} {results[op]['best']:<10.0f} {our_gap:<10.1f}% {PAPER_RESULTS[op]['best']:<15} {paper_gap:<15.1f}%")

    # Tempo total
    total_elapsed = time.time() - total_start_time
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)

    print(f"\nTempo total de execucao: {minutes}m {seconds}s")

    # Salvar resultados
    output_data = {
        'instance': 'ft53',
        'optimal': OPTIMAL_VALUE,
        'selection_type': selection_type,
        'num_runs': num_runs,
        'paper_results': PAPER_RESULTS,
        'our_results': results,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'execution_time_seconds': total_elapsed
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"ft53_benchmark_{output_data['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    latest_path = output_dir / "ft53_benchmark_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Gerar tabela markdown para a apresentacao
    print("\n" + "=" * 80)
    print("TABELA MARKDOWN PARA APRESENTACAO")
    print("=" * 80)
    print("""
## Resultados ft53 (53 cidades, ATSP)

| Operador | Metrica | Nossa Impl. | Artigo | Status |
|----------|---------|-------------|--------|--------|""")

    for op in operators:
        our = results[op]
        paper = PAPER_RESULTS[op]
        best_status = "MELHOR" if our['best'] < paper['best'] else ("IGUAL" if our['best'] == paper['best'] else "PIOR")
        avg_status = "MELHOR" if our['average'] < paper['average'] else ("IGUAL" if our['average'] == paper['average'] else "PIOR")

        print(f"| {op} | Best | {our['best']:.0f} | {paper['best']} | {best_status} |")
        print(f"| | Average | {our['average']:.1f} | {paper['average']} | {avg_status} |")
        print(f"| | Worst | {our['worst']:.0f} | {paper['worst']} | |")


if __name__ == "__main__":
    main()
