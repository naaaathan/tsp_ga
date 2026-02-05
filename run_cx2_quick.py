"""
Teste Rapido de Parametros CX2 - Focado em combinacoes-chave de parametros
"""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser


def run_test(fitness_calc, pop_size, max_gen, cx_prob, mut_prob, sel_type, elite, num_runs=5):
    """Executa um teste rapido com os parametros fornecidos."""
    all_values = []

    for run in range(num_runs):
        config = GAConfig(
            population_size=pop_size,
            max_generations=max_gen,
            crossover_probability=cx_prob,
            mutation_probability=mut_prob,
            crossover_type='CX2',
            selection_type=sel_type,
            elitism_count=elite,
            random_seed=run
        )

        ga = GeneticAlgorithm(fitness_calc, config)
        result = ga.run(verbose=False)
        all_values.append(result.best_fitness)

    return {
        'best': min(all_values),
        'worst': max(all_values),
        'average': np.mean(all_values),
    }


def main():
    print("Teste Rapido de Parametros CX2")
    print("="*60)

    # Carregar instancia
    parser = TSPLibParser()
    instance = parser.parse("tsplib/instances/dantzig42.tsp")
    distance_matrix = instance.get_distance_matrix()
    fitness_calc = FitnessCalculator(distance_matrix)

    optimal = 699
    print(f"Instancia: dantzig42, Otimo: {optimal}")
    print(f"CX2 do Artigo: Melhor=699, Media=802")
    print("-"*60)

    # Configuracoes de teste (rapido - 5 execucoes cada)
    configs = [
        # nome, pop, ger, cx, mut, sel, elite
        ("Baseline", 150, 500, 0.80, 0.10, 'roulette', 2),
        ("Pop=300", 300, 500, 0.80, 0.10, 'roulette', 2),
        ("Ger=1000", 150, 1000, 0.80, 0.10, 'roulette', 2),
        ("Pop=300,Ger=1000", 300, 1000, 0.80, 0.10, 'roulette', 2),
        ("Elite=10", 150, 500, 0.80, 0.10, 'roulette', 10),
        ("Mut=0.20", 150, 500, 0.80, 0.20, 'roulette', 2),
        ("Torneio", 150, 500, 0.80, 0.10, 'tournament', 2),
        ("Combinado1", 300, 1000, 0.85, 0.15, 'roulette', 10),
        ("Combinado2", 400, 1500, 0.80, 0.10, 'tournament', 5),
        ("Agressivo", 500, 2000, 0.90, 0.15, 'roulette', 15),
    ]

    results = []
    print(f"\n{'Config':<20} | {'Melhor':<8} | {'Media':<8} | {'Gap%':<8} | Tempo")
    print("-"*60)

    for name, pop, gen, cx, mut, sel, elite in configs:
        start = time.time()
        r = run_test(fitness_calc, pop, gen, cx, mut, sel, elite, num_runs=5)
        elapsed = time.time() - start
        gap = ((r['best'] - optimal) / optimal * 100)
        print(f"{name:<20} | {r['best']:<8.0f} | {r['average']:<8.1f} | {gap:<8.1f}% | {elapsed:.1f}s")
        results.append((name, r, gap))

    # Encontrar o melhor
    best = min(results, key=lambda x: x[1]['best'])
    print("\n" + "="*60)
    print(f"Melhor configuracao: {best[0]} com Melhor={best[1]['best']:.0f}")

    # Execucao estendida na melhor configuracao
    print("\nExecutando teste estendido (30 execucoes) na melhor configuracao...")
    # Encontrar os parametros da melhor configuracao
    best_idx = [c[0] for c in configs].index(best[0])
    _, pop, gen, cx, mut, sel, elite = configs[best_idx]

    r = run_test(fitness_calc, pop, gen, cx, mut, sel, elite, num_runs=30)
    gap = ((r['best'] - optimal) / optimal * 100)
    print(f"Estendido: Melhor={r['best']:.0f}, Media={r['average']:.1f}, Gap={gap:.1f}%")


if __name__ == "__main__":
    main()
