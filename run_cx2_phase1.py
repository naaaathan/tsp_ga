"""
Otimizacao CX2 - Fase 1: Analise de Sensibilidade de Parametros

Este script testa cada parametro individualmente para identificar quais tem maior impacto
no desempenho do CX2.

Tempo estimado de execucao: ~20-30 minutos
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


def run_experiment(fitness_calc, config, num_runs=10):
    """Executa multiplos experimentos do AG e retorna estatisticas."""
    all_values = []
    best_tour = None
    best_fitness = float('inf')

    for run in range(num_runs):
        run_config = GAConfig(
            population_size=config['population_size'],
            max_generations=config['max_generations'],
            crossover_probability=config['crossover_probability'],
            mutation_probability=config['mutation_probability'],
            crossover_type='CX2',
            mutation_type=config.get('mutation_type', 'swap'),
            selection_type=config.get('selection_type', 'roulette'),
            elitism_count=config.get('elitism_count', 2),
            random_seed=run
        )

        ga = GeneticAlgorithm(fitness_calc, run_config)
        result = ga.run(verbose=False)
        all_values.append(result.best_fitness)

        if result.best_fitness < best_fitness:
            best_fitness = result.best_fitness
            best_tour = result.best_tour

    return {
        'best': float(min(all_values)),
        'worst': float(max(all_values)),
        'average': float(np.mean(all_values)),
        'std': float(np.std(all_values)),
        'all_values': [float(v) for v in all_values],
        'best_tour': [int(c) for c in best_tour] if best_tour else None,
        'config': config
    }


def print_result(name, result, optimal):
    """Imprime um unico resultado."""
    gap = ((result['best'] - optimal) / optimal * 100)
    print(f"  {name:<35} | Melhor: {result['best']:>6.0f} | "
          f"Media: {result['average']:>7.1f} | Gap: {gap:>6.1f}%")


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("OTIMIZACAO CX2 - FASE 1: ANALISE DE SENSIBILIDADE DE PARAMETROS")
    print("=" * 80)
    print("\nEsta fase testa cada parametro individualmente para encontrar os mais impactantes.")
    print("Tempo estimado: 20-30 minutos\n")

    # Carregar instancia dantzig42
    parser = TSPLibParser()
    instance_path = Path("tsplib/instances/dantzig42.tsp")

    if not instance_path.exists():
        print(f"Erro: {instance_path} nao encontrado")
        return

    instance = parser.parse(str(instance_path))
    distance_matrix = instance.get_distance_matrix()
    fitness_calc = FitnessCalculator(distance_matrix)

    optimal = 699
    paper_best = 699
    paper_avg = 802
    num_runs = 10

    print(f"Instancia: dantzig42 ({instance.dimension} cidades)")
    print(f"Otimo: {optimal}")
    print(f"CX2 do artigo: Melhor={paper_best}, Media={paper_avg}")
    print(f"Execucoes por configuracao: {num_runs}")
    print("-" * 80)

    # Configuracao base
    baseline = {
        'population_size': 150,
        'max_generations': 500,
        'crossover_probability': 0.80,
        'mutation_probability': 0.10,
        'selection_type': 'roulette',
        'elitism_count': 2,
    }

    all_results = {
        'instance': 'dantzig42',
        'optimal': optimal,
        'paper_cx2': {'best': paper_best, 'avg': paper_avg},
        'baseline': baseline,
        'num_runs': num_runs,
        'groups': {}
    }

    # ===== GRUPO 1: Tipo de Selecao (MAIOR PRIORIDADE) =====
    print("\n[1/6] TIPO DE SELECAO (maior impacto esperado)")
    print("-" * 50)

    selection_results = []
    for sel_type in ['roulette', 'tournament', 'rank']:
        config = baseline.copy()
        config['selection_type'] = sel_type

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Selection={sel_type}'
        elapsed = time.time() - start

        print_result(f"Selection={sel_type} ({elapsed:.1f}s)", result, optimal)
        selection_results.append(result)

    all_results['groups']['selection'] = selection_results

    # Encontrar melhor tipo de selecao
    best_selection = min(selection_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor selecao: {best_selection['name']} (Melhor={best_selection['best']:.0f})")

    # ===== GRUPO 2: Elitismo (ALTA PRIORIDADE) =====
    print("\n[2/6] ELITISMO")
    print("-" * 50)

    elitism_results = []
    for elite in [0, 2, 5, 10, 15, 20]:
        config = baseline.copy()
        config['elitism_count'] = elite

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Elite={elite}'
        elapsed = time.time() - start

        print_result(f"Elite={elite} ({elapsed:.1f}s)", result, optimal)
        elitism_results.append(result)

    all_results['groups']['elitism'] = elitism_results

    best_elitism = min(elitism_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor elitismo: {best_elitism['name']} (Melhor={best_elitism['best']:.0f})")

    # ===== GRUPO 3: Geracoes =====
    print("\n[3/6] GERACOES")
    print("-" * 50)

    generations_results = []
    for gen in [500, 750, 1000, 1500, 2000]:
        config = baseline.copy()
        config['max_generations'] = gen

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Gen={gen}'
        elapsed = time.time() - start

        print_result(f"Gen={gen} ({elapsed:.1f}s)", result, optimal)
        generations_results.append(result)

    all_results['groups']['generations'] = generations_results

    best_gen = min(generations_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor geracoes: {best_gen['name']} (Melhor={best_gen['best']:.0f})")

    # ===== GRUPO 4: Tamanho da Populacao =====
    print("\n[4/6] TAMANHO DA POPULACAO")
    print("-" * 50)

    population_results = []
    for pop in [150, 200, 300, 400, 500]:
        config = baseline.copy()
        config['population_size'] = pop

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Pop={pop}'
        elapsed = time.time() - start

        print_result(f"Pop={pop} ({elapsed:.1f}s)", result, optimal)
        population_results.append(result)

    all_results['groups']['population'] = population_results

    best_pop = min(population_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor populacao: {best_pop['name']} (Melhor={best_pop['best']:.0f})")

    # ===== GRUPO 5: Probabilidade de Mutacao =====
    print("\n[5/6] PROBABILIDADE DE MUTACAO")
    print("-" * 50)

    mutation_results = []
    for pm in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        config = baseline.copy()
        config['mutation_probability'] = pm

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Pm={pm}'
        elapsed = time.time() - start

        print_result(f"Pm={pm} ({elapsed:.1f}s)", result, optimal)
        mutation_results.append(result)

    all_results['groups']['mutation'] = mutation_results

    best_mut = min(mutation_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor mutacao: {best_mut['name']} (Melhor={best_mut['best']:.0f})")

    # ===== GRUPO 6: Probabilidade de Cruzamento =====
    print("\n[6/6] PROBABILIDADE DE CRUZAMENTO")
    print("-" * 50)

    crossover_results = []
    for pc in [0.70, 0.80, 0.85, 0.90]:
        config = baseline.copy()
        config['crossover_probability'] = pc

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = f'Pc={pc}'
        elapsed = time.time() - start

        print_result(f"Pc={pc} ({elapsed:.1f}s)", result, optimal)
        crossover_results.append(result)

    all_results['groups']['crossover'] = crossover_results

    best_cx = min(crossover_results, key=lambda x: x['best'])
    print(f"\n  >> Melhor cruzamento: {best_cx['name']} (Melhor={best_cx['best']:.0f})")

    # ===== RESUMO =====
    print("\n" + "=" * 80)
    print("RESUMO DA FASE 1 - MELHORES PARAMETROS POR GRUPO")
    print("=" * 80)

    summary = [
        ('Selecao', best_selection),
        ('Elitismo', best_elitism),
        ('Geracoes', best_gen),
        ('Populacao', best_pop),
        ('Mutacao', best_mut),
        ('Cruzamento', best_cx),
    ]

    print(f"\n{'Parametro':<15} {'Melhor Config':<25} {'Melhor':<10} {'Media':<10} {'Gap%':<10}")
    print("-" * 70)

    for param_name, result in summary:
        gap = ((result['best'] - optimal) / optimal * 100)
        print(f"{param_name:<15} {result['name']:<25} {result['best']:<10.0f} "
              f"{result['average']:<10.1f} {gap:<10.1f}%")

    # Identificar o melhor geral
    all_individual_results = (selection_results + elitism_results + generations_results +
                             population_results + mutation_results + crossover_results)
    overall_best = min(all_individual_results, key=lambda x: x['best'])

    print(f"\n{'='*80}")
    print(f"MELHOR GERAL DA FASE 1: {overall_best['name']}")
    print(f"  Melhor: {overall_best['best']:.0f}")
    print(f"  Media: {overall_best['average']:.1f}")
    print(f"  Gap: {((overall_best['best'] - optimal) / optimal * 100):.1f}%")
    print(f"{'='*80}")

    # Recomendacoes para a Fase 2
    all_results['recommendations'] = {
        'best_selection': best_selection['config']['selection_type'],
        'best_elitism': best_elitism['config']['elitism_count'],
        'best_generations': best_gen['config']['max_generations'],
        'best_population': best_pop['config']['population_size'],
        'best_mutation': best_mut['config']['mutation_probability'],
        'best_crossover': best_cx['config']['crossover_probability'],
        'overall_best': overall_best['name'],
        'overall_best_value': overall_best['best']
    }

    # Salvar resultados
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)

    all_results['execution_time_seconds'] = total_elapsed
    all_results['execution_time_formatted'] = f"{hours}h {minutes}m {seconds}s"
    all_results['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"cx2_phase1_{all_results['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

    latest_path = output_dir / "cx2_phase1_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")
    print(f"Tempo total de execucao: {hours}h {minutes}m {seconds}s")

    # Verificar criterios de sucesso
    print("\n" + "=" * 80)
    print("VERIFICACAO DOS CRITERIOS DE SUCESSO")
    print("=" * 80)

    baseline_best = 1382  # De execucoes anteriores
    if overall_best['best'] < baseline_best:
        improvement = ((baseline_best - overall_best['best']) / baseline_best * 100)
        print(f"[OK] Melhorou em relacao ao baseline (1382): {overall_best['best']:.0f} ({improvement:.1f}% de melhoria)")
    else:
        print(f"[!!] Nao melhorou em relacao ao baseline (1382)")

    if overall_best['best'] < 1000:
        print(f"[OK] Alcancou Melhor < 1000 (minimo aceitavel)")
    else:
        print(f"[!!] Nao alcancou Melhor < 1000 - MAIS EXPLORACAO NECESSARIA")

    if overall_best['best'] <= 699:
        print(f"[OK] Igualou ou superou o resultado do artigo (699)!")
    else:
        print(f"[..] Ainda acima do melhor do artigo (699)")

    print("\n>> Proximo passo: Executar a Fase 2 (run_cx2_phase2.py) para testar combinacoes")


if __name__ == "__main__":
    main()
