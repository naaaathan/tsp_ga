"""
Script de Otimizacao de Parametros do CX2

Este script tenta melhorar o desempenho do CX2 testando diferentes combinacoes de parametros.
Objetivo: Alcancar resultados proximos ou melhores do que os reportados em Hussain et al., 2017

Resultados do CX2 no artigo para dantzig42:
- Melhor: 699 (otimo!)
- Pior: 920
- Media: 802
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import product
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
        'best': min(all_values),
        'worst': max(all_values),
        'average': np.mean(all_values),
        'std': np.std(all_values),
        'all_values': all_values,
        'best_tour': best_tour,
        'config': config
    }


def print_result(result, optimal, config_name):
    """Imprime um unico resultado."""
    gap = ((result['best'] - optimal) / optimal * 100)
    print(f"  {config_name:<40} | Melhor: {result['best']:>6.0f} | "
          f"Media: {result['average']:>7.1f} | Desvio: {gap:>6.1f}%")


def main():
    total_start_time = time.time()

    print("="*80)
    print("OTIMIZACAO DE PARAMETROS DO CX2")
    print("Objetivo: Melhorar o desempenho do CX2 para igualar os resultados do artigo")
    print("="*80)

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

    print(f"\nInstancia: dantzig42 ({instance.dimension} cidades)")
    print(f"Otimo: {optimal}")
    print(f"Melhor CX2 do artigo: {paper_best}, Media: {paper_avg}")
    print("-"*80)

    # Definir variacoes de parametros para testar
    experiments = []

    # ===== GRUPO DE EXPERIMENTOS 1: Linha de base vs parametros do artigo =====
    print("\n[GRUPO 1] Testando parametros de linha de base...")

    # Linha de base (parametros do artigo)
    baseline = {
        'population_size': 150,
        'max_generations': 500,
        'crossover_probability': 0.80,
        'mutation_probability': 0.10,
        'selection_type': 'roulette',
        'elitism_count': 2,
    }
    experiments.append(('Linha de base (params do artigo)', baseline))

    # ===== GRUPO DE EXPERIMENTOS 2: Tamanho da populacao =====
    print("\n[GRUPO 2] Testando variacoes no tamanho da populacao...")

    for pop_size in [200, 250, 300, 400, 500]:
        config = baseline.copy()
        config['population_size'] = pop_size
        experiments.append((f'Pop={pop_size}', config))

    # ===== GRUPO DE EXPERIMENTOS 3: Geracoes =====
    print("\n[GRUPO 3] Testando variacoes no numero de geracoes...")

    for max_gen in [750, 1000, 1500, 2000]:
        config = baseline.copy()
        config['max_generations'] = max_gen
        experiments.append((f'Gen={max_gen}', config))

    # ===== GRUPO DE EXPERIMENTOS 4: Probabilidade de cruzamento =====
    print("\n[GRUPO 4] Testando variacoes na probabilidade de cruzamento...")

    for cx_prob in [0.70, 0.85, 0.90, 0.95, 1.0]:
        config = baseline.copy()
        config['crossover_probability'] = cx_prob
        experiments.append((f'Pc={cx_prob}', config))

    # ===== GRUPO DE EXPERIMENTOS 5: Probabilidade de mutacao =====
    print("\n[GRUPO 5] Testando variacoes na probabilidade de mutacao...")

    for mut_prob in [0.05, 0.15, 0.20, 0.25, 0.30]:
        config = baseline.copy()
        config['mutation_probability'] = mut_prob
        experiments.append((f'Pm={mut_prob}', config))

    # ===== GRUPO DE EXPERIMENTOS 6: Elitismo =====
    print("\n[GRUPO 6] Testando variacoes no elitismo...")

    for elite in [0, 5, 10, 15, 20]:
        config = baseline.copy()
        config['elitism_count'] = elite
        experiments.append((f'Elite={elite}', config))

    # ===== GRUPO DE EXPERIMENTOS 7: Tipo de selecao =====
    print("\n[GRUPO 7] Testando variacoes no tipo de selecao...")

    for sel_type in ['tournament', 'rank']:
        config = baseline.copy()
        config['selection_type'] = sel_type
        experiments.append((f'Sel={sel_type}', config))

    # ===== GRUPO DE EXPERIMENTOS 8: Melhorias combinadas =====
    print("\n[GRUPO 8] Testando melhorias combinadas de parametros...")

    combined_configs = [
        ('Pop=300, Gen=1000', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 2,
        }),
        ('Pop=300, Gen=1000, Elite=10', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 10,
        }),
        ('Pop=400, Gen=1500, Elite=10', {
            'population_size': 400,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 10,
        }),
        ('Pop=300, Gen=1000, Pm=0.15', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'roulette',
            'elitism_count': 5,
        }),
        ('Pop=300, Gen=1000, Pc=0.90', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.90,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 5,
        }),
        ('Torneio, Pop=300, Gen=1000', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 5,
        }),
        ('Pop=500, Gen=2000, Elite=15', {
            'population_size': 500,
            'max_generations': 2000,
            'crossover_probability': 0.85,
            'mutation_probability': 0.12,
            'selection_type': 'roulette',
            'elitism_count': 15,
        }),
        ('Alta diversidade: Pm=0.20, Pop=400', {
            'population_size': 400,
            'max_generations': 1000,
            'crossover_probability': 0.75,
            'mutation_probability': 0.20,
            'selection_type': 'roulette',
            'elitism_count': 5,
        }),
        ('Agressivo: Pc=0.95, Gen=1500', {
            'population_size': 300,
            'max_generations': 1500,
            'crossover_probability': 0.95,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 10,
        }),
    ]
    experiments.extend(combined_configs)

    # Executar todos os experimentos
    print("\n" + "="*80)
    print("EXECUTANDO EXPERIMENTOS")
    print("="*80)

    results = []
    num_runs = 10  # Execucoes rapidas para busca de parametros

    for i, (name, config) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        start_time = time.time()

        result = run_experiment(fitness_calc, config, num_runs=num_runs)
        result['name'] = name
        results.append(result)

        elapsed = time.time() - start_time
        print_result(result, optimal, f"  Resultado ({elapsed:.1f}s)")

    # Ordenar resultados pelo melhor valor
    results_sorted = sorted(results, key=lambda x: x['best'])

    # Imprimir os 10 melhores resultados
    print("\n" + "="*80)
    print("10 MELHORES CONFIGURACOES (por melhor valor)")
    print("="*80)
    print(f"{'Pos':<5} {'Configuracao':<45} {'Melhor':<8} {'Media':<10} {'Desvio%':<8}")
    print("-"*80)

    for i, result in enumerate(results_sorted[:10]):
        gap = ((result['best'] - optimal) / optimal * 100)
        print(f"{i+1:<5} {result['name']:<45} {result['best']:<8.0f} "
              f"{result['average']:<10.1f} {gap:<8.1f}%")

    # Imprimir comparacao com o artigo
    print("\n" + "="*80)
    print("COMPARACAO COM O ARTIGO")
    print("="*80)

    best_result = results_sorted[0]
    print(f"\nResultados do CX2 no artigo:")
    print(f"  Melhor: {paper_best}, Media: {paper_avg}")

    print(f"\nNossa melhor configuracao: {best_result['name']}")
    print(f"  Melhor: {best_result['best']:.0f}, Media: {best_result['average']:.1f}")

    if best_result['best'] <= paper_best:
        print("\n*** SUCESSO: Igualamos ou superamos o melhor resultado do artigo! ***")
    else:
        diff = best_result['best'] - paper_best
        print(f"\n  Diferenca em relacao ao melhor do artigo: +{diff:.0f}")

    # Executar experimentos estendidos na melhor configuracao
    print("\n" + "="*80)
    print("EXECUCAO ESTENDIDA NA MELHOR CONFIGURACAO (30 execucoes)")
    print("="*80)

    best_config = best_result['config']
    print(f"Configuracao: {best_result['name']}")
    print(f"Parametros: {best_config}")

    extended_result = run_experiment(fitness_calc, best_config, num_runs=30)

    print(f"\nResultados estendidos (30 execucoes):")
    print(f"  Melhor:       {extended_result['best']:.0f}")
    print(f"  Pior:         {extended_result['worst']:.0f}")
    print(f"  Media:        {extended_result['average']:.1f}")
    print(f"  Desvio pad.:  {extended_result['std']:.1f}")

    gap = ((extended_result['best'] - optimal) / optimal * 100)
    print(f"  Desvio do otimo: {gap:.1f}%")

    # Salvar resultados
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Converter tipos do numpy
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

    results_to_save = {
        'instance': 'dantzig42',
        'optimal': optimal,
        'paper_cx2': {'best': paper_best, 'avg': paper_avg},
        'experiments': [convert(r) for r in results],
        'best_config': convert(best_result),
        'extended_result': convert(extended_result),
        'timestamp': timestamp
    }

    filepath = output_dir / f'cx2_optimization_{timestamp}.json'
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Salvar tambem como mais recente
    latest_path = output_dir / 'cx2_optimization_latest.json'
    with open(latest_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Calcular tempo total de execucao
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)

    print("\n" + "="*80)
    print("OTIMIZACAO CONCLUIDA")
    print(f"Tempo total de execucao: {hours}h {minutes}m {seconds}s ({total_elapsed:.1f} segundos)")
    print("="*80)

    # Salvar tempo de execucao nos resultados
    results_to_save['total_execution_time_seconds'] = total_elapsed
    results_to_save['total_execution_time_formatted'] = f"{hours}h {minutes}m {seconds}s"

    # Atualizar arquivos salvos com o tempo de execucao
    with open(filepath, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    with open(latest_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)


if __name__ == "__main__":
    main()
