"""
Otimizacao CX2 - Fase 2: Combinacoes de Parametros

Este script testa combinacoes dos melhores parametros encontrados na Fase 1.
Ele le os resultados da Fase 1 para determinar automaticamente os melhores parametros individuais.

Tempo estimado de execucao: ~15-25 minutos
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
    print(f"  {name:<45} | Melhor: {result['best']:>6.0f} | "
          f"Media: {result['average']:>7.1f} | Gap: {gap:>6.1f}%")


def load_phase1_results():
    """Carrega os resultados da Fase 1, se disponiveis."""
    phase1_path = Path("results/cx2_phase1_latest.json")
    if phase1_path.exists():
        with open(phase1_path, 'r') as f:
            return json.load(f)
    return None


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("OTIMIZACAO CX2 - FASE 2: COMBINACOES DE PARAMETROS")
    print("=" * 80)
    print("\nEsta fase testa combinacoes dos melhores parametros da Fase 1.")
    print("Tempo estimado: 15-25 minutos\n")

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

    # Carregar resultados da Fase 1
    phase1 = load_phase1_results()
    if phase1:
        print("\n[INFO] Resultados da Fase 1 encontrados. Usando parametros recomendados.")
        rec = phase1.get('recommendations', {})
        best_sel = rec.get('best_selection', 'tournament')
        best_elite = rec.get('best_elitism', 15)
        best_gen = rec.get('best_generations', 1000)
        best_pop = rec.get('best_population', 300)
        best_mut = rec.get('best_mutation', 0.15)
        best_cx = rec.get('best_crossover', 0.80)
        print(f"  Melhor selecao: {best_sel}")
        print(f"  Melhor elitismo: {best_elite}")
        print(f"  Melhor geracoes: {best_gen}")
        print(f"  Melhor populacao: {best_pop}")
        print(f"  Melhor mutacao: {best_mut}")
        print(f"  Melhor cruzamento: {best_cx}")
    else:
        print("\n[AVISO] Resultados da Fase 1 nao encontrados. Usando estimativas padrao.")
        best_sel = 'tournament'
        best_elite = 15
        best_gen = 1000
        best_pop = 300
        best_mut = 0.15
        best_cx = 0.80

    print("-" * 80)

    # Definir combinacoes para testar
    combinations = [
        # Combinacao 1: Somente melhor selecao
        ('Somente Torneio', {
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        }),

        # Combinacao 2: Torneio + Elitismo Alto
        ('Torneio + Elite=15', {
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 15,
        }),

        # Combinacao 3: Torneio + Mais Geracoes
        ('Torneio + Ger=1000', {
            'population_size': 150,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        }),

        # Combinacao 4: Torneio + Populacao Maior
        ('Torneio + Pop=300', {
            'population_size': 300,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        }),

        # Combinacao 5: Torneio + Mutacao Mais Alta
        ('Torneio + Pm=0.15', {
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'tournament',
            'elitism_count': 2,
        }),

        # Combinacao 6: Todos os melhores da Fase 1
        ('Todos melhores params', {
            'population_size': best_pop,
            'max_generations': best_gen,
            'crossover_probability': best_cx,
            'mutation_probability': best_mut,
            'selection_type': best_sel,
            'elitism_count': best_elite,
        }),

        # Combinacao 7: Torneio + Elite + Ger
        ('Torneio + Elite=10 + Ger=1000', {
            'population_size': 150,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 10,
        }),

        # Combinacao 8: Torneio + Pop + Elite
        ('Torneio + Pop=300 + Elite=10', {
            'population_size': 300,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 10,
        }),

        # Combinacao 9: Agressivo - todas as melhorias
        ('Agressivo (todas melhorias)', {
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'tournament',
            'elitism_count': 15,
        }),

        # Combinacao 10: Elitismo alto somente (sem torneio)
        ('Roleta + Elite=20', {
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'roulette',
            'elitism_count': 20,
        }),

        # Combinacao 11: Muito agressivo
        ('Muito Agressivo', {
            'population_size': 400,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.12,
            'selection_type': 'tournament',
            'elitism_count': 15,
        }),

        # Combinacao 12: Equilibrado
        ('Equilibrado (Pop=250, Ger=750)', {
            'population_size': 250,
            'max_generations': 750,
            'crossover_probability': 0.80,
            'mutation_probability': 0.12,
            'selection_type': 'tournament',
            'elitism_count': 10,
        }),
    ]

    all_results = {
        'instance': 'dantzig42',
        'optimal': optimal,
        'paper_cx2': {'best': paper_best, 'avg': paper_avg},
        'num_runs': num_runs,
        'combinations': []
    }

    # Executar todas as combinacoes
    print("\n" + "=" * 80)
    print("EXECUTANDO EXPERIMENTOS DE COMBINACAO")
    print("=" * 80 + "\n")

    for i, (name, config) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] {name}")

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = name
        elapsed = time.time() - start

        print_result(f"  Resultado ({elapsed:.1f}s)", result, optimal)
        all_results['combinations'].append(result)

    # Ordenar pelo melhor valor
    sorted_results = sorted(all_results['combinations'], key=lambda x: x['best'])

    # Imprimir ranking
    print("\n" + "=" * 80)
    print("RESULTADOS DA FASE 2 - CLASSIFICADOS POR MELHOR VALOR")
    print("=" * 80)
    print(f"\n{'Pos':<5} {'Configuracao':<45} {'Melhor':<8} {'Media':<10} {'Gap%':<8}")
    print("-" * 80)

    for i, result in enumerate(sorted_results):
        gap = ((result['best'] - optimal) / optimal * 100)
        marker = " ***" if result['best'] <= paper_best else ""
        print(f"{i+1:<5} {result['name']:<45} {result['best']:<8.0f} "
              f"{result['average']:<10.1f} {gap:<8.1f}%{marker}")

    # Melhor resultado
    best_result = sorted_results[0]

    print(f"\n{'='*80}")
    print(f"MELHOR CONFIGURACAO DA FASE 2: {best_result['name']}")
    print(f"{'='*80}")
    print(f"  Melhor: {best_result['best']:.0f}")
    print(f"  Media: {best_result['average']:.1f}")
    print(f"  Pior: {best_result['worst']:.0f}")
    print(f"  Desvio Padrao: {best_result['std']:.1f}")
    print(f"  Gap: {((best_result['best'] - optimal) / optimal * 100):.1f}%")
    print(f"\n  Configuracao: {best_result['config']}")

    # Top 5 para a Fase 3
    top5 = sorted_results[:5]
    all_results['top5_for_phase3'] = [
        {'name': r['name'], 'config': r['config'], 'best': r['best']}
        for r in top5
    ]

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

    filepath = output_dir / f"cx2_phase2_{all_results['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

    latest_path = output_dir / "cx2_phase2_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")
    print(f"Tempo total de execucao: {hours}h {minutes}m {seconds}s")

    # Verificacao dos criterios de sucesso
    print("\n" + "=" * 80)
    print("VERIFICACAO DOS CRITERIOS DE SUCESSO")
    print("=" * 80)

    baseline_best = 1382
    if best_result['best'] < baseline_best:
        improvement = ((baseline_best - best_result['best']) / baseline_best * 100)
        print(f"[OK] Melhorou em relacao ao baseline (1382): {best_result['best']:.0f} ({improvement:.1f}% de melhoria)")
    else:
        print(f"[!!] Nao melhorou em relacao ao baseline (1382)")

    if best_result['best'] < 1000:
        print(f"[OK] Alcancou Melhor < 1000 (minimo aceitavel)")
    else:
        print(f"[!!] Nao alcancou Melhor < 1000 - MAIS EXPLORACAO NECESSARIA")

    if best_result['best'] < 800:
        print(f"[OK] Alcancou Melhor < 800 (bom resultado)")
    else:
        print(f"[..] Melhor >= 800")

    if best_result['best'] <= paper_best:
        print(f"[OK] Igualou ou superou o resultado do artigo ({paper_best})!")
    else:
        print(f"[..] Ainda acima do melhor do artigo ({paper_best})")

    print(f"\n{'='*80}")
    print("TOP 5 CONFIGURACOES PARA VALIDACAO NA FASE 3")
    print("=" * 80)
    for i, r in enumerate(top5):
        print(f"  {i+1}. {r['name']} (Melhor={r['best']:.0f})")

    print("\n>> Proximo passo: Execute a Fase 3 (run_cx2_phase3.py) para validar com 30 execucoes")


if __name__ == "__main__":
    main()
