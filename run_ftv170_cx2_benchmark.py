"""
Benchmark ftv170 - CX2 Only
Comparacao direta com os resultados do artigo para o operador CX2

Metodologia em 3 fases:
- Fase 1: Analise de Sensibilidade (29 configs, 10 runs cada)
- Fase 2: Combinacoes Promissoras (12 configs, 10 runs cada)
- Fase 3: Validacao Final (5 melhores configs, 30 runs cada)

Artigo CX2 para ftv170 (Tabela 5 - instancias >100 cidades):
- Best: 6421
- Worst: 8416
- Average: 7019

Parametros do artigo para >100 cidades:
- Population: 200
- Generations: 1000

Valor otimo: 2755
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


PAPER_CX2 = {'best': 6421, 'worst': 8416, 'average': 7019}
OPTIMAL_VALUE = 2755


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
            crossover_type='CX2',  # SEMPRE CX2
            mutation_type=config.get('mutation_type', 'swap'),
            selection_type=config.get('selection_type', 'tournament'),
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
        'median': float(np.median(all_values)),
        'q1': float(np.percentile(all_values, 25)),
        'q3': float(np.percentile(all_values, 75)),
        'all_values': [float(v) for v in all_values],
        'best_tour': [int(c) for c in best_tour] if best_tour else None,
        'config': config,
        'num_runs': num_runs
    }


def phase1_sensitivity_analysis(fitness_calc):
    """Fase 1: Analise de sensibilidade de cada parametro."""
    print("\n" + "=" * 80)
    print("FASE 1: ANALISE DE SENSIBILIDADE (CX2)")
    print("=" * 80)

    # Baseline (parametros do artigo para >100 cidades com torneio)
    baseline = {
        'population_size': 200,
        'max_generations': 1000,
        'crossover_probability': 0.80,
        'mutation_probability': 0.10,
        'selection_type': 'tournament',
        'elitism_count': 2,
    }

    configs = []

    # 1. Tipos de selecao
    for sel in ['roulette', 'tournament', 'rank']:
        cfg = baseline.copy()
        cfg['selection_type'] = sel
        cfg['name'] = f'Selection={sel}'
        configs.append(cfg)

    # 2. Tamanhos de populacao
    for pop in [100, 150, 200, 300, 400]:
        cfg = baseline.copy()
        cfg['population_size'] = pop
        cfg['name'] = f'Pop={pop}'
        configs.append(cfg)

    # 3. Numero de geracoes
    for gen in [500, 750, 1000, 1500, 2000, 2500]:
        cfg = baseline.copy()
        cfg['max_generations'] = gen
        cfg['name'] = f'Gen={gen}'
        configs.append(cfg)

    # 4. Probabilidade de crossover
    for pc in [0.60, 0.70, 0.80, 0.90]:
        cfg = baseline.copy()
        cfg['crossover_probability'] = pc
        cfg['name'] = f'Pc={pc}'
        configs.append(cfg)

    # 5. Probabilidade de mutacao
    for pm in [0.05, 0.10, 0.15, 0.20, 0.30]:
        cfg = baseline.copy()
        cfg['mutation_probability'] = pm
        cfg['name'] = f'Pm={pm}'
        configs.append(cfg)

    # 6. Elitismo
    for elite in [0, 2, 5, 10, 15, 20]:
        cfg = baseline.copy()
        cfg['elitism_count'] = elite
        cfg['name'] = f'Elite={elite}'
        configs.append(cfg)

    print(f"Total de configuracoes: {len(configs)}")
    print(f"Runs por configuracao: 10")

    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testando: {config['name']}")
        result = run_experiment(fitness_calc, config, num_runs=10)
        result['name'] = config['name']
        gap = ((result['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"    Best: {result['best']:.0f} | Avg: {result['average']:.1f} | Gap: {gap:.1f}%")
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nFase 1 concluida em {elapsed/60:.1f} minutos")

    return results, elapsed


def phase2_combinations(fitness_calc, phase1_results):
    """Fase 2: Testar combinacoes dos melhores parametros CX2."""
    print("\n" + "=" * 80)
    print("FASE 2: COMBINACOES PROMISSORAS (CX2)")
    print("=" * 80)

    # Analisa fase 1 para encontrar melhores parametros
    best_by_group = {}
    for r in phase1_results:
        name = r['name']
        if '=' in name:
            group = name.split('=')[0]
            if group not in best_by_group or r['best'] < best_by_group[group]['best']:
                best_by_group[group] = r

    print("\nMelhores por grupo na Fase 1:")
    for group, r in best_by_group.items():
        print(f"  {group}: {r['name']} (Best={r['best']:.0f})")

    configs = [
        {
            'name': 'Baseline (Tournament)',
            'population_size': 200,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        },
        {
            'name': 'Tournament + Gen=1500',
            'population_size': 200,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        },
        {
            'name': 'Tournament + Gen=2000',
            'population_size': 200,
            'max_generations': 2000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
        },
        {
            'name': 'Tournament + Gen=1500 + Pop=300',
            'population_size': 300,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 5,
        },
        {
            'name': 'Tournament + Elite=15',
            'population_size': 200,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 15,
        },
        {
            'name': 'Balanced (Pop=300, Gen=1500)',
            'population_size': 300,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.12,
            'selection_type': 'tournament',
            'elitism_count': 10,
        },
        {
            'name': 'Aggressive (Pop=400, Gen=2500)',
            'population_size': 400,
            'max_generations': 2500,
            'crossover_probability': 0.70,
            'mutation_probability': 0.20,
            'selection_type': 'tournament',
            'elitism_count': 15,
        },
        {
            'name': 'Tournament + Pop=300 + Elite=10',
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 10,
        },
        {
            'name': 'High Gen (Gen=2000, Elite=10)',
            'population_size': 250,
            'max_generations': 2000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'tournament',
            'elitism_count': 10,
        },
        {
            'name': 'Large Pop (Pop=400, Gen=1500)',
            'population_size': 400,
            'max_generations': 1500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 10,
        },
        {
            'name': 'Optimized (all best params)',
            'population_size': 350,
            'max_generations': 2000,
            'crossover_probability': 0.75,
            'mutation_probability': 0.12,
            'selection_type': 'tournament',
            'elitism_count': 12,
        },
        {
            'name': 'Max Resources (Pop=500, Gen=3000)',
            'population_size': 500,
            'max_generations': 3000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'tournament',
            'elitism_count': 20,
        },
    ]

    print(f"\nTotal de configuracoes: {len(configs)}")

    results = []
    start_time = time.time()

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testando: {config['name']}")
        result = run_experiment(fitness_calc, config, num_runs=10)
        result['name'] = config['name']
        gap = ((result['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"    Best: {result['best']:.0f} | Avg: {result['average']:.1f} | Gap: {gap:.1f}%")
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nFase 2 concluida em {elapsed/60:.1f} minutos")

    results_sorted = sorted(results, key=lambda x: x['best'])
    print("\nTop 5 da Fase 2:")
    for i, r in enumerate(results_sorted[:5]):
        gap = ((r['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"  {i+1}. {r['name']}: Best={r['best']:.0f}, Gap={gap:.1f}%")

    return results, elapsed


def phase3_validation(fitness_calc, phase2_results):
    """Fase 3: Validacao com 30 runs nas melhores configs CX2."""
    print("\n" + "=" * 80)
    print("FASE 3: VALIDACAO FINAL CX2 (30 runs)")
    print("=" * 80)

    sorted_results = sorted(phase2_results, key=lambda x: x['best'])
    top5_configs = [r['config'] for r in sorted_results[:5]]
    top5_names = [r['name'] for r in sorted_results[:5]]

    print("\nConfiguracoes para validacao:")
    for i, name in enumerate(top5_names):
        print(f"  {i+1}. {name}")

    results = []
    start_time = time.time()

    for i, (config, name) in enumerate(zip(top5_configs, top5_names)):
        print(f"\n[{i+1}/5] Validando: {name}")
        print(f"    Executando 30 runs independentes...")

        result = run_experiment(fitness_calc, config, num_runs=30)
        result['name'] = name
        gap = ((result['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)

        print(f"    Best: {result['best']:.0f} | Avg: {result['average']:.1f} | "
              f"Worst: {result['worst']:.0f} | Gap: {gap:.1f}%")
        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nFase 3 concluida em {elapsed/60:.1f} minutos")

    return results, elapsed


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("BENCHMARK FTV170 - CX2 ONLY (3 FASES)")
    print("=" * 80)
    print("\nComparando nossa implementacao CX2 com o artigo")
    print(f"Artigo CX2: Best={PAPER_CX2['best']}, Avg={PAPER_CX2['average']}")
    print("\nATENCAO: Instancia grande (171 cidades) - execucao demorada!")

    parser = TSPLibParser()
    instance_path = Path("tsplib/instances/ftv170.atsp")

    if not instance_path.exists():
        print(f"Erro: {instance_path} nao encontrado")
        return

    instance = parser.parse(str(instance_path))
    distance_matrix = instance.get_distance_matrix()
    fitness_calc = FitnessCalculator(distance_matrix)

    print(f"\nInstancia: ftv170 ({instance.dimension} cidades)")
    print(f"Valor otimo: {OPTIMAL_VALUE}")

    # Executar as 3 fases
    phase1_results, time1 = phase1_sensitivity_analysis(fitness_calc)
    phase2_results, time2 = phase2_combinations(fitness_calc, phase1_results)
    phase3_results, time3 = phase3_validation(fitness_calc, phase2_results)

    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    execution_time_formatted = f"{hours}h {minutes}m {seconds}s"

    # Salvar resultados
    output_data = {
        'instance': 'ftv170',
        'optimal': OPTIMAL_VALUE,
        'paper_cx2': PAPER_CX2,
        'phase1': {
            'results': phase1_results,
            'execution_time_seconds': time1
        },
        'phase2': {
            'results': phase2_results,
            'execution_time_seconds': time2
        },
        'phase3': {
            'results': phase3_results,
            'execution_time_seconds': time3
        },
        'best_config': sorted(phase3_results, key=lambda x: x['best'])[0],
        'execution_time_seconds': total_elapsed,
        'execution_time_formatted': execution_time_formatted,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"ftv170_cx2_benchmark_{output_data['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    latest_path = output_dir / "ftv170_cx2_benchmark_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Resumo final
    print("\n" + "=" * 80)
    print("RESUMO FINAL - CX2")
    print("=" * 80)

    best = sorted(phase3_results, key=lambda x: x['best'])[0]
    gap = ((best['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
    paper_gap = ((PAPER_CX2['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)

    print(f"\nMelhor configuracao CX2: {best['name']}")
    print(f"\n{'Metrica':<12} {'Nossa Impl.':<15} {'Artigo':<15} {'Melhoria':<15}")
    print("-" * 57)
    print(f"{'Best':<12} {best['best']:<15.0f} {PAPER_CX2['best']:<15} {((PAPER_CX2['best']-best['best'])/PAPER_CX2['best']*100):.1f}%")
    print(f"{'Average':<12} {best['average']:<15.1f} {PAPER_CX2['average']:<15} {((PAPER_CX2['average']-best['average'])/PAPER_CX2['average']*100):.1f}%")
    print(f"{'Worst':<12} {best['worst']:<15.0f} {PAPER_CX2['worst']:<15}")
    print(f"{'Gap%':<12} {gap:<15.1f}% {paper_gap:<15.1f}%")

    print(f"\nTempo total: {execution_time_formatted}")

    # Tabela para apresentacao
    print("\n" + "=" * 80)
    print("TABELA MARKDOWN PARA APRESENTACAO")
    print("=" * 80)

    print(f"""
## Fase 3: Validacao Final CX2 (30 runs)

| Rank | Configuracao | Best | Average | Worst | Std | Gap% |
|------|--------------|------|---------|-------|-----|------|""")

    for i, r in enumerate(sorted(phase3_results, key=lambda x: x['best'])):
        g = ((r['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"| {i+1} | {r['name']} | {r['best']:.0f} | {r['average']:.1f} | {r['worst']:.0f} | {r['std']:.1f} | {g:.1f}% |")

    print(f"""
## Comparacao com o Artigo (CX2)

| Metrica | Nossa Impl. | Artigo | Melhoria |
|---------|-------------|--------|----------|
| Best | {best['best']:.0f} | {PAPER_CX2['best']} | {((PAPER_CX2['best']-best['best'])/PAPER_CX2['best']*100):.1f}% |
| Average | {best['average']:.1f} | {PAPER_CX2['average']} | {((PAPER_CX2['average']-best['average'])/PAPER_CX2['average']*100):.1f}% |

## Tempo de Execucao

| Fase | Tempo |
|------|-------|
| Fase 1 (Sensibilidade) | {time1/60:.1f} min |
| Fase 2 (Combinacoes) | {time2/60:.1f} min |
| Fase 3 (Validacao) | {time3/60:.1f} min |
| **Total** | **{total_elapsed/60:.1f} min** |
""")

    print("\n" + "=" * 80)
    print("BENCHMARK CX2 FINALIZADO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
