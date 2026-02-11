"""
Benchmark Completo ft53 - Seguindo o mesmo padrao do dantzig42

Este script executa o benchmark completo em 3 fases:
- Fase 1: Analise de Sensibilidade (29 configs, 10 runs cada)
- Fase 2: Combinacoes Promissoras (12 configs, 10 runs cada)
- Fase 3: Validacao Final (5 melhores configs, 30 runs cada)

Tempo estimado: 1-2 horas

Instancia: ft53 (53 cidades, ATSP)
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
            crossover_type=config.get('crossover_type', 'CX2'),
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
    print("FASE 1: ANALISE DE SENSIBILIDADE")
    print("=" * 80)

    # Baseline (parametros do artigo com torneio)
    baseline = {
        'population_size': 150,
        'max_generations': 500,
        'crossover_probability': 0.80,
        'mutation_probability': 0.10,
        'selection_type': 'tournament',
        'elitism_count': 2,
        'crossover_type': 'CX2'
    }

    configs = []

    # 1. Tipos de selecao
    for sel in ['roulette', 'tournament', 'rank']:
        cfg = baseline.copy()
        cfg['selection_type'] = sel
        cfg['name'] = f'Selection={sel}'
        configs.append(cfg)

    # 2. Tamanhos de populacao
    for pop in [100, 150, 200, 300, 500]:
        cfg = baseline.copy()
        cfg['population_size'] = pop
        cfg['name'] = f'Pop={pop}'
        configs.append(cfg)

    # 3. Numero de geracoes
    for gen in [250, 500, 750, 1000, 1500, 2000]:
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
    print(f"Total de execucoes: {len(configs) * 10}")

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
    """Fase 2: Testar combinacoes dos melhores parametros."""
    print("\n" + "=" * 80)
    print("FASE 2: COMBINACOES PROMISSORAS")
    print("=" * 80)

    # Analisa fase 1 para encontrar melhores parametros por grupo
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

    # Configuracoes combinadas
    configs = [
        # Baseline com torneio
        {
            'name': 'Baseline (Tournament)',
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
            'crossover_type': 'CX2'
        },
        # Tournament + mais geracoes
        {
            'name': 'Tournament + Gen=1000',
            'population_size': 150,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
            'crossover_type': 'CX2'
        },
        # Tournament + mais geracoes + mais populacao
        {
            'name': 'Tournament + Gen=1000 + Pop=300',
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 5,
            'crossover_type': 'CX2'
        },
        # Tournament + elite alto
        {
            'name': 'Tournament + Elite=15',
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 15,
            'crossover_type': 'CX2'
        },
        # Tournament + mutacao alta
        {
            'name': 'Tournament + Pm=0.20',
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.20,
            'selection_type': 'tournament',
            'elitism_count': 2,
            'crossover_type': 'CX2'
        },
        # Configuracao balanceada
        {
            'name': 'Balanced (Pop=250, Gen=750)',
            'population_size': 250,
            'max_generations': 750,
            'crossover_probability': 0.80,
            'mutation_probability': 0.12,
            'selection_type': 'tournament',
            'elitism_count': 5,
            'crossover_type': 'CX2'
        },
        # Configuracao agressiva
        {
            'name': 'Aggressive (Pop=500, Gen=2000)',
            'population_size': 500,
            'max_generations': 2000,
            'crossover_probability': 0.70,
            'mutation_probability': 0.30,
            'selection_type': 'tournament',
            'elitism_count': 15,
            'crossover_type': 'CX2'
        },
        # Tournament + Pop grande + Elite alto
        {
            'name': 'Tournament + Pop=300 + Elite=10',
            'population_size': 300,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 10,
            'crossover_type': 'CX2'
        },
        # Exploratorio: mais mutacao + mais geracoes
        {
            'name': 'Exploration (Pm=0.25, Gen=1500)',
            'population_size': 200,
            'max_generations': 1500,
            'crossover_probability': 0.75,
            'mutation_probability': 0.25,
            'selection_type': 'tournament',
            'elitism_count': 10,
            'crossover_type': 'CX2'
        },
        # PMX com torneio (para comparacao)
        {
            'name': 'PMX + Tournament',
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
            'crossover_type': 'PMX'
        },
        # OX com torneio (para comparacao)
        {
            'name': 'OX + Tournament',
            'population_size': 150,
            'max_generations': 500,
            'crossover_probability': 0.80,
            'mutation_probability': 0.10,
            'selection_type': 'tournament',
            'elitism_count': 2,
            'crossover_type': 'OX'
        },
        # OX otimizado (melhor operador no benchmark anterior)
        {
            'name': 'OX Optimized',
            'population_size': 300,
            'max_generations': 1000,
            'crossover_probability': 0.80,
            'mutation_probability': 0.15,
            'selection_type': 'tournament',
            'elitism_count': 10,
            'crossover_type': 'OX'
        },
    ]

    print(f"\nTotal de configuracoes: {len(configs)}")
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
    print(f"\nFase 2 concluida em {elapsed/60:.1f} minutos")

    # Ordena por melhor resultado
    results_sorted = sorted(results, key=lambda x: x['best'])

    print("\nTop 5 da Fase 2:")
    for i, r in enumerate(results_sorted[:5]):
        gap = ((r['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        print(f"  {i+1}. {r['name']}: Best={r['best']:.0f}, Avg={r['average']:.1f}, Gap={gap:.1f}%")

    return results, elapsed


def phase3_validation(fitness_calc, phase2_results):
    """Fase 3: Validacao com 30 runs nas melhores configs."""
    print("\n" + "=" * 80)
    print("FASE 3: VALIDACAO FINAL (30 runs)")
    print("=" * 80)

    # Ordena fase 2 e pega top 5
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


def generate_markdown_report(all_results, execution_times):
    """Gera relatorio em markdown para a apresentacao."""

    phase1, phase2, phase3 = all_results
    time1, time2, time3 = execution_times
    total_time = time1 + time2 + time3

    # Ordena fase 3 pelo melhor
    phase3_sorted = sorted(phase3, key=lambda x: x['best'])
    best_result = phase3_sorted[0]

    report = f"""
# Benchmark ft53 - Resultados Completos

## Informacoes da Instancia

| Propriedade | Valor |
|-------------|-------|
| Nome | ft53 |
| Tipo | ATSP (assimetrico) |
| Cidades | 53 |
| Valor Otimo | {OPTIMAL_VALUE} |

## Resultados do Artigo (Hussain et al., 2017)

| Operador | Best | Average | Worst |
|----------|------|---------|-------|
| PMX | {PAPER_RESULTS['PMX']['best']} | {PAPER_RESULTS['PMX']['average']} | {PAPER_RESULTS['PMX']['worst']} |
| OX | {PAPER_RESULTS['OX']['best']} | {PAPER_RESULTS['OX']['average']} | {PAPER_RESULTS['OX']['worst']} |
| CX2 | {PAPER_RESULTS['CX2']['best']} | {PAPER_RESULTS['CX2']['average']} | {PAPER_RESULTS['CX2']['worst']} |

---

## Fase 1: Analise de Sensibilidade

| Grupo | Melhor Config | Best | Average | Gap% |
|-------|---------------|------|---------|------|
"""

    # Agrupa fase 1 por tipo de parametro
    groups = {}
    for r in phase1:
        name = r['name']
        if '=' in name:
            group = name.split('=')[0]
            if group not in groups:
                groups[group] = []
            groups[group].append(r)

    for group, results in groups.items():
        best = min(results, key=lambda x: x['best'])
        gap = ((best['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        report += f"| {group} | {best['name']} | {best['best']:.0f} | {best['average']:.1f} | {gap:.1f}% |\n"

    report += f"""
**Tempo de execucao Fase 1:** {time1/60:.1f} minutos

---

## Fase 2: Combinacoes Promissoras

| Rank | Configuracao | Best | Average | Gap% |
|------|--------------|------|---------|------|
"""

    phase2_sorted = sorted(phase2, key=lambda x: x['best'])
    for i, r in enumerate(phase2_sorted[:10]):
        gap = ((r['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        report += f"| {i+1} | {r['name']} | {r['best']:.0f} | {r['average']:.1f} | {gap:.1f}% |\n"

    report += f"""
**Tempo de execucao Fase 2:** {time2/60:.1f} minutos

---

## Fase 3: Validacao Final (30 runs)

| Rank | Configuracao | Best | Average | Worst | Std | Gap% |
|------|--------------|------|---------|-------|-----|------|
"""

    for i, r in enumerate(phase3_sorted):
        gap = ((r['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)
        report += f"| {i+1} | {r['name']} | {r['best']:.0f} | {r['average']:.1f} | {r['worst']:.0f} | {r['std']:.1f} | {gap:.1f}% |\n"

    report += f"""
**Tempo de execucao Fase 3:** {time3/60:.1f} minutos

---

## Melhor Configuracao Encontrada

**Nome:** {best_result['name']}

**Parametros:**
```
population_size: {best_result['config']['population_size']}
max_generations: {best_result['config']['max_generations']}
crossover_probability: {best_result['config']['crossover_probability']}
mutation_probability: {best_result['config']['mutation_probability']}
selection_type: {best_result['config']['selection_type']}
elitism_count: {best_result['config']['elitism_count']}
crossover_type: {best_result['config'].get('crossover_type', 'CX2')}
```

**Estatisticas (30 runs):**

| Metrica | Valor |
|---------|-------|
| Best | {best_result['best']:.0f} |
| Worst | {best_result['worst']:.0f} |
| Average | {best_result['average']:.1f} |
| Std | {best_result['std']:.1f} |
| Median | {best_result['median']:.1f} |
| Q1 (25%) | {best_result['q1']:.1f} |
| Q3 (75%) | {best_result['q3']:.1f} |
| Gap% | {((best_result['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100):.1f}% |

---

## Comparacao com o Artigo

| Metrica | Nossa Impl. | Artigo (CX2) | Melhoria |
|---------|-------------|--------------|----------|
| Best | {best_result['best']:.0f} | {PAPER_RESULTS['CX2']['best']} | {((PAPER_RESULTS['CX2']['best'] - best_result['best']) / PAPER_RESULTS['CX2']['best'] * 100):.1f}% |
| Average | {best_result['average']:.1f} | {PAPER_RESULTS['CX2']['average']} | {((PAPER_RESULTS['CX2']['average'] - best_result['average']) / PAPER_RESULTS['CX2']['average'] * 100):.1f}% |

---

## Tempo Total de Execucao

| Fase | Tempo |
|------|-------|
| Fase 1 (Sensibilidade) | {time1/60:.1f} min |
| Fase 2 (Combinacoes) | {time2/60:.1f} min |
| Fase 3 (Validacao) | {time3/60:.1f} min |
| **Total** | **{total_time/60:.1f} min** |

---

*Benchmark executado em {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    return report


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("BENCHMARK COMPLETO FT53 - 3 FASES")
    print("=" * 80)
    print("\nTempo estimado: 1-2 horas")
    print("NAO INTERROMPA A EXECUCAO!")

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
        'instance': 'ft53',
        'optimal': OPTIMAL_VALUE,
        'paper_results': PAPER_RESULTS,
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

    filepath = output_dir / f"ft53_full_benchmark_{output_data['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)

    latest_path = output_dir / "ft53_full_benchmark_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Gerar relatorio markdown
    report = generate_markdown_report(
        [phase1_results, phase2_results, phase3_results],
        [time1, time2, time3]
    )

    report_path = Path("misc/ft53_benchmark_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Relatorio gerado: {report_path}")

    # Imprimir resumo final
    print("\n" + "=" * 80)
    print("RESUMO FINAL")
    print("=" * 80)

    best = sorted(phase3_results, key=lambda x: x['best'])[0]
    gap = ((best['best'] - OPTIMAL_VALUE) / OPTIMAL_VALUE * 100)

    print(f"\nMelhor resultado encontrado:")
    print(f"  Configuracao: {best['name']}")
    print(f"  Best: {best['best']:.0f}")
    print(f"  Average: {best['average']:.1f}")
    print(f"  Gap: {gap:.1f}%")

    print(f"\nComparacao com artigo (CX2):")
    print(f"  Artigo Best: {PAPER_RESULTS['CX2']['best']}")
    print(f"  Nossa Best: {best['best']:.0f}")
    improvement = ((PAPER_RESULTS['CX2']['best'] - best['best']) / PAPER_RESULTS['CX2']['best'] * 100)
    print(f"  Melhoria: {improvement:.1f}%")

    print(f"\nTempo total de execucao: {execution_time_formatted}")
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETO FINALIZADO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
