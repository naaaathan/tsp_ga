"""
Otimizacao CX2 - Fase 3: Validacao e Documentacao

Este script:
1. Executa 30 rodadas independentes nas melhores configuracoes da Fase 2
2. Calcula estatisticas completas
3. Gera PROJETO_FINAL_2.md com os resultados da otimizacao

Tempo de execucao esperado: ~20-30 minutos
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


def run_experiment(fitness_calc, config, num_runs=30):
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
        'median': float(np.median(all_values)),
        'q1': float(np.percentile(all_values, 25)),
        'q3': float(np.percentile(all_values, 75)),
        'all_values': [float(v) for v in all_values],
        'best_tour': [int(c) for c in best_tour] if best_tour else None,
        'config': config,
        'num_runs': num_runs
    }


def load_phase2_results():
    """Carrega os resultados da Fase 2, se disponiveis."""
    phase2_path = Path("results/cx2_phase2_latest.json")
    if phase2_path.exists():
        with open(phase2_path, 'r') as f:
            return json.load(f)
    return None


def generate_projeto_final_2(results, optimal, paper_best, paper_avg, execution_time):
    """Gera PROJETO_FINAL_2.md com os resultados da otimizacao."""

    best_result = results[0]  # Ja ordenado pelo melhor

    content = f"""# Projeto Final 2: Otimizacao do Operador CX2

## Resumo Executivo

Este documento apresenta os resultados da otimizacao de parametros do operador CX2 (Modified Cycle Crossover) para o Problema do Caixeiro Viajante (TSP).

**Objetivo:** Melhorar o desempenho do CX2 atraves de ajuste de parametros.

**Resultado Principal:**
- **Baseline CX2 (parametros do artigo):** Best = 1382, Gap = 97.7%
- **CX2 Otimizado:** Best = {best_result['best']:.0f}, Gap = {((best_result['best'] - optimal) / optimal * 100):.1f}%
- **Artigo original:** Best = {paper_best}, Gap = 0%

---

## 1. Metodologia

### 1.1 Abordagem em Fases

A otimizacao foi realizada em tres fases:

1. **Fase 1 - Analise de Sensibilidade:** Teste individual de cada parametro
2. **Fase 2 - Combinacoes:** Teste de combinacoes dos melhores parametros
3. **Fase 3 - Validacao:** Execucao de 30 runs nas melhores configuracoes

### 1.2 Instancia de Teste

- **Instancia:** dantzig42 (42 cidades)
- **Valor Otimo:** {optimal}
- **Resultado do Artigo:** Best = {paper_best}, Average = {paper_avg}

---

## 2. Resultados da Validacao (30 runs)

### 2.1 Top 5 Configuracoes

| Rank | Configuracao | Best | Average | Worst | Std | Gap% |
|------|--------------|------|---------|-------|-----|------|
"""

    for i, r in enumerate(results[:5]):
        gap = ((r['best'] - optimal) / optimal * 100)
        content += f"| {i+1} | {r['name']} | {r['best']:.0f} | {r['average']:.1f} | {r['worst']:.0f} | {r['std']:.1f} | {gap:.1f}% |\n"

    content += f"""
### 2.2 Melhor Configuracao

**Nome:** {best_result['name']}

**Parametros:**
```
population_size: {best_result['config']['population_size']}
max_generations: {best_result['config']['max_generations']}
crossover_probability: {best_result['config']['crossover_probability']}
mutation_probability: {best_result['config']['mutation_probability']}
selection_type: {best_result['config']['selection_type']}
elitism_count: {best_result['config']['elitism_count']}
crossover_type: CX2
mutation_type: swap
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
| Gap% | {((best_result['best'] - optimal) / optimal * 100):.1f}% |

---

## 3. Comparacao: Baseline vs Otimizado vs Artigo

### 3.1 Tabela Comparativa

| Metrica | Baseline (Roleta) | CX2 Otimizado | Artigo |
|---------|-------------------|---------------|--------|
| Best | 1382 | {best_result['best']:.0f} | {paper_best} |
| Average | 1654.4 | {best_result['average']:.1f} | {paper_avg} |
| Gap% | 97.7% | {((best_result['best'] - optimal) / optimal * 100):.1f}% | 0% |

### 3.2 Melhoria Obtida

"""

    baseline_best = 1382
    improvement = ((baseline_best - best_result['best']) / baseline_best * 100)

    if best_result['best'] <= paper_best:
        content += f"""**SUCESSO:** A configuracao otimizada atingiu ou superou o resultado do artigo!

- Melhoria sobre baseline: **{improvement:.1f}%**
- Resultado: Best = {best_result['best']:.0f} vs Artigo = {paper_best}
"""
    else:
        content += f"""**Melhoria significativa sobre baseline:**

- Melhoria: **{improvement:.1f}%** (de 1382 para {best_result['best']:.0f})
- Ainda acima do artigo em: {best_result['best'] - paper_best:.0f} unidades
"""

    content += f"""
---

## 4. Analise dos Parametros Chave

### 4.1 Fator de Maior Impacto

O parametro com maior impacto no desempenho do CX2 foi o **tipo de selecao**.

| Tipo de Selecao | Best Tipico |
|-----------------|-------------|
| Roleta (baseline) | ~1382 |
| Torneio | ~678-750 |
| Ranking | ~1310-1400 |

**Conclusao:** A selecao por torneio fornece pressao seletiva mais adequada para o CX2.

### 4.2 Outros Fatores Importantes

1. **Elitismo:** Valores entre 10-20 melhoram a convergencia
2. **Geracoes:** Mais geracoes (1000+) permitem melhor exploracao
3. **Mutacao:** Taxa entre 0.10-0.15 e adequada

---

## 5. Distribuicao dos Resultados

### 5.1 Valores das 30 Execucoes (Melhor Config)

```
{best_result['all_values']}
```

### 5.2 Histograma (representacao textual)

"""

    # Criar histograma simples
    values = best_result['all_values']
    min_val = min(values)
    max_val = max(values)
    bins = 5
    bin_size = (max_val - min_val) / bins if max_val > min_val else 1

    for i in range(bins):
        bin_start = min_val + i * bin_size
        bin_end = min_val + (i + 1) * bin_size
        count = sum(1 for v in values if bin_start <= v < bin_end or (i == bins-1 and v == max_val))
        bar = '#' * count
        content += f"[{bin_start:6.0f}-{bin_end:6.0f}] {bar} ({count})\n"

    content += f"""
---

## 6. Conclusoes

### 6.1 Objetivos Alcancados

"""

    if best_result['best'] < 1382:
        content += "- [X] Melhorou sobre baseline (1382)\n"
    else:
        content += "- [ ] Melhorou sobre baseline (1382)\n"

    if best_result['best'] < 1000:
        content += "- [X] Atingiu Best < 1000 (criterio minimo)\n"
    else:
        content += "- [ ] Atingiu Best < 1000 (criterio minimo)\n"

    if best_result['best'] < 800:
        content += "- [X] Atingiu Best < 800 (bom resultado)\n"
    else:
        content += "- [ ] Atingiu Best < 800 (bom resultado)\n"

    if best_result['best'] <= paper_best:
        content += "- [X] Igualou ou superou artigo (699)\n"
    else:
        content += "- [ ] Igualou ou superou artigo (699)\n"

    content += f"""
### 6.2 Principais Descobertas

1. **Selecao por torneio** e crucial para o bom desempenho do CX2
2. O artigo provavelmente utilizou torneio ou configuracao similar nao documentada
3. Com ajuste de parametros, o CX2 pode atingir resultados competitivos

### 6.3 Licoes Aprendidas

- Parametros padrao nem sempre sao ideais para todos os operadores
- A combinacao crossover+selecao e mais importante que parametros individuais
- Experimentacao sistematica e essencial para otimizacao

---

## 7. Informacoes de Execucao

- **Tempo total de execucao:** {execution_time}
- **Data:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Numero de runs por config:** 30

---

## Anexo: Todos os Resultados das 30 Runs

"""

    for r in results:
        gap = ((r['best'] - optimal) / optimal * 100)
        content += f"""
### {r['name']}

- Best: {r['best']:.0f} | Avg: {r['average']:.1f} | Worst: {r['worst']:.0f} | Gap: {gap:.1f}%
- Config: pop={r['config']['population_size']}, gen={r['config']['max_generations']}, sel={r['config']['selection_type']}, elite={r['config']['elitism_count']}
"""

    content += """
---

*Documento gerado automaticamente pelo script run_cx2_phase3.py*
*Este documento complementa PROJETO_FINAL.md com os resultados da otimizacao*
"""

    return content


def main():
    total_start_time = time.time()

    print("=" * 80)
    print("OTIMIZACAO CX2 - FASE 3: VALIDACAO (30 rodadas)")
    print("=" * 80)
    print("\nEsta fase valida as melhores configuracoes com 30 rodadas independentes.")
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
    num_runs = 30

    print(f"Instancia: dantzig42 ({instance.dimension} cidades)")
    print(f"Otimo: {optimal}")
    print(f"CX2 do artigo: Best={paper_best}, Avg={paper_avg}")
    print(f"Rodadas por configuracao: {num_runs}")

    # Carregar resultados da Fase 2 ou usar valores padrao
    phase2 = load_phase2_results()

    if phase2 and 'top5_for_phase3' in phase2:
        print("\n[INFO] Carregando as 5 melhores configuracoes da Fase 2...")
        configs_to_test = [
            (item['name'], item['config']) for item in phase2['top5_for_phase3']
        ]
    else:
        print("\n[AVISO] Resultados da Fase 2 nao encontrados. Usando configuracoes padrao.")
        configs_to_test = [
            ('Tournament (default)', {
                'population_size': 150,
                'max_generations': 500,
                'crossover_probability': 0.80,
                'mutation_probability': 0.10,
                'selection_type': 'tournament',
                'elitism_count': 2,
            }),
            ('Tournament + Elite=15', {
                'population_size': 150,
                'max_generations': 500,
                'crossover_probability': 0.80,
                'mutation_probability': 0.10,
                'selection_type': 'tournament',
                'elitism_count': 15,
            }),
            ('Tournament + Gen=1000', {
                'population_size': 150,
                'max_generations': 1000,
                'crossover_probability': 0.80,
                'mutation_probability': 0.10,
                'selection_type': 'tournament',
                'elitism_count': 2,
            }),
            ('Roulette + Elite=20', {
                'population_size': 150,
                'max_generations': 500,
                'crossover_probability': 0.80,
                'mutation_probability': 0.10,
                'selection_type': 'roulette',
                'elitism_count': 20,
            }),
            ('Aggressive', {
                'population_size': 300,
                'max_generations': 1000,
                'crossover_probability': 0.80,
                'mutation_probability': 0.15,
                'selection_type': 'tournament',
                'elitism_count': 15,
            }),
        ]

    print(f"\nConfiguracoes a serem validadas:")
    for name, _ in configs_to_test:
        print(f"  - {name}")

    print("\n" + "-" * 80)

    # Executar validacao
    results = []

    for i, (name, config) in enumerate(configs_to_test):
        print(f"\n[{i+1}/{len(configs_to_test)}] Validando: {name}")
        print(f"    Executando 30 rodadas independentes...")

        start = time.time()
        result = run_experiment(fitness_calc, config, num_runs)
        result['name'] = name
        elapsed = time.time() - start

        gap = ((result['best'] - optimal) / optimal * 100)
        print(f"    Concluido em {elapsed:.1f}s")
        print(f"    Melhor: {result['best']:.0f} | Media: {result['average']:.1f} | "
              f"Pior: {result['worst']:.0f} | Gap: {gap:.1f}%")

        results.append(result)

    # Ordenar pelo melhor valor
    results_sorted = sorted(results, key=lambda x: x['best'])

    # Imprimir ranking final
    print("\n" + "=" * 80)
    print("RESULTADOS FINAIS DA VALIDACAO (30 rodadas cada)")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Configuracao':<40} {'Melhor':<8} {'Media':<10} {'Pior':<8} {'Gap%':<8}")
    print("-" * 80)

    for i, result in enumerate(results_sorted):
        gap = ((result['best'] - optimal) / optimal * 100)
        marker = " ***" if result['best'] <= paper_best else ""
        print(f"{i+1:<5} {result['name']:<40} {result['best']:<8.0f} "
              f"{result['average']:<10.1f} {result['worst']:<8.0f} {gap:<8.1f}%{marker}")

    # Melhor resultado
    best_result = results_sorted[0]

    print(f"\n{'='*80}")
    print(f"MELHOR CONFIGURACAO VALIDADA: {best_result['name']}")
    print(f"{'='*80}")
    print(f"\n  Melhor:  {best_result['best']:.0f}")
    print(f"  Media:   {best_result['average']:.1f}")
    print(f"  Pior:    {best_result['worst']:.0f}")
    print(f"  Desvio:  {best_result['std']:.1f}")
    print(f"  Mediana: {best_result['median']:.1f}")
    print(f"  Gap:     {((best_result['best'] - optimal) / optimal * 100):.1f}%")
    print(f"\n  Parametros: {best_result['config']}")

    # Calcular tempo de execucao
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    execution_time = f"{hours}h {minutes}m {seconds}s"

    # Salvar resultados
    all_results = {
        'instance': 'dantzig42',
        'optimal': optimal,
        'paper_cx2': {'best': paper_best, 'avg': paper_avg},
        'num_runs': num_runs,
        'validated_configs': results_sorted,
        'best_config': results_sorted[0],
        'execution_time_seconds': total_elapsed,
        'execution_time_formatted': execution_time,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"cx2_phase3_{all_results['timestamp']}.json"
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)

    latest_path = output_dir / "cx2_phase3_latest.json"
    with open(latest_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResultados salvos em: {filepath}")

    # Gerar PROJETO_FINAL_2.md
    print("\n" + "=" * 80)
    print("GERANDO PROJETO_FINAL_2.md")
    print("=" * 80)

    doc_content = generate_projeto_final_2(
        results_sorted, optimal, paper_best, paper_avg, execution_time
    )

    doc_path = Path("PROJETO_FINAL_2.md")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    print(f"\nDocumentacao gerada: {doc_path}")
    print(f"Tempo total de execucao: {execution_time}")

    # Verificacao final de criterios de sucesso
    print("\n" + "=" * 80)
    print("VERIFICACAO FINAL DOS CRITERIOS DE SUCESSO")
    print("=" * 80)

    baseline_best = 1382

    if best_result['best'] < baseline_best:
        improvement = ((baseline_best - best_result['best']) / baseline_best * 100)
        print(f"[OK] Melhorou sobre o baseline (1382): {best_result['best']:.0f} ({improvement:.1f}% melhor)")
    else:
        print(f"[!!] Nao melhorou sobre o baseline (1382)")

    if best_result['best'] < 1000:
        print(f"[OK] Atingiu Best < 1000 (minimo aceitavel)")
    else:
        print(f"[!!] Nao atingiu Best < 1000")

    if best_result['best'] < 800:
        print(f"[OK] Atingiu Best < 800 (bom resultado)")

    if best_result['best'] <= paper_best:
        print(f"[OK] IGUALOU OU SUPEROU O RESULTADO DO ARTIGO ({paper_best})!")
    else:
        print(f"[..] Ainda acima do melhor do artigo ({paper_best}) por {best_result['best'] - paper_best:.0f}")

    print("\n" + "=" * 80)
    print("OTIMIZACAO CONCLUIDA!")
    print("=" * 80)
    print(f"\nDocumentacao: PROJETO_FINAL_2.md")
    print(f"Nota: PROJETO_FINAL.md NAO foi modificado (conforme solicitado)")


if __name__ == "__main__":
    main()
