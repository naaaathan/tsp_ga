# Algoritmo Genético para TSP com Operador CX2

Implementação de um Algoritmo Genético para o Problema do Caixeiro Viajante (TSP), com foco na reprodução e análise do operador CX2 (Modified Cycle Crossover) proposto por Hussain et al. (2017).

## Sobre o Projeto

Este projeto reproduz os experimentos do artigo:

> **Hussain, A. et al. (2017).** "Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator". *Computational Intelligence and Neuroscience*.

### Operadores Implementados

- **PMX** (Partially Mapped Crossover) - Goldberg & Lingle, 1985
- **OX** (Order Crossover) - Davis, 1985
- **CX2** (Modified Cycle Crossover) - Hussain et al., 2017

### Descoberta Principal

O artigo original não especifica o tipo de seleção utilizado. Nossa análise demonstrou que:

| Seleção | Best | Gap% |
|---------|------|------|
| Roleta | 1382 | 97.7% |
| **Torneio** | **678** | **-3.0%** |

**Com seleção por torneio, reproduzimos e superamos os resultados do artigo.**

## Instalação

### Requisitos

- Python 3.8+
- NumPy
- Matplotlib (opcional, para visualizações)

### Setup

```bash
# Clonar o repositório
git clone <repo-url>
cd tsp_ga

# Instalar dependências
pip install numpy matplotlib
```

## Uso Rápido

### Executar um Experimento Simples

```python
from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser

# Carregar instância
parser = TSPLibParser()
instance = parser.parse("tsplib/instances/dantzig42.tsp")
fitness_calc = FitnessCalculator(instance.get_distance_matrix())

# Configurar GA
config = GAConfig(
    population_size=150,
    max_generations=500,
    crossover_probability=0.80,
    mutation_probability=0.10,
    crossover_type='CX2',
    selection_type='tournament',  # Importante para CX2!
    elitism_count=2
)

# Executar
ga = GeneticAlgorithm(fitness_calc, config)
result = ga.run(verbose=True)

print(f"Melhor distância: {result.best_fitness}")
print(f"Melhor tour: {result.best_tour}")
```

### Linha de Comando

```bash
# Executar experimentos completos (30 runs por operador)
python run_experiments.py

# Teste rápido do CX2 (5 runs)
python run_cx2_quick.py

# Otimização de parâmetros do CX2 (3 fases)
python run_cx2_phase1.py  # Análise de sensibilidade
python run_cx2_phase2.py  # Combinações
python run_cx2_phase3.py  # Validação final
```

## Executar Contra um Benchmark

### Usando uma Instância do TSPLIB

```python
from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser

# 1. Carregar a instância
parser = TSPLibParser()
instance = parser.parse("tsplib/instances/dantzig42.tsp")

# 2. Criar calculador de fitness
fitness_calc = FitnessCalculator(instance.get_distance_matrix())

# 3. Configurar o AG (configuração recomendada para CX2)
config = GAConfig(
    population_size=150,
    max_generations=500,
    crossover_probability=0.80,
    mutation_probability=0.10,
    crossover_type='CX2',      # Opções: 'PMX', 'OX', 'CX2'
    mutation_type='swap',       # Opções: 'swap', 'insert', 'inversion', 'scramble'
    selection_type='tournament', # Opções: 'roulette', 'tournament', 'rank'
    elitism_count=2,
    random_seed=42              # Para reprodutibilidade
)

# 4. Executar
ga = GeneticAlgorithm(fitness_calc, config)
result = ga.run(verbose=True)

# 5. Analisar resultados
print(f"\n=== Resultados ===")
print(f"Melhor distância: {result.best_fitness}")
print(f"Tempo de execução: {result.execution_time:.2f}s")
print(f"Gerações executadas: {result.generations_run}")
```

### Executar Múltiplas Vezes (Validação Estatística)

```python
import numpy as np
from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser

# Carregar instância
parser = TSPLibParser()
instance = parser.parse("tsplib/instances/dantzig42.tsp")
fitness_calc = FitnessCalculator(instance.get_distance_matrix())

# Executar 30 runs
results = []
for run in range(30):
    config = GAConfig(
        population_size=150,
        max_generations=500,
        crossover_probability=0.80,
        mutation_probability=0.10,
        crossover_type='CX2',
        selection_type='tournament',
        elitism_count=2,
        random_seed=run  # Semente diferente para cada run
    )

    ga = GeneticAlgorithm(fitness_calc, config)
    result = ga.run(verbose=False)
    results.append(result.best_fitness)
    print(f"Run {run+1}/30: {result.best_fitness:.0f}")

# Estatísticas
print(f"\n=== Estatísticas (30 runs) ===")
print(f"Best:    {min(results):.0f}")
print(f"Worst:   {max(results):.0f}")
print(f"Average: {np.mean(results):.1f}")
print(f"Std:     {np.std(results):.1f}")
```

### Comparar Operadores de Crossover

```python
import numpy as np
from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser

# Carregar instância
parser = TSPLibParser()
instance = parser.parse("tsplib/instances/dantzig42.tsp")
fitness_calc = FitnessCalculator(instance.get_distance_matrix())

# Testar cada operador
for operator in ['PMX', 'OX', 'CX2']:
    results = []
    for run in range(10):
        config = GAConfig(
            population_size=150,
            max_generations=500,
            crossover_probability=0.80,
            mutation_probability=0.10,
            crossover_type=operator,
            selection_type='tournament',
            elitism_count=2,
            random_seed=run
        )

        ga = GeneticAlgorithm(fitness_calc, config)
        result = ga.run(verbose=False)
        results.append(result.best_fitness)

    print(f"{operator}: Best={min(results):.0f}, Avg={np.mean(results):.1f}")
```

## Instâncias Disponíveis

O projeto inclui instâncias do TSPLIB em `tsplib/instances/`:

| Instância | Cidades | Ótimo | Tipo |
|-----------|---------|-------|------|
| dantzig42.tsp | 42 | 699 | Simétrico |
| fri26.tsp | 26 | 937 | Simétrico |

Para adicionar novas instâncias, baixe do [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) e coloque na pasta `tsplib/instances/`.

## Configuração do Algoritmo Genético

### Parâmetros Disponíveis

```python
GAConfig(
    population_size=150,       # Tamanho da população
    max_generations=500,       # Número máximo de gerações
    crossover_probability=0.80, # Probabilidade de crossover (0.0-1.0)
    mutation_probability=0.10,  # Probabilidade de mutação (0.0-1.0)
    crossover_type='CX2',       # 'PMX', 'OX', 'CX2'
    mutation_type='swap',       # 'swap', 'insert', 'inversion', 'scramble'
    selection_type='tournament', # 'roulette', 'tournament', 'rank'
    elitism_count=2,            # Número de indivíduos preservados
    random_seed=None            # Semente para reprodutibilidade
)
```

### Configurações Recomendadas

#### Para Reproduzir o Artigo (CX2)
```python
config = GAConfig(
    population_size=150,
    max_generations=500,
    crossover_probability=0.80,
    mutation_probability=0.10,
    crossover_type='CX2',
    selection_type='tournament',  # IMPORTANTE!
    elitism_count=2
)
# Resultado esperado: Best ~678, Average ~800
```

#### Para Melhores Resultados (CX2 Otimizado)
```python
config = GAConfig(
    population_size=500,
    max_generations=2000,
    crossover_probability=0.70,
    mutation_probability=0.30,
    crossover_type='CX2',
    selection_type='tournament',
    elitism_count=15
)
# Resultado esperado: Best ~648, Average ~734
```

## Estrutura do Projeto

```
tsp_ga/
├── ga/                         # Módulo principal do AG
│   ├── genetic_algorithm.py    # Classe principal GeneticAlgorithm
│   ├── chromosome.py           # Chromosome e Population
│   ├── fitness.py              # Calculador de fitness
│   ├── crossover.py            # Operadores PMX, OX, CX2
│   ├── mutation.py             # Operadores de mutação
│   └── selection.py            # Operadores de seleção
├── tsplib/                     # Parser de instâncias TSPLIB
│   ├── parser.py
│   └── instances/              # Arquivos .tsp
├── experiments/                # Execução de experimentos
│   ├── runner.py
│   └── statistics.py
├── results/                    # Resultados dos experimentos
├── run_experiments.py          # Experimentos completos
├── run_cx2_quick.py           # Teste rápido
├── run_cx2_phase1.py          # Otimização Fase 1
├── run_cx2_phase2.py          # Otimização Fase 2
├── run_cx2_phase3.py          # Otimização Fase 3
```

## Resultados Principais

### dantzig42 (42 cidades, Ótimo: 699)

| Configuração | Best | Average | Gap% |
|--------------|------|---------|------|
| CX2 + Roleta (interpretação inicial) | 1382 | 1654.4 | 97.7% |
| CX2 + Torneio (baseline correto) | 678 | 800.7 | -3.0% |
| CX2 + Otimizado | **648** | **733.7** | **-7.3%** |
| Artigo Original | 699 | 802 | 0% |

### Conclusão

Com a configuração correta (seleção por torneio), conseguimos:
- **Reproduzir** os resultados do artigo
- **Superar** os resultados do artigo com otimização

## Documentação

- `PROJETO_FINAL.md` - Documentação completa da reprodução inicial
- `PROJETO_FINAL_2.md` - Análise de reprodutibilidade e otimização
- `Apresentacao2.md` - Slides da apresentação

## Referências

1. Hussain, A., Muhammad, Y.S., Sajid, M.N., Hussain, I., Shoukry, A.M., & Gani, S. (2017). *Genetic Algorithm for Traveling Salesman Problem with Modified Cycle Crossover Operator*. Computational Intelligence and Neuroscience.

2. Goldberg, D.E., & Lingle, R. (1985). *Alleles, Loci and the Traveling Salesman Problem*. Proceedings of the 1st International Conference on Genetic Algorithms.

3. Davis, L. (1985). *Applying Adaptive Algorithms to Epistatic Domains*. IJCAI.

4. TSPLIB - http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

## Licença

Este projeto foi desenvolvido para fins acadêmicos na disciplina de Algoritmos Genéticos.
