"""
Testes de validacao para a implementacao do AG.

Os testes incluem:
1. Exemplo de 7 cidades do artigo (Tabela 2, otimo = 159)
2. Corretude dos operadores de cruzamento
3. Verificacoes de validade dos cromossomos
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ga.chromosome import Chromosome, Population
from ga.fitness import FitnessCalculator
from ga.crossover import PMX, OX, CX2, CrossoverFactory
from ga.mutation import Mutation
from ga.selection import Selection
from ga.genetic_algorithm import GeneticAlgorithm, GAConfig


def test_chromosome_validity():
    """Testar se os cromossomos sao permutacoes validas."""
    print("=" * 60)
    print("TESTE: Validade do Cromossomo")
    print("=" * 60)

    # Criar cromossomo aleatorio
    c = Chromosome(n_cities=10)
    assert c.is_valid(), "Cromossomo aleatorio deve ser valido"
    assert len(c) == 10, "Comprimento do cromossomo deve corresponder a n_cities"
    assert set(c.genes) == set(range(10)), "Deve conter todas as cidades de 0 a 9"

    # Criar a partir de genes
    c2 = Chromosome(genes=[0, 2, 1, 3, 4])
    assert c2.is_valid(), "Cromossomo manual deve ser valido"

    # Testar copia
    c3 = c2.copy()
    assert np.array_equal(c2.genes, c3.genes), "Copia deve ter os mesmos genes"

    print("[PASSOU] Todos os testes de cromossomo passaram")
    return True


def test_crossover_validity():
    """Testar se os operadores de cruzamento produzem descendentes validos."""
    print("\n" + "=" * 60)
    print("TESTE: Validade dos Operadores de Cruzamento")
    print("=" * 60)

    n = 8
    p1 = Chromosome(genes=np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    p2 = Chromosome(genes=np.array([7, 6, 5, 4, 3, 2, 1, 0]))

    for name, operator in [('PMX', PMX.crossover),
                           ('OX', OX.crossover),
                           ('CX2', CX2.crossover_simple)]:
        print(f"\nTestando {name}...")

        # Executar multiplas vezes para testar diferentes pontos de corte aleatorios
        for _ in range(10):
            o1, o2 = operator(p1, p2)

            assert o1.is_valid(), f"Descendente 1 do {name} deve ser valido"
            assert o2.is_valid(), f"Descendente 2 do {name} deve ser valido"
            assert len(o1) == n, f"Descendente 1 do {name} deve ter {n} cidades"
            assert len(o2) == n, f"Descendente 2 do {name} deve ter {n} cidades"

        print(f"  [PASSOU] {name} produz descendentes validos")

    print("\n[PASSOU] Todos os testes de cruzamento passaram")
    return True


def test_cx2_paper_example():
    """Testar CX2 com o exemplo do artigo (Caso 1)."""
    print("\n" + "=" * 60)
    print("TESTE: Exemplo do Artigo CX2 (Caso 1)")
    print("=" * 60)

    # Exemplo do artigo (indexado a partir de 1 no artigo, usamos indexacao a partir de 0)
    # P1 = (3 4 8 2 7 1 6 5) -> (2 3 7 1 6 0 5 4) em indexacao base 0
    # P2 = (4 2 5 1 6 8 3 7) -> (3 1 4 0 5 7 2 6) em indexacao base 0

    # Usando valores originais indexados a partir de 1 como base 0 (apenas subtrair 1)
    p1 = Chromosome(genes=np.array([2, 3, 7, 1, 6, 0, 5, 4]))  # 3,4,8,2,7,1,6,5 - 1
    p2 = Chromosome(genes=np.array([3, 1, 4, 0, 5, 7, 2, 6]))  # 4,2,5,1,6,8,3,7 - 1

    print(f"P1: {p1.genes + 1}")  # Imprimir com indexacao base 1 para comparacao
    print(f"P2: {p2.genes + 1}")

    o1, o2 = CX2.crossover_simple(p1, p2)

    print(f"O1: {o1.genes + 1}")
    print(f"O2: {o2.genes + 1}")

    # Verificar validade
    assert o1.is_valid(), "O1 deve ser valido"
    assert o2.is_valid(), "O2 deve ser valido"

    print("\n[PASSOU] Teste do exemplo do artigo CX2 passou (produz descendentes validos)")
    return True


def test_paper_7city_instance():
    """
    Testar com a instancia de 7 cidades da Tabela 2 do artigo.
    Caminho otimo: 6->1->5->3->4->2->7 com distancia 159

    Matriz de distancias do artigo (Tabela 2):
    Cidade  1    2    3    4    5    6    7
    1       0   34   36   37   31   33   35
    2      34    0   29   23   22   25   24
    3      36   29    0   17   12   18   17
    4      37   23   17    0   32   30   29
    5      31   22   12   17    0   26   24
    6      33   25   18   30   26    0   19
    7      35   24   17   29   24   19    0
    """
    print("\n" + "=" * 60)
    print("TESTE: Instancia de 7 Cidades do Artigo (Tabela 2)")
    print("=" * 60)

    # Matriz de distancias (indexacao base 0, entao cidade 1 = indice 0)
    distance_matrix = np.array([
        [0,  34, 36, 37, 31, 33, 35],  # Cidade 1
        [34,  0, 29, 23, 22, 25, 24],  # Cidade 2
        [36, 29,  0, 17, 12, 18, 17],  # Cidade 3
        [37, 23, 17,  0, 32, 30, 29],  # Cidade 4
        [31, 22, 12, 17,  0, 26, 24],  # Cidade 5
        [33, 25, 18, 30, 26,  0, 19],  # Cidade 6
        [35, 24, 17, 29, 24, 19,  0],  # Cidade 7
    ], dtype=np.float64)

    # Rota otima: 6->1->5->3->4->2->7 (indexacao base 1)
    # Em indexacao base 0: 5->0->4->2->3->1->6
    optimal_tour = [5, 0, 4, 2, 3, 1, 6]
    optimal_distance = 159

    # Verificar distancia otima
    fitness_calc = FitnessCalculator(distance_matrix)
    calculated_optimal = fitness_calc.get_tour_distance(optimal_tour)
    print(f"Rota otima (base 0): {optimal_tour}")
    print(f"Rota otima (base 1): {[x+1 for x in optimal_tour]}")
    print(f"Distancia otima esperada: {optimal_distance}")
    print(f"Distancia otima calculada: {calculated_optimal}")

    assert abs(calculated_optimal - optimal_distance) < 1e-6, \
        f"Discrepancia na distancia otima: {calculated_optimal} vs {optimal_distance}"

    # Executar AG com os parametros do artigo
    print("\nExecutando AG com parametros do artigo...")
    config = GAConfig(
        population_size=30,
        max_generations=10,
        crossover_probability=0.8,
        mutation_probability=0.1,
        crossover_type='CX2',
        random_seed=42
    )

    # Executar multiplos experimentos
    results_by_operator = {}
    for operator in ['PMX', 'OX', 'CX2']:
        config.crossover_type = operator
        best_results = []

        for run in range(30):
            config.random_seed = run
            ga = GeneticAlgorithm(fitness_calc, config)
            result = ga.run(verbose=False)
            best_results.append(result.best_fitness)

        results_by_operator[operator] = {
            'best': min(best_results),
            'worst': max(best_results),
            'average': np.mean(best_results),
            'optimal_count': sum(1 for x in best_results if x == optimal_distance)
        }

    print("\nResultados (30 execucoes cada):")
    print("-" * 50)
    print(f"{'Operador':<10} {'Melhor':<10} {'Pior':<10} {'Media':<10} {'Otimo':<10}")
    print("-" * 50)
    for op, res in results_by_operator.items():
        print(f"{op:<10} {res['best']:<10.1f} {res['worst']:<10.1f} "
              f"{res['average']:<10.1f} {res['optimal_count']}/30")

    print("\nResultados do artigo (Tabela 3):")
    print("-" * 50)
    print("PMX: Melhor=159, Pior=165, Media=159.7, Otimo=17/30")
    print("OX:  Melhor=159, Pior=163, Media=160.3, Otimo=14/30")
    print("CX2: Melhor=159, Pior=162, Media=159.2, Otimo=24/30")

    print("\n[PASSOU] Teste da instancia de 7 cidades concluido")
    return True


def test_mutation():
    """Testar operadores de mutacao."""
    print("\n" + "=" * 60)
    print("TESTE: Operadores de Mutacao")
    print("=" * 60)

    c = Chromosome(genes=np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    original = c.genes.copy()

    # Testar mutacao por troca (com probabilidade 1.0 para garantir que aconteca)
    c_swap = c.copy()
    Mutation.swap(c_swap, probability=1.0)
    assert c_swap.is_valid(), "Mutacao por troca deve produzir cromossomo valido"
    print(f"Troca: {original} -> {c_swap.genes}")

    # Testar mutacao por inversao
    c_inv = c.copy()
    Mutation.inversion(c_inv, probability=1.0)
    assert c_inv.is_valid(), "Mutacao por inversao deve produzir cromossomo valido"
    print(f"Inversao: {original} -> {c_inv.genes}")

    print("\n[PASSOU] Todos os testes de mutacao passaram")
    return True


def test_selection():
    """Testar operadores de selecao."""
    print("\n" + "=" * 60)
    print("TESTE: Operadores de Selecao")
    print("=" * 60)

    # Criar uma populacao com valores de aptidao conhecidos
    pop = Population(10, 5)
    for i, c in enumerate(pop.chromosomes):
        c.fitness = (i + 1) * 100  # 100, 200, 300, ..., 1000

    # Testar roleta (melhor aptidao = menor valor = maior probabilidade de selecao)
    selected = Selection.roulette_wheel(pop, 5)
    assert len(selected) == 5, "Deve selecionar 5 pais"
    print(f"Valores de aptidao selecionados por roleta: {[c.fitness for c in selected]}")

    # Testar torneio
    selected_tour = Selection.tournament(pop, 5, tournament_size=3)
    assert len(selected_tour) == 5, "Deve selecionar 5 pais"
    print(f"Valores de aptidao selecionados por torneio: {[c.fitness for c in selected_tour]}")

    # Testar elitismo
    elite = Selection.elitism(pop, 2)
    assert len(elite) == 2, "Deve selecionar 2 elites"
    assert elite[0].fitness == 100, "O melhor deve ter aptidao 100"
    assert elite[1].fitness == 200, "O segundo melhor deve ter aptidao 200"
    print(f"Valores de aptidao dos elites: {[c.fitness for c in elite]}")

    print("\n[PASSOU] Todos os testes de selecao passaram")
    return True


def run_all_tests():
    """Executar todos os testes de validacao."""
    print("\n" + "=" * 60)
    print("EXECUTANDO TODOS OS TESTES DE VALIDACAO")
    print("=" * 60)

    tests = [
        ("Validade do Cromossomo", test_chromosome_validity),
        ("Validade do Cruzamento", test_crossover_validity),
        ("Exemplo do Artigo CX2", test_cx2_paper_example),
        ("Operadores de Mutacao", test_mutation),
        ("Operadores de Selecao", test_selection),
        ("Instancia de 7 Cidades", test_paper_7city_instance),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n[FALHOU] {name} FALHOU: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESUMO: {passed} passaram, {failed} falharam")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
