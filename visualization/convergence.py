"""
Visualizacao de convergencia para experimentos com AG.

Cria graficos mostrando a melhoria do fitness ao longo das geracoes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.fitness import FitnessCalculator
from tsplib.parser import TSPLibParser


class ConvergencePlotter:
    """Cria graficos de convergencia para execucoes do AG."""

    def __init__(self, output_dir: str = "results/plots"):
        """
        Inicializa o plotador.

        Args:
            output_dir: Diretorio para salvar os graficos.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_single_run(self, result, instance_name: str, operator: str,
                        save: bool = True):
        """
        Plota a curva de convergencia para uma unica execucao do AG.

        Args:
            result: Objeto GAResult com o historico de fitness.
            instance_name: Nome da instancia.
            operator: Nome do operador de cruzamento.
            save: Se deve salvar o grafico.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        generations = range(len(result.fitness_history))

        ax.plot(generations, result.fitness_history, 'b-', label='Melhor', linewidth=2)
        ax.plot(generations, result.avg_fitness_history, 'g--', label='Media', alpha=0.7)
        ax.plot(generations, result.worst_fitness_history, 'r:', label='Pior', alpha=0.5)

        ax.set_xlabel('Geracao')
        ax.set_ylabel('Fitness (Distancia)')
        ax.set_title(f'{instance_name} - Convergencia {operator}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{instance_name}_{operator}_convergence.png'
            plt.savefig(filepath, dpi=150)
            print(f"Salvo: {filepath}")

        plt.close()

    def plot_operator_comparison_convergence(
        self,
        results: Dict[str, 'GAResult'],
        instance_name: str,
        save: bool = True
    ):
        """
        Plota curvas de convergencia comparando multiplos operadores.

        Args:
            results: Dicionario mapeando nome do operador para GAResult.
            instance_name: Nome da instancia.
            save: Se deve salvar o grafico.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = {'PMX': 'blue', 'OX': 'green', 'CX2': 'red'}

        for operator, result in results.items():
            generations = range(len(result.fitness_history))
            color = colors.get(operator, 'black')

            ax.plot(generations, result.fitness_history, '-',
                    color=color, label=f'{operator} Melhor', linewidth=2)
            ax.plot(generations, result.avg_fitness_history, '--',
                    color=color, label=f'{operator} Media', alpha=0.5)

        ax.set_xlabel('Geracao')
        ax.set_ylabel('Fitness (Distancia)')
        ax.set_title(f'{instance_name} - Comparacao de Convergencia')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{instance_name}_convergence_comparison.png'
            plt.savefig(filepath, dpi=150)
            print(f"Salvo: {filepath}")

        plt.close()

    def run_and_plot_convergence(
        self,
        instance_file: str,
        operators: List[str] = None,
        save: bool = True
    ):
        """
        Executa o AG e plota a convergencia para uma instancia.

        Args:
            instance_file: Caminho para o arquivo de instancia TSPLIB.
            operators: Lista de operadores para comparar.
            save: Se deve salvar os graficos.
        """
        operators = operators or ['PMX', 'OX', 'CX2']

        parser = TSPLibParser()
        instance = parser.parse(instance_file)
        distance_matrix = instance.get_distance_matrix()
        fitness_calc = FitnessCalculator(distance_matrix)

        results = {}

        print(f"\nExecutando analise de convergencia para {instance.name}...")

        for operator in operators:
            print(f"  Executando {operator}...")

            config = GAConfig(
                population_size=150 if instance.dimension < 100 else 200,
                max_generations=500 if instance.dimension < 100 else 1000,
                crossover_probability=0.80,
                mutation_probability=0.10,
                crossover_type=operator,
                random_seed=42  # Semente fixa para reprodutibilidade
            )

            ga = GeneticAlgorithm(fitness_calc, config)
            result = ga.run(verbose=False)
            results[operator] = result

            # Plotar convergencia individual
            self.plot_single_run(result, instance.name, operator, save)

        # Plotar comparacao
        self.plot_operator_comparison_convergence(results, instance.name, save)

        return results


def main():
    """Gera graficos de convergencia para as instancias disponiveis."""
    plotter = ConvergencePlotter()

    instances_dir = Path("tsplib/instances")
    instance_files = list(instances_dir.glob("*.tsp"))

    if not instance_files:
        print("Nenhum arquivo de instancia encontrado")
        return

    for instance_file in instance_files[:2]:  # Limitado para demonstracao
        try:
            plotter.run_and_plot_convergence(str(instance_file))
        except Exception as e:
            print(f"Erro ao processar {instance_file}: {e}")


if __name__ == "__main__":
    main()
