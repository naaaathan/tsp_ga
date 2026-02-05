"""
Visualizacoes comparativas para experimentos com AG.

Cria graficos de barras comparando operadores e resultados com o artigo.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional


class ComparisonPlotter:
    """Cria graficos comparativos para resultados de experimentos com AG."""

    # Resultados do artigo para comparacao
    PAPER_RESULTS = {
        'gr21': {
            'optimal': 2707,
            'PMX': {'best': 2962, 'worst': 3322, 'average': 3127},
            'OX': {'best': 3005, 'worst': 3693, 'average': 3208},
            'CX2': {'best': 2995, 'worst': 3576, 'average': 3145},
        },
        'fri26': {
            'optimal': 937,
            'PMX': {'best': 1056, 'worst': 1294, 'average': 1133},
            'OX': {'best': 1051, 'worst': 1323, 'average': 1158},
            'CX2': {'best': 1099, 'worst': 1278, 'average': 1128},
        },
        'dantzig42': {
            'optimal': 699,
            'PMX': {'best': 1298, 'worst': 1606, 'average': 1425},
            'OX': {'best': 1222, 'worst': 1562, 'average': 1301},
            'CX2': {'best': 699, 'worst': 920, 'average': 802},
        },
        'ftv33': {
            'optimal': 1286,
            'PMX': {'best': 1708, 'worst': 2399, 'average': 2012},
            'OX': {'best': 1804, 'worst': 2366, 'average': 2098},
            'CX2': {'best': 1811, 'worst': 2322, 'average': 2083},
        },
        'ftv38': {
            'optimal': 1530,
            'PMX': {'best': 2345, 'worst': 2726, 'average': 2578},
            'OX': {'best': 2371, 'worst': 2913, 'average': 2617},
            'CX2': {'best': 2252, 'worst': 2718, 'average': 2560},
        },
        'ft53': {
            'optimal': 6905,
            'PMX': {'best': 13445, 'worst': 16947, 'average': 14949},
            'OX': {'best': 13826, 'worst': 16279, 'average': 14724},
            'CX2': {'best': 10987, 'worst': 13055, 'average': 12243},
        },
    }

    def __init__(self, results_file: str = None, output_dir: str = "results/plots"):
        """
        Inicializa o plotador.

        Args:
            results_file: Caminho para o arquivo JSON de resultados.
            output_dir: Diretorio para salvar os graficos.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.our_results = {}
        if results_file and Path(results_file).exists():
            with open(results_file) as f:
                self.our_results = json.load(f)

    def plot_operator_comparison(self, instance_name: str, save: bool = True):
        """
        Cria grafico de barras comparando tres operadores para uma instancia.

        Args:
            instance_name: Nome da instancia.
            save: Se deve salvar o grafico.
        """
        if instance_name not in self.our_results:
            print(f"Sem resultados para {instance_name}")
            return

        data = self.our_results[instance_name]
        operators = ['PMX', 'OX', 'CX2']

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(3)  # Melhor, Pior, Media
        width = 0.25

        for i, op in enumerate(operators):
            if op in data['results']:
                r = data['results'][op]
                values = [r['best'], r['worst'], r['average']]
                ax.bar(x + i * width, values, width, label=op)

        ax.set_xlabel('Metrica')
        ax.set_ylabel('Distancia')
        ax.set_title(f'{instance_name} - Comparacao de Operadores de Cruzamento (30 execucoes)')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Melhor', 'Pior', 'Media'])
        ax.legend()

        # Adicionar linha do otimo se conhecido
        optimal = data.get('optimal')
        if optimal:
            ax.axhline(y=optimal, color='r', linestyle='--', label=f'Otimo ({optimal})')
            ax.legend()

        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{instance_name}_operator_comparison.png'
            plt.savefig(filepath, dpi=150)
            print(f"Salvo: {filepath}")

        plt.close()

    def plot_paper_comparison(self, instance_name: str, save: bool = True):
        """
        Cria grafico de barras comparando nossos resultados com os do artigo.

        Args:
            instance_name: Nome da instancia.
            save: Se deve salvar o grafico.
        """
        if instance_name not in self.our_results:
            print(f"Sem resultados para {instance_name}")
            return

        if instance_name not in self.PAPER_RESULTS:
            print(f"Sem resultados do artigo para {instance_name}")
            return

        our_data = self.our_results[instance_name]['results']
        paper_data = self.PAPER_RESULTS[instance_name]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        operators = ['PMX', 'OX', 'CX2']

        for idx, op in enumerate(operators):
            ax = axes[idx]

            if op in our_data and op in paper_data:
                metrics = ['best', 'worst', 'average']
                x = np.arange(len(metrics))
                width = 0.35

                our_vals = [our_data[op][m] for m in metrics]
                paper_vals = [paper_data[op][m] for m in metrics]

                ax.bar(x - width/2, our_vals, width, label='Nosso', color='steelblue')
                ax.bar(x + width/2, paper_vals, width, label='Artigo', color='coral')

                ax.set_xlabel('Metrica')
                ax.set_ylabel('Distancia')
                ax.set_title(f'{op}')
                ax.set_xticks(x)
                ax.set_xticklabels(['Melhor', 'Pior', 'Media'])
                ax.legend()

                # Adicionar linha do otimo
                optimal = paper_data.get('optimal')
                if optimal:
                    ax.axhline(y=optimal, color='green', linestyle='--', alpha=0.7)

        plt.suptitle(f'{instance_name} - Nossos Resultados vs Artigo (Otimo: {paper_data.get("optimal", "N/D")})')
        plt.tight_layout()

        if save:
            filepath = self.output_dir / f'{instance_name}_paper_comparison.png'
            plt.savefig(filepath, dpi=150)
            print(f"Salvo: {filepath}")

        plt.close()

    def plot_all_instances_summary(self, save: bool = True):
        """Cria grafico resumo mostrando os melhores resultados de todas as instancias."""
        if not self.our_results:
            print("Sem resultados para plotar")
            return

        instances = list(self.our_results.keys())
        operators = ['PMX', 'OX', 'CX2']

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(instances))
        width = 0.25

        for i, op in enumerate(operators):
            best_values = []
            for inst in instances:
                if op in self.our_results[inst]['results']:
                    best_values.append(self.our_results[inst]['results'][op]['best'])
                else:
                    best_values.append(0)

            ax.bar(x + i * width, best_values, width, label=op)

        # Adicionar valores otimos como marcadores
        optimals = [self.our_results[inst].get('optimal', 0) for inst in instances]
        ax.scatter(x + width, optimals, color='red', marker='*', s=100,
                   label='Otimo', zorder=5)

        ax.set_xlabel('Instancia')
        ax.set_ylabel('Melhor Distancia Encontrada')
        ax.set_title('Melhores Resultados por Instancia e Operador')
        ax.set_xticks(x + width)
        ax.set_xticklabels(instances, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()

        if save:
            filepath = self.output_dir / 'all_instances_summary.png'
            plt.savefig(filepath, dpi=150)
            print(f"Salvo: {filepath}")

        plt.close()

    def generate_results_table_latex(self) -> str:
        """Gera tabela LaTeX dos resultados."""
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\caption{Resultados Experimentais (30 execucoes cada)}")
        lines.append(r"\begin{tabular}{|l|c|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"Instancia & N & Otimo & Operador & Melhor & Pior & Media \\")
        lines.append(r"\hline")

        for inst_name, data in self.our_results.items():
            first = True
            for op in ['PMX', 'OX', 'CX2']:
                if op in data['results']:
                    r = data['results'][op]
                    if first:
                        lines.append(f"{inst_name} & {data['dimension']} & "
                                   f"{data.get('optimal', 'N/D')} & {op} & "
                                   f"{r['best']:.0f} & {r['worst']:.0f} & "
                                   f"{r['average']:.1f} \\\\")
                        first = False
                    else:
                        lines.append(f" &  &  & {op} & "
                                   f"{r['best']:.0f} & {r['worst']:.0f} & "
                                   f"{r['average']:.1f} \\\\")
            lines.append(r"\hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def generate_markdown_table(self) -> str:
        """Gera tabela Markdown dos resultados."""
        lines = []
        lines.append("## Resultados Experimentais (30 execucoes cada)\n")
        lines.append("| Instancia | N | Otimo | Operador | Melhor | Pior | Media | Gap% |")
        lines.append("|-----------|---|-------|----------|--------|------|-------|------|")

        for inst_name, data in self.our_results.items():
            for i, op in enumerate(['PMX', 'OX', 'CX2']):
                if op in data['results']:
                    r = data['results'][op]
                    optimal = data.get('optimal')
                    gap = ((r['best'] - optimal) / optimal * 100) if optimal else 0

                    if i == 0:
                        lines.append(f"| {inst_name} | {data['dimension']} | "
                                   f"{optimal or 'N/D'} | {op} | "
                                   f"{r['best']:.0f} | {r['worst']:.0f} | "
                                   f"{r['average']:.1f} | {gap:.1f}% |")
                    else:
                        lines.append(f"| | | | {op} | "
                                   f"{r['best']:.0f} | {r['worst']:.0f} | "
                                   f"{r['average']:.1f} | {gap:.1f}% |")

        return "\n".join(lines)


def main():
    """Gera todos os graficos comparativos."""
    results_file = "results/results_latest.json"

    if not Path(results_file).exists():
        print(f"Arquivo de resultados nao encontrado: {results_file}")
        return

    plotter = ComparisonPlotter(results_file)

    # Gerar graficos para cada instancia
    for instance in plotter.our_results.keys():
        print(f"\nGerando graficos para {instance}...")
        plotter.plot_operator_comparison(instance)
        plotter.plot_paper_comparison(instance)

    # Gerar grafico resumo
    plotter.plot_all_instances_summary()

    # Gerar tabelas
    print("\n" + "="*60)
    print("TABELA MARKDOWN")
    print("="*60)
    print(plotter.generate_markdown_table())

    print("\n" + "="*60)
    print("TABELA LATEX")
    print("="*60)
    print(plotter.generate_results_table_latex())


if __name__ == "__main__":
    main()
