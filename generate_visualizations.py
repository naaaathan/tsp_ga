"""
Gerar todas as visualizacoes para os resultados do experimento GA para o TSP.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from visualization.comparison import ComparisonPlotter


def main():
    """Gerar todas as visualizacoes."""
    print("="*60)
    print("GERANDO VISUALIZACOES")
    print("="*60)

    results_file = "results/results_latest.json"

    if not Path(results_file).exists():
        print(f"Arquivo de resultados nao encontrado: {results_file}")
        print("Execute os experimentos primeiro: python run_experiments.py")
        return

    plotter = ComparisonPlotter(results_file)

    # Gerar graficos para cada instancia
    for instance in plotter.our_results.keys():
        print(f"\nGerando graficos para {instance}...")
        plotter.plot_operator_comparison(instance)
        plotter.plot_paper_comparison(instance)

    # Gerar grafico resumo
    print("\nGerando grafico resumo...")
    plotter.plot_all_instances_summary()

    # Gerar tabelas
    print("\n" + "="*60)
    print("TABELA DE RESULTADOS (Markdown)")
    print("="*60)
    print(plotter.generate_markdown_table())

    # Salvar tabela markdown em arquivo
    md_table = plotter.generate_markdown_table()
    with open("results/results_table.md", "w") as f:
        f.write(md_table)
    print("\nTabela markdown salva em: results/results_table.md")

    print("\n" + "="*60)
    print("VISUALIZACOES CONCLUIDAS")
    print("="*60)
    print(f"Graficos salvos em: {plotter.output_dir}")


if __name__ == "__main__":
    main()
