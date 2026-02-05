"""
Análise estatística para experimentos do AG.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentStatistics:
    """Medidas estatísticas para resultados de experimentos."""

    @staticmethod
    def compute_statistics(values: List[float]) -> Dict[str, float]:
        """
        Calcula estatísticas básicas para uma lista de valores.

        Args:
            values: Lista de valores de aptidão de múltiplas execuções.

        Returns:
            Dicionário com estatísticas.
        """
        return {
            'best': min(values),
            'worst': max(values),
            'average': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75),
        }

    @staticmethod
    def compute_gap(obtained: float, optimal: float) -> float:
        """
        Calcula o gap percentual em relação à solução ótima.

        Args:
            obtained: Melhor solução encontrada.
            optimal: Valor ótimo conhecido.

        Returns:
            Percentual de gap.
        """
        if optimal == 0:
            return 0.0
        return ((obtained - optimal) / optimal) * 100

    @staticmethod
    def compute_hit_rate(values: List[float], target: float, tolerance: float = 0.0) -> float:
        """
        Calcula a taxa de acerto (fração de execuções que atingiram o alvo).

        Args:
            values: Lista de valores de aptidão.
            target: Valor alvo a ser atingido.
            tolerance: Diferença aceitável em relação ao alvo.

        Returns:
            Taxa de acerto como fração.
        """
        hits = sum(1 for v in values if abs(v - target) <= tolerance)
        return hits / len(values)

    @staticmethod
    def compare_operators(results: Dict[str, List[float]], optimal: Optional[float] = None) -> str:
        """
        Gera relatório de comparação para múltiplos operadores.

        Args:
            results: Dicionário mapeando nome do operador para lista de valores de aptidão.
            optimal: Valor ótimo conhecido (opcional).

        Returns:
            String de comparação formatada.
        """
        lines = []
        lines.append("="*70)
        lines.append(f"{'Operador':<10} {'Melhor':<10} {'Pior':<10} {'Média':<10} "
                    f"{'Desvio':<10} {'Gap%':<10}")
        lines.append("-"*70)

        for operator, values in results.items():
            stats = ExperimentStatistics.compute_statistics(values)
            gap = ""
            if optimal:
                gap = f"{ExperimentStatistics.compute_gap(stats['best'], optimal):.2f}%"

            lines.append(f"{operator:<10} {stats['best']:<10.2f} {stats['worst']:<10.2f} "
                        f"{stats['average']:<10.2f} {stats['std']:<10.2f} {gap:<10}")

        lines.append("="*70)
        return "\n".join(lines)

    @staticmethod
    def format_paper_comparison(our_results: Dict, paper_results: Dict) -> str:
        """
        Formata a comparação entre nossos resultados e os resultados do artigo.

        Args:
            our_results: Nossos resultados experimentais.
            paper_results: Resultados do artigo.

        Returns:
            String de comparação formatada.
        """
        lines = []
        lines.append("\n" + "="*80)
        lines.append("COMPARAÇÃO COM OS RESULTADOS DO ARTIGO")
        lines.append("="*80)

        lines.append(f"\n{'Operador':<8} | {'Métrica':<8} | {'Nosso':<12} | {'Artigo':<12} | {'Dif':<10}")
        lines.append("-"*60)

        for operator in our_results.keys():
            if operator not in paper_results:
                continue

            our = our_results[operator]
            paper = paper_results[operator]

            for metric in ['best', 'worst', 'average']:
                our_val = our.get(metric, 0)
                paper_val = paper.get(metric, 0)
                diff = our_val - paper_val
                diff_str = f"{diff:+.2f}"

                lines.append(f"{operator if metric == 'best' else '':<8} | "
                           f"{metric:<8} | {our_val:<12.2f} | {paper_val:<12.2f} | {diff_str:<10}")

            lines.append("-"*60)

        return "\n".join(lines)
