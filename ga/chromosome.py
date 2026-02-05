"""
Representação de cromossomo para o TSP usando representação por caminho.

Representação por Caminho:
- Forma mais natural de representar um percurso
- Exemplo: percurso 1→4→8→2→5→3→6→7 = [1, 4, 8, 2, 5, 3, 6, 7]
- Cada gene representa um índice de cidade
- Cromossomo válido visita cada cidade exatamente uma vez
"""

import numpy as np
from typing import List, Optional
import copy


class Chromosome:
    """
    Representa um indivíduo (solução) no algoritmo genético.
    Usa representação por caminho onde o cromossomo é uma permutação de cidades.
    """

    def __init__(self, genes: Optional[List[int]] = None, n_cities: Optional[int] = None):
        """
        Inicializa um cromossomo.

        Args:
            genes: Lista de índices de cidades representando o percurso. Se None, inicialização aleatória.
            n_cities: Número de cidades (obrigatório se genes for None).
        """
        if genes is not None:
            self.genes = np.array(genes, dtype=np.int32)
            self.n_cities = len(genes)
        elif n_cities is not None:
            self.n_cities = n_cities
            self.genes = self._random_init()
        else:
            raise ValueError("Either genes or n_cities must be provided")

        self.fitness: Optional[float] = None

    def _random_init(self) -> np.ndarray:
        """Gera uma permutação aleatória de cidades (indexação a partir de 0)."""
        return np.random.permutation(self.n_cities).astype(np.int32)

    def is_valid(self) -> bool:
        """
        Verifica se o cromossomo representa um percurso válido.
        Um percurso válido visita cada cidade exatamente uma vez.
        """
        if len(self.genes) != self.n_cities:
            return False
        return len(set(self.genes)) == self.n_cities and \
               min(self.genes) == 0 and \
               max(self.genes) == self.n_cities - 1

    def copy(self) -> 'Chromosome':
        """Cria uma cópia profunda deste cromossomo."""
        new_chromosome = Chromosome(genes=self.genes.copy())
        new_chromosome.fitness = self.fitness
        return new_chromosome

    def get_tour(self) -> List[int]:
        """Retorna o percurso como uma lista de índices de cidades."""
        return self.genes.tolist()

    def get_edges(self) -> List[tuple]:
        """
        Retorna o percurso como uma lista de arestas (cidade_i, cidade_j).
        Inclui a aresta de retorno à cidade inicial.
        """
        edges = []
        for i in range(self.n_cities - 1):
            edges.append((self.genes[i], self.genes[i + 1]))
        # Retorno à cidade inicial
        edges.append((self.genes[-1], self.genes[0]))
        return edges

    def __len__(self) -> int:
        return self.n_cities

    def __getitem__(self, idx: int) -> int:
        return self.genes[idx]

    def __setitem__(self, idx: int, value: int):
        self.genes[idx] = value
        self.fitness = None  # Invalida fitness quando os genes mudam

    def __repr__(self) -> str:
        return f"Chromosome(tour={self.genes.tolist()}, fitness={self.fitness})"

    def __eq__(self, other: 'Chromosome') -> bool:
        if not isinstance(other, Chromosome):
            return False
        return np.array_equal(self.genes, other.genes)

    def __hash__(self):
        return hash(tuple(self.genes))


class Population:
    """
    Representa uma população de cromossomos.
    """

    def __init__(self, size: int, n_cities: int, chromosomes: Optional[List[Chromosome]] = None):
        """
        Inicializa uma população.

        Args:
            size: Tamanho da população.
            n_cities: Número de cidades na instância do TSP.
            chromosomes: Lista opcional de cromossomos pré-criados.
        """
        self.size = size
        self.n_cities = n_cities

        if chromosomes is not None:
            self.chromosomes = chromosomes
        else:
            self.chromosomes = [Chromosome(n_cities=n_cities) for _ in range(size)]

    def get_best(self) -> Chromosome:
        """Retorna o cromossomo com o melhor (menor) fitness."""
        return min(self.chromosomes, key=lambda c: c.fitness if c.fitness is not None else float('inf'))

    def get_worst(self) -> Chromosome:
        """Retorna o cromossomo com o pior (maior) fitness."""
        return max(self.chromosomes, key=lambda c: c.fitness if c.fitness is not None else float('-inf'))

    def get_average_fitness(self) -> float:
        """Retorna o fitness médio da população."""
        valid_fitness = [c.fitness for c in self.chromosomes if c.fitness is not None]
        return np.mean(valid_fitness) if valid_fitness else 0.0

    def sort_by_fitness(self, reverse: bool = False):
        """Ordena a população por fitness (ascendente por padrão, melhor primeiro)."""
        self.chromosomes.sort(
            key=lambda c: c.fitness if c.fitness is not None else float('inf'),
            reverse=reverse
        )

    def __len__(self) -> int:
        return len(self.chromosomes)

    def __getitem__(self, idx: int) -> Chromosome:
        return self.chromosomes[idx]

    def __setitem__(self, idx: int, chromosome: Chromosome):
        self.chromosomes[idx] = chromosome

    def __iter__(self):
        return iter(self.chromosomes)
