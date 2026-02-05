"""
Cálculo de fitness para o TSP.

O fitness de um cromossomo é a distância total do percurso.
Para o TSP, queremos MINIMIZAR a distância total.
"""

import numpy as np
from typing import Union, Optional
from .chromosome import Chromosome, Population


class FitnessCalculator:
    """
    Calcula o fitness (distância total do percurso) para cromossomos do TSP.
    """

    def __init__(self, distance_matrix: np.ndarray):
        """
        Inicializa o calculador de fitness.

        Args:
            distance_matrix: Array numpy 2D onde distance_matrix[i][j] é a
                           distância da cidade i até a cidade j.
        """
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)

        # Valida a matriz de distâncias
        if distance_matrix.shape[0] != distance_matrix.shape[1]:
            raise ValueError("Distance matrix must be square")

    def calculate(self, chromosome: Chromosome) -> float:
        """
        Calcula a distância total do percurso para um cromossomo.

        Args:
            chromosome: O cromossomo a ser avaliado.

        Returns:
            Distância total do percurso.
        """
        total_distance = 0.0
        genes = chromosome.genes

        # Soma distâncias entre cidades consecutivas
        for i in range(len(genes) - 1):
            total_distance += self.distance_matrix[genes[i]][genes[i + 1]]

        # Adiciona distância da última cidade de volta à primeira
        total_distance += self.distance_matrix[genes[-1]][genes[0]]

        # Armazena fitness no cromossomo
        chromosome.fitness = total_distance

        return total_distance

    def evaluate_population(self, population: Population) -> None:
        """
        Avalia o fitness de todos os cromossomos em uma população.

        Args:
            population: A população a ser avaliada.
        """
        for chromosome in population:
            if chromosome.fitness is None:
                self.calculate(chromosome)

    def get_tour_distance(self, tour: list) -> float:
        """
        Calcula a distância de um percurso dado como lista de índices de cidades.

        Args:
            tour: Lista de índices de cidades.

        Returns:
            Distância total do percurso.
        """
        total_distance = 0.0

        for i in range(len(tour) - 1):
            total_distance += self.distance_matrix[tour[i]][tour[i + 1]]

        # Retorno ao início
        total_distance += self.distance_matrix[tour[-1]][tour[0]]

        return total_distance

    @staticmethod
    def create_from_coordinates(coordinates: np.ndarray,
                                distance_type: str = 'EUC_2D') -> 'FitnessCalculator':
        """
        Cria um FitnessCalculator a partir de coordenadas de cidades.

        Args:
            coordinates: Array Nx2 de coordenadas (x, y) para cada cidade.
            distance_type: Tipo de cálculo de distância ('EUC_2D', 'CEIL_2D', 'GEO', etc.)

        Returns:
            Instância de FitnessCalculator.
        """
        n_cities = len(coordinates)
        distance_matrix = np.zeros((n_cities, n_cities))

        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    if distance_type == 'EUC_2D':
                        # Distância euclidiana
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
                    elif distance_type == 'CEIL_2D':
                        # Teto da distância euclidiana
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        distance_matrix[i][j] = np.ceil(np.sqrt(dx**2 + dy**2))
                    elif distance_type == 'ATT':
                        # Distância pseudo-euclidiana (usada em att48, att532)
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        rij = np.sqrt((dx**2 + dy**2) / 10.0)
                        tij = int(np.round(rij))
                        if tij < rij:
                            distance_matrix[i][j] = tij + 1
                        else:
                            distance_matrix[i][j] = tij
                    elif distance_type == 'GEO':
                        # Distância geográfica
                        distance_matrix[i][j] = FitnessCalculator._geo_distance(
                            coordinates[i], coordinates[j]
                        )
                    else:
                        # Padrão: euclidiana
                        dx = coordinates[i][0] - coordinates[j][0]
                        dy = coordinates[i][1] - coordinates[j][1]
                        distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)

        return FitnessCalculator(distance_matrix)

    @staticmethod
    def _geo_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Calcula a distância geográfica entre duas coordenadas."""
        PI = 3.141592
        RRR = 6378.388

        def to_radians(coord):
            deg = int(coord[0])
            min_val = coord[0] - deg
            lat = PI * (deg + 5.0 * min_val / 3.0) / 180.0

            deg = int(coord[1])
            min_val = coord[1] - deg
            lon = PI * (deg + 5.0 * min_val / 3.0) / 180.0

            return lat, lon

        lat1, lon1 = to_radians(coord1)
        lat2, lon2 = to_radians(coord2)

        q1 = np.cos(lon1 - lon2)
        q2 = np.cos(lat1 - lat2)
        q3 = np.cos(lat1 + lat2)

        return int(RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    @staticmethod
    def create_from_explicit_matrix(matrix: np.ndarray) -> 'FitnessCalculator':
        """
        Cria um FitnessCalculator a partir de uma matriz de distâncias explícita.

        Args:
            matrix: Matriz de distâncias NxN.

        Returns:
            Instância de FitnessCalculator.
        """
        return FitnessCalculator(matrix.astype(np.float64))
