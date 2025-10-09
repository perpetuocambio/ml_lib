"""
Módulo para lectura y preprocesamiento de datos ecológicos
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

# Importar componentes generales de ml_lib
from ml_lib.core import TransformerInterface
from ml_lib.linalg import Matrix


class EcologicalDataReader:
    """
    Clase para leer y procesar diferentes tipos de datos ecológicos
    """
    
    @staticmethod
    def read_species_abundance_csv(
        file_path: Union[str, Path],
        species_column: str = 'species',
        site_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Lee un archivo CSV de abundancia de especies.
        
        Args:
            file_path: Ruta al archivo CSV
            species_column: Nombre de la columna que contiene los IDs de especies
            site_columns: Lista opcional de columnas de sitios
            
        Returns:
            Tuple de (datos, lista de especies, lista de sitios)
        """
        df = pd.read_csv(file_path)
        
        # Identificar especies y sitios
        if species_column in df.columns:
            species = df[species_column].tolist()
            df = df.set_index(species_column)
        else:
            species = [f"Species_{i}" for i in range(len(df))]
            df.index = species
        
        # Si no se especifican sitios, usar todas las columnas restantes
        if site_columns is None:
            sites = df.columns.tolist()
        else:
            sites = site_columns
            df = df[site_columns]
        
        return df, species, sites
    
    @staticmethod
    def read_enviromental_data_csv(
        file_path: Union[str, Path],
        site_column: str = 'site'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Lee un archivo CSV de datos ambientales.
        
        Args:
            file_path: Ruta al archivo CSV
            site_column: Nombre de la columna que contiene los IDs de sitios
            
        Returns:
            Tuple de (datos ambientales, lista de sitios)
        """
        df = pd.read_csv(file_path)
        
        if site_column in df.columns:
            sites = df[site_column].tolist()
            df = df.set_index(site_column)
        else:
            sites = [f"Site_{i}" for i in range(len(df))]
            df.index = sites
        
        return df, sites


class AbundanceNormalizer(TransformerInterface[np.ndarray]):
    """
    Transformador para normalizar datos de abundancia usando técnicas de ml_lib
    """
    
    def __init__(self, method: str = 'relative'):
        self.method = method
        self.fitted = False
        self.stats_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'AbundanceNormalizer':
        """Ajusta el transformador a los datos."""
        if self.method == 'relative':
            # Calcular totales por columna para normalización relativa
            self.stats_ = np.sum(X, axis=0, keepdims=True)
            self.stats_[self.stats_ == 0] = 1  # Evitar división por cero
        elif self.method in ['sqrt', 'log']:
            # No requiere ajuste previo
            pass
        else:
            raise ValueError(f"Método de normalización no soportado: {self.method}")
        
        self.fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la transformación a los datos."""
        if not self.fitted:
            raise ValueError("El transformador debe ser ajustado primero")
        
        if self.method == 'relative':
            return X / self.stats_
        elif self.method == 'presence_absence':
            return (X > 0).astype(float)
        elif self.method == 'sqrt':
            return np.sqrt(X)
        elif self.method == 'log':
            return np.log1p(X)  # log(x+1) para manejar ceros
        else:
            raise ValueError(f"Método de normalización no soportado: {self.method}")
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Aplica la transformación inversa."""
        if self.method == 'relative' and self.fitted:
            return X * self.stats_
        else:
            # Para otros métodos, no hay transformación inversa simple
            raise NotImplementedError(f"Inversa no implementada para el método: {self.method}")


class DataFilter:
    """
    Clase para filtrar datos ecológicos basados en criterios generales
    """
    
    @staticmethod
    def filter_by_occurrence(
        data_matrix: np.ndarray,
        item_names: List[str],
        min_occurrence_proportion: float = 0.1,
        min_total_value: float = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Filtra filas basadas en criterios de ocurrencia y valor total.
        
        Args:
            data_matrix: Matriz de datos (items x features)
            item_names: Lista de nombres de items
            min_occurrence_proportion: Mínima proporción de features donde debe aparecer el item
            min_total_value: Mínimo valor total para pasar el filtro
            
        Returns:
            Tuple de (matriz filtrada, nombres de items filtrados)
        """
        # Calcular estadísticas por fila (item)
        occurrence_proportion = np.sum(data_matrix != 0, axis=1) / data_matrix.shape[1]
        total_values = np.sum(data_matrix, axis=1)
        
        # Aplicar filtros
        valid_items = (
            (occurrence_proportion >= min_occurrence_proportion) &
            (total_values >= min_total_value)
        )
        
        filtered_matrix = data_matrix[valid_items]
        filtered_names = [item_names[i] for i, is_valid in enumerate(valid_items) if is_valid]
        
        return filtered_matrix, filtered_names