"""
Módulo para análisis ecológico usando componentes generales de ml_lib
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

# Importar componentes generales de ml_lib
from ml_lib.core import (
    EstimatorInterface,
    BaseModel,
    ModelConfig,
    ValidationService,
    ErrorHandler,
    LoggingService,
)
from ml_lib.linalg.services.linalg import Matrix, BLASService, DecompositionService

# Importar componentes generales de scikit-learn para demostración
# En una implementación completa, estos serían reemplazados por componentes de ml_lib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from ecoml_analyzer.data import AbundanceNormalizer


class EcologicalAnalyzer:
    """
    Analizador ecológico usando componentes generales de ml_lib aplicados al dominio ecológico
    """

    def __init__(self):
        # Configurar servicios
        self.logger_service = LoggingService("EcologicalAnalyzer")
        self.logger = self.logger_service.get_logger()
        self.error_handler = ErrorHandler(self.logger)
        self.validation_service = ValidationService(self.logger)

        # Componentes generales que se aplican al dominio ecológico
        self.scaler = (
            StandardScaler()
        )  # Este sería reemplazado por un componente de ml_lib
        self.pca_components = None
        self.clusters = None

        self.logger.info(
            "EcologicalAnalyzer inicializado - usando componentes generales de ml_lib"
        )

    def analyze_composition(self, data_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analiza la composición de datos (aplicable a especies, sitios, etc.).

        Args:
            data_matrix: Matriz de datos (elementos x características)

        Returns:
            Diccionario con resultados del análisis de composición
        """
        # Validar entradas usando servicio de validación general
        self.validation_service.validate_input_shape(
            data_matrix, 2, "analyze_composition"
        )

        # Cálculos generales aplicables a datos multidimensionales
        total_values = np.sum(data_matrix, axis=0)  # Suma por característica
        relative_values = data_matrix / (
            np.sum(data_matrix, axis=1, keepdims=True) + 1e-8
        )  # Proporciones por fila
        richness = np.sum(
            data_matrix > 0, axis=1
        )  # Número de características presentes por fila
        diversity = -np.sum(
            relative_values * np.log(relative_values + 1e-8), axis=1
        )  # Diversidad tipo Shannon

        results = {
            "total_values": total_values,
            "relative_values": relative_values,
            "richness": richness,
            "diversity": diversity,
            "abundance_distribution": {
                "mean": np.mean(data_matrix),
                "std": np.std(data_matrix),
                "min": np.min(data_matrix),
                "max": np.max(data_matrix),
                "zeros_ratio": np.sum(data_matrix == 0) / data_matrix.size,
            },
        }

        self.logger.info(
            f"Análisis de composición completado para {data_matrix.shape[0]} elementos y {data_matrix.shape[1]} características"
        )
        return results

    def perform_dimensionality_reduction(
        self,
        data_matrix: np.ndarray,
        method: str = "pca",
        n_components: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Realiza reducción de dimensionalidad usando técnicas generales aplicadas al dominio ecológico.

        Args:
            data_matrix: Matriz de datos (muestras x características)
            method: Método de reducción ('pca', 'custom_pca' usando ml_lib)
            n_components: Número de componentes a mantener

        Returns:
            Diccionario con resultados de la reducción de dimensionalidad
        """
        # Validar entradas
        self.validation_service.validate_input_shape(
            data_matrix, 2, "perform_dimensionality_reduction"
        )

        # Estandarizar datos
        X_scaled = self.scaler.fit_transform(data_matrix)

        if method == "pca":
            # Usando PCA de scikit-learn como ejemplo (en la realidad, usaríamos nuestro PCA de ml_lib)
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X_scaled)

            results = {
                "transformed_data": X_transformed,
                "components": pca.components_,
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
                "method": "PCA",
                "n_components": X_transformed.shape[1],
            }

        elif method == "custom_pca":
            # Implementación usando componentes de ml_lib
            # Calcular covarianza
            cov_matrix = BLASService.gemm(X_scaled.T, X_scaled) / (
                X_scaled.shape[0] - 1
            )

            # Usar descomposición SVD de nuestra biblioteca
            try:
                U, S, Vt = DecompositionService.svd_decomposition(cov_matrix)
                # Los componentes principales son las filas de Vt
                components = Vt

                # Transformar datos
                X_transformed = X_scaled @ components.T

                # Calcular varianza explicada
                explained_variance_ratio = S.diagonal() / np.sum(S.diagonal())
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

                results = {
                    "transformed_data": X_transformed,
                    "components": components,
                    "explained_variance_ratio": explained_variance_ratio,
                    "cumulative_variance_ratio": cumulative_variance_ratio,
                    "method": "Custom PCA (ml_lib)",
                    "n_components": X_transformed.shape[1],
                }
            except Exception as e:
                self.logger.warning(
                    f"Error en SVD personalizado, usando PCA de sklearn: {e}"
                )
                # Fallback a sklearn PCA
                pca = PCA(n_components=n_components)
                X_transformed = pca.fit_transform(X_scaled)

                results = {
                    "transformed_data": X_transformed,
                    "components": pca.components_,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "cumulative_variance_ratio": np.cumsum(
                        pca.explained_variance_ratio_
                    ),
                    "method": "PCA (sklearn fallback)",
                    "n_components": X_transformed.shape[1],
                }

        else:
            raise ValueError(
                f"Método de reducción de dimensionalidad no soportado: {method}"
            )

        # Guardar componentes para uso posterior
        self.pca_components = results["transformed_data"]

        self.logger.info(f"Reducción de dimensionalidad ({method}) completada")
        return results

    def perform_clustering(
        self, data_matrix: np.ndarray, n_clusters: int = 2, method: str = "kmeans"
    ) -> Dict[str, Any]:
        """
        Realiza clustering de muestras usando técnicas generales aplicadas al dominio ecológico.

        Args:
            data_matrix: Matriz de datos (muestras x características)
            n_clusters: Número de clusters
            method: Método de clustering

        Returns:
            Diccionario con resultados del clustering
        """
        # Validar entradas
        self.validation_service.validate_input_shape(
            data_matrix, 2, "perform_clustering"
        )

        # Estandarizar datos
        X_scaled = self.scaler.fit_transform(data_matrix)

        if method == "kmeans":
            # Usando KMeans de scikit-learn como ejemplo (en la realidad, usaríamos nuestro KMeans de ml_lib)
            clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering_model.fit_predict(X_scaled)

            # Calcular métricas de clustering
            silhouette = (
                silhouette_score(X_scaled, cluster_labels)
                if len(np.unique(cluster_labels)) > 1
                else 0
            )
            calinski = (
                calinski_harabasz_score(X_scaled, cluster_labels)
                if len(np.unique(cluster_labels)) > 1
                else 0
            )

            results = {
                "cluster_labels": cluster_labels,
                "silhouette_score": silhouette,
                "calinski_harabasz_score": calinski,
                "n_clusters": n_clusters,
                "clustering_method": method,
                "cluster_centers": clustering_model.cluster_centers_,
            }

        else:
            raise ValueError(f"Método de clustering no soportado: {method}")

        # Guardar clusters para uso posterior
        self.clusters = cluster_labels

        self.logger.info(
            f"Clustering completado ({method}): {silhouette:.3f} silhouette, {calinski:.3f} Calinski-Harabasz"
        )
        return results


class EcologicalModelEstimator(EstimatorInterface[np.ndarray, np.ndarray]):
    """
    Estimador ecológico que implementa la interfaz general de ml_lib
    Este es un ejemplo de cómo se aplican técnicas generales al dominio ecológico
    """

    def __init__(self, model_type: str = "logistic", **kwargs):
        self.model_type = model_type
        self.model = None
        self.is_fitted = False
        self.kwargs = kwargs

        # Configurar servicios
        logger_service = LoggingService("EcologicalModelEstimator")
        self.logger = logger_service.get_logger()
        self.validation_service = ValidationService(self.logger)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EcologicalModelEstimator":
        """Ajusta el modelo a los datos ecológicos."""
        # Validar entradas
        self.validation_service.validate_input_shape(X, 2, "fit")
        self.validation_service.validate_input_shape(y, 1, "fit")
        self.validation_service.validate_same_length(X, y, "fit")

        if self.model_type == "logistic":
            # Usando LogisticRegression de scikit-learn como ejemplo (en la realidad, usaríamos nuestro modelo de ml_lib)
            self.model = LogisticRegression(**self.kwargs)
            self.model.fit(X, y)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

        self.is_fitted = True
        self.logger.info(f"Modelo {self.model_type} ajustado exitosamente")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Realiza predicciones en datos ecológicos."""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser ajustado antes de hacer predicciones")

        return self.model.predict(X)

    def get_params(self) -> dict:
        """Obtiene los parámetros del modelo."""
        return {
            "model_type": self.model_type,
            "model_params": getattr(self.model, "get_params", lambda: {})(),
            "is_fitted": self.is_fitted,
        }

    def set_params(self, **params) -> "EcologicalModelEstimator":
        """Establece los parámetros del modelo."""
        if "model_type" in params:
            self.model_type = params["model_type"]
        # Actualizar otros parámetros según sea necesario
        return self
