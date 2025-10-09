"""
Módulo para visualización ecológica que utiliza componentes generales de ml_lib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Importar componentes generales de visualización de ml_lib
from ml_lib.visualization import GeneralVisualization, VisualizationFactory, PlotConfig
from ml_lib.core import LoggingService
from ml_lib.visualization.models import ScatterPlotData, LinePlotData, BarPlotData, HeatmapData

# Asegurarse de que plotly está disponible
try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class EcologicalVisualizer:
    """
    Clase para visualizar resultados ecológicos usando componentes generales de ml_lib
    """
    
    def __init__(self, style: str = "seaborn"):
        """
        Inicializa el visualizador con un estilo específico.
        
        Args:
            style: Estilo de visualización ("seaborn", "matplotlib", "plotly")
        """
        # Configurar servicios generales de ml_lib
        self.logger_service = LoggingService("EcologicalVisualizer")
        self.logger = self.logger_service.get_logger()
        
        # Crear instancia de visualización general desde la fábrica
        self.general_viz = VisualizationFactory.create_visualization(
            PlotConfig(style=style)
        )
        
        if style == "seaborn":
            sns.set_style("whitegrid")
        elif style == "matplotlib":
            plt.style.use('default')
        
        self.style = style
        self.figures = []
    
    def plot_species_abundance_heatmap(
        self,
        abundance_matrix: np.ndarray,
        species_names: List[str],
        site_names: List[str],
        title: str = "Heatmap de Abundancia de Especies",
        figsize: Tuple[int, int] = (12, 8),
        normalize: bool = True
    ) -> plt.Figure:
        """
        Crea un heatmap de abundancia de especies usando componentes generales de ml_lib.
        
        Args:
            abundance_matrix: Matriz de abundancia (especies x sitios)
            species_names: Nombres de especies
            site_names: Nombres de sitios
            title: Título del gráfico
            figsize: Tamaño de la figura
            normalize: Si normalizar los datos
            
        Returns:
            Figura matplotlib
        """
        # Normalizar por sitio si se solicita
        if normalize:
            normalized_matrix = abundance_matrix / (np.sum(abundance_matrix, axis=0, keepdims=True) + 1e-8)
        else:
            normalized_matrix = abundance_matrix
        
        # Usar componente general de visualización para crear el heatmap
        fig = self.general_viz.plot_heatmap(
            normalized_matrix,
            x_labels=site_names,
            y_labels=species_names,
            title=title
        )
        
        self.figures.append(fig)
        return fig
    
    def plot_composition_analysis(
        self,
        composition_results: Dict[str, Any],  # Usar Dict en lugar de np.ndarray para mayor flexibilidad
        labels: List[str],
        title: str = "Análisis de Composición General",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualiza resultados generales de análisis de composición aplicados al dominio ecológico
        usando componentes generales de ml_lib.
        
        Args:
            composition_results: Diccionario con resultados de análisis de composición
            labels: Etiquetas para las muestras
            title: Título del gráfico
            figsize: Tamaño de la figura
            
        Returns:
            Figura matplotlib
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title)
        
        # Riqueza
        if 'richness' in composition_results and isinstance(composition_results['richness'], np.ndarray):
            # Usar componente general de visualización para crear el gráfico de barras
            x_vals = np.arange(len(labels))
            heights = composition_results['richness']
            axes[0, 0].bar(x_vals, heights)
            axes[0, 0].set_title('Riqueza')
            axes[0, 0].set_xlabel('Muestras')
            axes[0, 0].set_ylabel('Número de elementos')
            axes[0, 0].set_xticks(x_vals)
            axes[0, 0].set_xticklabels(labels, rotation=45, ha="right")
        
        # Diversidad
        if 'diversity' in composition_results and isinstance(composition_results['diversity'], np.ndarray):
            x_vals = np.arange(len(labels))
            heights = composition_results['diversity']
            axes[0, 1].bar(x_vals, heights)
            axes[0, 1].set_title('Diversidad')
            axes[0, 1].set_xlabel('Muestras')
            axes[0, 1].set_ylabel('Índice de Diversidad')
            axes[0, 1].set_xticks(x_vals)
            axes[0, 1].set_xticklabels(labels, rotation=45, ha="right")
        
        # Abundancia total
        if 'total_values' in composition_results and isinstance(composition_results['total_values'], np.ndarray):
            x_vals = np.arange(len(labels))
            heights = composition_results['total_values']
            axes[1, 0].bar(x_vals, heights)
            axes[1, 0].set_title('Abundancia Total')
            axes[1, 0].set_xlabel('Muestras')
            axes[1, 0].set_ylabel('Suma Total')
            axes[1, 0].set_xticks(x_vals)
            axes[1, 0].set_xticklabels(labels, rotation=45, ha="right")
        
        # Distribución
        if 'abundance_distribution' in composition_results:
            dist = composition_results['abundance_distribution']
            summary_data = [dist.get('mean', 0), dist.get('std', 0), dist.get('min', 0), dist.get('max', 0)]
            summary_labels = ['Media', 'Std', 'Mín', 'Máx']
            x_vals = np.arange(len(summary_labels))
            axes[1, 1].bar(x_vals, summary_data)
            axes[1, 1].set_title('Resumen de Distribución')
            axes[1, 1].set_ylabel('Valor')
            axes[1, 1].set_xticks(x_vals)
            axes[1, 1].set_xticklabels(summary_labels)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_dimensionality_reduction(
        self,
        dim_red_results: Dict[str, Any],
        labels: Optional[List[str]] = None,
        title: str = "Análisis de Reducción de Dimensionalidad",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualiza los resultados de reducción de dimensionalidad usando componentes generales.
        
        Args:
            dim_red_results: Diccionario con resultados de reducción de dimensionalidad
            labels: Etiquetas opcionales para las muestras
            title: Título del gráfico
            figsize: Tamaño de la figura
            
        Returns:
            Figura matplotlib
        """
        X_transformed = dim_red_results['transformed_data']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if X_transformed.shape[1] < 2:
            # Si solo hay un componente, hacer un gráfico 1D
            if labels is not None:
                scatter = ax.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]), 
                          c=range(len(X_transformed)), cmap='viridis', alpha=0.7, s=50)
                for i, label in enumerate(labels):
                    ax.annotate(label, (X_transformed[i, 0], 0), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8, alpha=0.7)
            else:
                scatter = ax.scatter(X_transformed[:, 0], np.zeros_like(X_transformed[:, 0]), 
                          alpha=0.7, s=50)
            ax.set_xlabel(f'{dim_red_results["method"]} Component 1')
            ax.set_ylabel('Dimensión 0')
        else:
            # Gráfico 2D
            if labels is not None:
                unique_labels = list(set(labels))
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    ax.scatter(
                        X_transformed[mask, 0],
                        X_transformed[mask, 1],
                        label=label,
                        alpha=0.7,
                        s=50
                    )
                ax.legend()
            else:
                ax.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7, s=50)
            
            method = dim_red_results["method"]
            if "explained_variance_ratio" in dim_red_results:
                exp_var = dim_red_results["explained_variance_ratio"]
                ax.set_xlabel(f'{method}1 ({exp_var[0]:.2%} varianza)')
                ax.set_ylabel(f'{method}2 ({exp_var[1]:.2%} varianza)' if len(exp_var) > 1 else f'{method}2')
            else:
                ax.set_xlabel(f'{method} Component 1')
                ax.set_ylabel(f'{method} Component 2')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_clustering_results(
        self,
        clustering_results: Dict[str, Any],
        data_matrix: np.ndarray,
        labels: List[str],
        title: str = "Resultados de Clustering General",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Visualiza resultados del clustering general usando componentes generales de ml_lib.
        
        Args:
            clustering_results: Diccionario con resultados del clustering
            data_matrix: Matriz de datos original (muestras x características)
            labels: Nombres de las muestras
            title: Título del gráfico
            figsize: Tamaño de la figura
            
        Returns:
            Figura matplotlib
        """
        cluster_labels = clustering_results['cluster_labels']
        
        # Realizar PCA para visualización 2D
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Estandarizar datos
        X_scaled = StandardScaler().fit_transform(data_matrix)
        
        # Aplicar PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colorear puntos por cluster
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels,
            cmap='tab10',
            alpha=0.7,
            s=60
        )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        ax.set_title(f'{title} (K={clustering_results["n_clusters"]}) - {clustering_results["clustering_method"]}')
        ax.grid(True, alpha=0.3)
        
        # Añadir leyenda con nombres de muestras
        for i, sample_name in enumerate(labels):
            ax.annotate(sample_name, (X_pca[i, 0], X_pca[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7)
        
        # Añadir leyenda de clusters
        legend1 = ax.legend(*scatter.legend_elements(),
                           title="Clusters", loc="center left", bbox_to_anchor=(1, 0.5))
        ax.add_artist(legend1)
        
        # Añadir métricas como texto
        textstr = f'Silhouette: {clustering_results["silhouette_score"]:.3f}\n' \
                  f'C-H Score: {clustering_results["calinski_harabasz_score"]:.1f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def plot_model_performance(
        self,
        model_results: Dict[str, Any],
        title: str = "Desempeño del Modelo",
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """
        Visualiza el desempeño de modelos predictivos usando componentes generales.
        
        Args:
            model_results: Diccionario con resultados del modelo
            title: Título del gráfico
            figsize: Tamaño de la figura
            
        Returns:
            Figura matplotlib
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Suponiendo que tenemos predicciones y valores verdaderos
        if 'predictions' in model_results and 'true_values' in model_results:
            y_pred = model_results['predictions']
            y_true = model_results['true_values']
            
            # Gráfico de dispersión de predicciones vs valores verdaderos
            ax.scatter(y_true, y_pred, alpha=0.6)
            ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            ax.set_xlabel('Valores Verdaderos')
            ax.set_ylabel('Predicciones')
            ax.set_title(title)
            
            # Añadir métricas como texto
            accuracy = model_results.get('accuracy', 0)
            textstr = f'Precisión: {accuracy:.3f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        self.figures.append(fig)
        return fig
    
    def save_all_figures(self, output_dir: str):
        """
        Guarda todas las figuras generadas usando componentes generales de ml_lib.
        
        Args:
            output_dir: Directorio donde guardar las figuras
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            self.general_viz.save_plot(fig, f"{output_dir}/figure_{i+1}.png")
        
        print(f"Guardadas {len(self.figures)} figuras en {output_dir}")


class InteractiveEcologicalVisualizer:
    """
    Visualizador interactivo que podría usar componentes generales de ml_lib
    """
    
    def __init__(self):
        self.logger_service = LoggingService("InteractiveEcologicalVisualizer")
        self.logger = self.logger_service.get_logger()
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly no está disponible. Instala 'plotly' para visualizaciones interactivas.")
    
    def plot_interactive_dimensionality_reduction(
        self,
        dim_red_results: Dict[str, Any],
        labels: Optional[List[str]] = None,
        metadata: Optional[pd.DataFrame] = None,
        title: str = "Reducción de Dimensionalidad Interactiva"
    ) -> go.Figure:
        """
        Crea un análisis de reducción de dimensionalidad interactivo con Plotly.
        
        Args:
            dim_red_results: Diccionario con resultados de reducción de dimensionalidad
            labels: Nombres de muestras (opcional)
            metadata: DataFrame opcional con metadatos
            title: Título del gráfico
            
        Returns:
            Figura Plotly
        """
        X_transformed = dim_red_results['transformed_data']
        
        # Crear DataFrame para plotly
        df = pd.DataFrame({
            'Comp1': X_transformed[:, 0],
            'Comp2': X_transformed[:, 1] if X_transformed.shape[1] > 1 else np.zeros(len(X_transformed))
        })
        
        if labels is not None:
            df['Sample'] = labels
            hover_data_cols = ['Sample']
        else:
            df['Sample_ID'] = [f'Sample_{i}' for i in range(len(X_transformed))]
            hover_data_cols = ['Sample_ID']
        
        if metadata is not None:
            for col in metadata.columns:
                df[col] = metadata[col]
                hover_data_cols.append(col)
        
        fig = px.scatter(
            df,
            x='Comp1',
            y='Comp2',
            title=title,
            hover_data=hover_data_cols
        )
        
        method = dim_red_results["method"]
        if "explained_variance_ratio" in dim_red_results:
            exp_var = dim_red_results["explained_variance_ratio"]
            fig.update_layout(
                xaxis_title=f'{method}1 ({exp_var[0]:.2%} varianza)',
                yaxis_title=f'{method}2 ({exp_var[1]:.2%} varianza)' if len(exp_var) > 1 else f'{method}2',
                width=800,
                height=600
            )
        else:
            fig.update_layout(
                xaxis_title=f'{method} Component 1',
                yaxis_title=f'{method} Component 2',
                width=800,
                height=600
            )
        
        return fig