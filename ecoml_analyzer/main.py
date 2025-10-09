"""
Aplicación principal de EcoML Analyzer
Demostración de componentes generales de ml_lib aplicados al dominio ecológico
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os

# Importar componentes de nuestra biblioteca
from ml_lib.core import LoggingService
from ecoml_analyzer.data import EcologicalDataReader, AbundanceNormalizer, DataFilter
from ecoml_analyzer.analysis import EcologicalAnalyzer, EcologicalModelEstimator
from ecoml_analyzer.visualization import EcologicalVisualizer


def generate_synthetic_ecological_data(
    n_species: int = 50,
    n_sites: int = 20,
    n_habitats: int = 3,
    random_state: int = 42
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    """
    Genera datos sintéticos ecológicos para demostración.
    """
    np.random.seed(random_state)
    
    # Crear nombres de especies y sitios
    species_names = [f"Species_{i:03d}" for i in range(n_species)]
    site_names = [f"Site_{i:02d}" for i in range(n_sites)]
    
    # Generar matriz de abundancia con estructura de hábitat
    abundance_matrix = np.zeros((n_species, n_sites))
    
    # Definir especies preferentes para cada hábitat
    species_per_habitat = n_species // n_habitats
    for habitat in range(n_habitats):
        start_idx = habitat * species_per_habitat
        end_idx = (habitat + 1) * species_per_habitat if habitat < n_habitats - 1 else n_species
        
        # Asignar rangos de especies a cada hábitat
        habitat_species = range(start_idx, end_idx)
        
        # Definir sitios para cada hábitat
        sites_per_habitat = n_sites // n_habitats
        habitat_sites_start = habitat * sites_per_habitat
        habitat_sites_end = (habitat + 1) * sites_per_habitat if habitat < n_habitats - 1 else n_sites
        
        # Añadir abundancia a especies en su hábitat
        for site_idx in range(habitat_sites_start, habitat_sites_end):
            for sp_idx in habitat_species:
                # Abundancia base con variación
                base_abundance = np.random.poisson(5)
                # Añadir efecto de hábitat
                habitat_effect = 10 if np.random.random() > 0.3 else 5  # Mayor abundancia para especies de hábitat
                abundance_matrix[sp_idx, site_idx] = base_abundance + habitat_effect
    
    # Añadir algo de ruido y ocurrencia cero para realismo
    noise = np.random.poisson(1, size=(n_species, n_sites))
    abundance_matrix = np.maximum(abundance_matrix + noise, 0)
    
    # Generar datos ambientales simples
    environmental_data = np.random.normal(0, 1, (n_sites, 5))  # 5 variables ambientales
    
    return abundance_matrix, species_names, site_names, environmental_data


def main():
    """
    Función principal que demuestra componentes generales de ml_lib aplicados al dominio ecológico.
    """
    print("🌿 EcoML Analyzer - Demostración de componentes generales de ml_lib")
    print("=" * 65)
    
    # Configurar logging
    logger_service = LoggingService("EcoML_Demo")
    logger = logger_service.get_logger()
    logger.info("Iniciando demostración de EcoML Analyzer con componentes generales")
    
    # 1. Generar datos sintéticos
    print("\n1. Generando datos ecológicos sintéticos...")
    abundance_matrix, species_names, site_names, environmental_data = generate_synthetic_ecological_data()
    
    print(f"   - Matriz de abundancia: {abundance_matrix.shape}")
    print(f"   - Especies: {len(species_names)}")
    print(f"   - Sitios: {len(site_names)}")
    print(f"   - Datos ambientales: {environmental_data.shape}")
    
    # 2. Preprocesamiento de datos usando componentes generales
    print("\n2. Preprocesando datos con componentes generales...")
    
    # Usar el transformador de normalización que implementa la interfaz general
    normalizer = AbundanceNormalizer(method='sqrt')
    normalized_matrix = normalizer.fit_transform(abundance_matrix)
    print(f"   - Datos normalizados usando transformador general: {normalized_matrix.shape}")
    
    # Usar el filtro general para datos
    filtered_matrix, filtered_species = DataFilter.filter_by_occurrence(
        normalized_matrix, 
        species_names,
        min_occurrence_proportion=0.1,
        min_total_value=1
    )
    print(f"   - Especies después de filtrado: {len(filtered_matrix)}")
    
    # 3. Análisis usando componentes generales aplicados al dominio
    print("\n3. Aplicando técnicas generales de ml_lib al dominio ecológico...")
    analyzer = EcologicalAnalyzer()
    
    composition_results = analyzer.analyze_composition(filtered_matrix.T)  # Transponer para análisis por sitio
    print(f"   - Análisis de composición completado")
    print(f"   - Riqueza media por sitio: {composition_results['richness'].mean():.2f}")
    print(f"   - Diversidad media: {composition_results['diversity'].mean():.2f}")
    
    # 4. Reducción de dimensionalidad con componentes generales
    print("\n4. Aplicando reducción de dimensionalidad general...")
    # Transponer para tener sitios como muestras y especies como características
    dim_red_results = analyzer.perform_dimensionality_reduction(
        filtered_matrix.T,  # sitios x especies
        method='custom_pca',
        n_components=5
    )
    
    print(f"   - Método: {dim_red_results['method']}")
    print(f"   - Componentes principales: {dim_red_results['n_components']}")
    print(f"   - Varianza explicada PC1: {dim_red_results['explained_variance_ratio'][0]:.2%}")
    
    # 5. Clustering con componentes generales
    print("\n5. Aplicando clustering general...")
    clustering_results = analyzer.perform_clustering(
        filtered_matrix.T,  # sitios x especies
        n_clusters=3,
        method='kmeans'
    )
    
    print(f"   - Método de clustering: {clustering_results['clustering_method']}")
    print(f"   - Número de clusters: {clustering_results['n_clusters']}")
    print(f"   - Silhouette Score: {clustering_results['silhouette_score']:.3f}")
    print(f"   - Calinski-Harabasz Score: {clustering_results['calinski_harabasz_score']:.1f}")
    
    # 6. Demostración del estimador general aplicado a datos ecológicos
    print("\n6. Demostrando estimador general aplicado al dominio ecológico...")
    
    # Simular una tarea de clasificación binaria (por ejemplo, presencia/ausencia de una especie)
    # Usar los primeros componentes principales como features
    X_features = dim_red_results['transformed_data'][:, :3]  # Usar primeros 3 componentes
    y_target = (filtered_matrix[0, :] > np.median(filtered_matrix[0, :])).astype(int)  # Clasificación basada en abundancia de primera especie
    
    # Crear y entrenar el estimador ecológico que implementa la interfaz general
    eco_model = EcologicalModelEstimator(model_type='logistic', random_state=42)
    eco_model.fit(X_features, y_target)
    
    # Realizar predicciones
    predictions = eco_model.predict(X_features)
    accuracy = np.mean(predictions == y_target)
    
    print(f"   - Estimador entrenado con {X_features.shape[1]} features")
    print(f"   - Precisión en entrenamiento: {accuracy:.3f}")
    print(f"   - Demostrando interfaz general EstimatorInterface aplicada a ecología")
    
    # 7. Visualización de resultados
    print("\n7. Generando visualizaciones...")
    visualizer = EcologicalVisualizer()
    
    # Heatmap de abundancia
    print("   - Creando heatmap de abundancia...")
    fig1 = visualizer.plot_species_abundance_heatmap(
        filtered_matrix[:15],  # Solo primeras 15 especies para claridad
        filtered_species[:15],
        site_names,
        "Heatmap de Abundancia de Especies (Primeras 15)"
    )
    
    # Análisis de composición
    print("   - Creando plot de análisis de composición...")
    fig2 = visualizer.plot_composition_analysis(
        {
            'richness': composition_results['richness'],
            'diversity': composition_results['diversity'],
            'total_values': np.sum(filtered_matrix, axis=0),
            'abundance_distribution': composition_results['abundance_distribution']
        },
        site_names,
        "Análisis de Composición Ecológica"
    )
    
    # Análisis de dimensionalidad
    print("   - Creando plot de análisis de dimensionalidad...")
    fig3 = visualizer.plot_dimensionality_reduction(
        dim_red_results,
        site_names,
        "Análisis de Dimensionalidad General Aplicado a Ecología"
    )
    
    # Resultados de clustering
    print("   - Creando plot de clustering...")
    fig5 = visualizer.plot_clustering_results(
        clustering_results,
        filtered_matrix.T,  # Pasar matriz en formato correcto
        site_names,
        "Clustering General Aplicado a Datos Ecológicos"
    )
    
    # 8. Guardar visualizaciones
    print("\n8. Guardando visualizaciones...")
    os.makedirs("ecoml_output", exist_ok=True)
    visualizer.save_all_figures("ecoml_output")
    print("   - Visualizaciones guardadas en ecoml_output/")
    
    # 9. Resumen de resultados
    print("\n📊 Resumen de la demostración:")
    print(f"   - Especies analizadas: {len(filtered_species)}")
    print(f"   - Sitios analizados: {len(site_names)}")
    print(f"   - Aplicación de componentes generales de ml_lib al dominio ecológico")
    print(f"   - Uso de interfaces generales: TransformerInterface, EstimatorInterface")
    print(f"   - Técnicas aplicadas: normalización, filtrado, PCA, clustering, clasificación")
    print(f"   - Precisión del modelo ecológico: {accuracy:.3f}")
    
    print("\n✅ Demostración completada exitosamente!")
    print("   La aplicación muestra cómo técnicas generales de ml_lib se aplican al dominio ecológico")
    print("   Las visualizaciones están disponibles en el directorio 'ecoml_output/'")
    
    # Mostrar que se usaron componentes generales
    print(f"\n🔍 Componentes generales de ml_lib demostrados:")
    print(f"   - Interfaz TransformerInterface para transformadores de datos")
    print(f"   - Interfaz EstimatorInterface para modelos predictivos")
    print(f"   - Servicios de validación y logging")
    print(f"   - Operaciones de álgebra lineal general (SVD para PCA)")
    print(f"   - Aplicación agnóstica al dominio con enfoque ecológico")
    
    logger.info("Demostración completada exitosamente")
    
    return {
        'abundance_matrix': filtered_matrix,
        'species_names': filtered_species,
        'site_names': site_names,
        'environmental_data': environmental_data,
        'composition_results': composition_results,
        'dim_red_results': dim_red_results,
        'clustering_results': clustering_results,
        'model_results': {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_values': y_target
        }
    }


def run_simple_example():
    """
    Ejecuta un ejemplo simple para probar rápidamente la funcionalidad.
    """
    print("🌿 Ejemplo simple de EcoML Analyzer")
    print("=" * 40)
    
    # Generar datos pequeños
    n_species, n_sites = 10, 8
    abundance_matrix = np.random.poisson(3, size=(n_species, n_sites))
    species_names = [f"SP_{i}" for i in range(n_species)]
    site_names = [f"SITE_{i}" for i in range(n_sites)]
    
    # Usar componentes generales aplicados al dominio
    normalizer = AbundanceNormalizer(method='sqrt')
    normalized_matrix = normalizer.fit_transform(abundance_matrix)
    
    analyzer = EcologicalAnalyzer()
    composition_results = analyzer.analyze_composition(normalized_matrix.T)
    
    print(f"Especies analizadas: {n_species}")
    print(f"Sitios analizados: {n_sites}")
    print(f"Riqueza media: {composition_results['richness'].mean():.2f}")
    print(f"Diversidad media: {composition_results['diversity'].mean():.2f}")
    print("¡Ejemplo simple completado!")
    print("El ejemplo demuestra componentes generales de ml_lib aplicados a datos ecológicos")


if __name__ == "__main__":
    # Ejecutar ejemplo simple primero
    run_simple_example()
    print("\n" + "="*60 + "\n")
    
    # Ejecutar demostración completa
    results = main()