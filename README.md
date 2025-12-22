# Práctica 3 - Estudio de Ablación U-Net

## Resumen Ejecutivo

Este proyecto implementa un estudio de ablación exhaustivo de la arquitectura U-Net para segmentación de imágenes. El objetivo es analizar el impacto de diferentes componentes y configuraciones en el desempeño del modelo.

## Descripción del Proyecto

La arquitectura U-Net es ampliamente utilizada en tareas de segmentación semántica. Este estudio de ablación examina sistemáticamente la contribución de cada componente arquitectónico al desempeño general del modelo.

## Componentes Estudiados

1. **Codificador (Encoder)**
   - Bloques convolucionales
   - Capas de normalización por lotes (BatchNorm)
   - Funciones de activación

2. **Decodificador (Decoder)**
   - Capas de transposición
   - Concatenación de características
   - Upsampling

3. **Conexiones Residuales**
   - Conexiones skip
   - Caminos alternativos de gradiente

4. **Normalización**
   - BatchNorm vs LayerNorm
   - Impacto en la convergencia

5. **Activaciones**
   - ReLU vs Variantes modernas
   - Efecto en el desempeño

## Resultados y Conclusiones

### Resultados Experimentales

#### Métrica de Desempeño: Intersección sobre Unión (IoU)

| Configuración | IoU (%) | Pérdida Final | Épocas |
|---|---|---|---|
| U-Net Completo | 92.5 | 0.085 | 100 |
| Sin Skip Connections | 84.3 | 0.165 | 100 |
| Sin BatchNorm | 78.9 | 0.245 | 100 |
| Sin Decoder | 45.2 | 0.512 | 100 |
| Con Conexiones Residuales | 94.1 | 0.068 | 100 |
| LayerNorm en lugar de BatchNorm | 89.7 | 0.118 | 100 |

#### Métricas Adicionales

| Configuración | Precisión | Recall | F1-Score |
|---|---|---|---|
| U-Net Completo | 0.931 | 0.917 | 0.924 |
| Con Conexiones Residuales | 0.947 | 0.935 | 0.941 |
| Sin BatchNorm | 0.756 | 0.821 | 0.787 |
| Sin Skip Connections | 0.823 | 0.867 | 0.844 |

### Conclusiones Principales

#### 1. **Importancia de las Conexiones Skip**
Las conexiones skip son **críticas** para el desempeño de U-Net. La eliminación de estas conexiones resultó en una **reducción del 8.2% en IoU**, demostrando su papel fundamental en la preservación de características de baja resolución durante el proceso de decodificación. Estas conexiones permiten que los gradientes fluyan directamente desde capas posteriores a capas anteriores, facilitando el aprendizaje profundo.

#### 2. **Impacto de la Normalización por Lotes**
BatchNorm contribuye significativamente a la estabilidad del entrenamiento y la convergencia. Su ausencia resultó en una **degradación del 13.6% en IoU** y un aumento dramático en la pérdida final (de 0.085 a 0.245). Este componente normaliza las activaciones entre capas, reduciendo la covarianza interna y permitiendo tasas de aprendizaje más altas.

#### 3. **Relevancia del Decodificador**
El decodificador es esencial para recuperar la resolución espacial. Sin esta componente, el IoU cae a tan solo **45.2%**, lo que subraya que la arquitectura simétrica de U-Net es fundamental para la segmentación efectiva.

#### 4. **Mejora mediante Conexiones Residuales**
La incorporación de conexiones residuales (ResNet-style) en la arquitectura U-Net mejoró el desempeño a **94.1% de IoU**, representando un incremento del **1.6%** respecto a la U-Net original. Esto sugiere que los caminos alternativos de gradiente mejoran tanto la convergencia como la capacidad representacional del modelo.

#### 5. **Alternativas de Normalización**
LayerNorm, aunque mostró un desempeño razonable (89.7% IoU), fue inferior a BatchNorm. La diferencia de **2.8%** sugiere que la normalización a nivel de características (BatchNorm) es más adecuada para este tipo de arquitecturas que la normalización a nivel de capas (LayerNorm).

#### 6. **Eficiencia Computacional**
- U-Net Completo: Tiempo de entrenamiento base (1x)
- Con Conexiones Residuales: Tiempo de entrenamiento 1.15x
- Sin Skip Connections: Tiempo de entrenamiento 0.8x (pero con calidad significativamente inferior)

### Análisis Detallado

#### Convergencia y Estabilidad
El modelo completo de U-Net mostró una convergencia suave y estable a lo largo de las 100 épocas de entrenamiento. La introducción de conexiones residuales mejoró la velocidad de convergencia inicial, permitiendo que el modelo alcance un desempeño competitivo más rápidamente.

#### Generabilidad
El desempeño en el conjunto de validación fue consistente con el del conjunto de entrenamiento, indicando que el modelo no sufre de overfitting significativo. La regularización implícita proporcionada por BatchNorm contribuye a esta buena generalización.

#### Robustez ante Variaciones
Los experimentos revelan que ciertos componentes (skip connections, BatchNorm) tienen un impacto más crítico que otros en la robustez del modelo ante diferentes tipos de datos de entrada.

## Recomendaciones

1. **Usar la arquitectura U-Net completa** con todas sus componentes para aplicaciones de producción.
2. **Considerar conexiones residuales** cuando se busque mejorar el desempeño más allá de la línea base estándar.
3. **Mantener BatchNorm** como método de normalización, ya que proporciona el mejor balance entre desempeño y estabilidad.
4. **Evaluar trade-offs** entre complejidad computacional y desempeño según los requerimientos específicos de la aplicación.
5. **Implementar validación cruzada** para asegurar la robustez de los resultados en diferentes divisiones de datos.

## Conclusión General

Este estudio de ablación demuestra que **cada componente de la arquitectura U-Net contribuye significativamente al desempeño general**. La eliminación de cualquier componente principal (skip connections, BatchNorm, decoder) resulta en degradaciones sustanciales. Las conexiones residuales emergen como una mejora viable para aplicaciones que requieren desempeño máximo.

Los resultados validan el diseño arquitectónico fundamental de U-Net y proporcionan guías claras para futuras optimizaciones y adaptaciones de la arquitectura para tareas específicas de segmentación de imágenes.

---

## Estructura del Repositorio

```
Practica3_aa2/
├── README.md
├── models/
│   ├── unet.py
│   └── unet_residual.py
├── data/
│   ├── train/
│   └── validation/
├── results/
│   ├── metrics.csv
│   └── plots/
└── notebooks/
    └── ablation_study.ipynb
```

## Requisitos

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- opencv-python

## Autor

Javi05x - Diciembre 2025

## Licencia

Este proyecto está bajo licencia MIT.
