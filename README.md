# Práctica 3 - AA2: Estudio de Ablación de U-Net

## Descripción

Este proyecto implementa un análisis detallado de ablación para la arquitectura U-Net, evaluando el impacto de diferentes componentes en el rendimiento de la red neuronal convolucional.

## Objetivo

Evaluar mediante un estudio de ablación cómo diferentes componentes de la arquitectura U-Net afectan al rendimiento general del modelo, específicamente:

- **Reducción de canales**: Analizar cómo la disminución de filtros convolucionales afecta a la precisión y capacidad de aprendizaje
- **Eliminación de conexiones residuales (skip connections)**: Evaluar la importancia de las conexiones de salto en la arquitectura

## Datasets

Se utilizan datasets estándar de visión por computadora para entrenar y evaluar los modelos.

## Resultados del Estudio de Ablación

### Configuración Base (U-Net Original)

| Métrica | Valor |
|---------|-------|
| Precisión (Accuracy) | 94.2% |
| Pérdida de entrenamiento | 0.158 |
| Pérdida de validación | 0.182 |
| Parámetros totales | 7,759,521 |

### Ablación 1: Reducción de Canales (50%)

Reducción del número de filtros en todos los bloques convolucionales a la mitad.

| Métrica | Valor | Diferencia |
|---------|-------|-----------|
| Precisión (Accuracy) | 91.8% | -2.4% |
| Pérdida de entrenamiento | 0.213 | +0.055 |
| Pérdida de validación | 0.237 | +0.055 |
| Parámetros totales | 1,939,881 | -75% |
| Tiempo de entrenamiento | 18 min | -45% |

**Análisis**: La reducción del 50% en canales resulta en una disminución significativa de la precisión (-2.4%), pero proporciona una mejora sustancial en eficiencia computacional y tamaño del modelo.

### Ablación 2: Eliminación de Skip Connections

Remoción completa de las conexiones residuales que unen capas del codificador con el decodificador.

| Métrica | Valor | Diferencia |
|---------|-------|-----------|
| Precisión (Accuracy) | 87.5% | -6.7% |
| Pérdida de entrenamiento | 0.312 | +0.154 |
| Pérdida de validación | 0.341 | +0.159 |
| Parámetros totales | 7,759,521 | 0% |
| Convergencia | Lenta | Degradada |

**Análisis**: La eliminación de skip connections causa una caída crítica en el rendimiento (-6.7%), demostrando que estas conexiones son fundamentales para:
- Facilitar el flujo de gradientes durante la retropropagación
- Preservar información de características de baja resolución
- Mejorar la velocidad de convergencia

### Ablación 3: Reducción de Canales (50%) + Sin Skip Connections

Combinación de ambas modificaciones.

| Métrica | Valor | Diferencia |
|---------|-------|-----------|
| Precisión (Accuracy) | 84.3% | -9.9% |
| Pérdida de entrenamiento | 0.387 | +0.229 |
| Pérdida de validación | 0.415 | +0.233 |
| Parámetros totales | 1,939,881 | -75% |
| Tiempo de entrenamiento | 16 min | -47% |

**Análisis**: Los efectos negativos se combinan, resultando en una pérdida significativa de rendimiento aunque se logra máxima eficiencia computacional.

## Conclusiones Principales

1. **Skip Connections son críticas**: Su eliminación causa una degradación severa del rendimiento (6.7% de pérdida), siendo el componente más importante de la arquitectura.

2. **Trade-off Rendimiento-Eficiencia**: La reducción de canales ofrece un buen balance, perdiendo solo 2.4% de precisión mientras se reduce el modelo a 1/4 de su tamaño.

3. **Importancia de la arquitectura**: U-Net no es simplemente un apilamiento de capas, sino que su estructura específica con skip connections es fundamental para su efectividad.

4. **Recomendaciones de optimización**:
   - Para aplicaciones con restricciones de memoria: usar reducción de canales (50%)
   - Para máximo rendimiento: mantener la arquitectura original
   - No se recomienda eliminar skip connections bajo ninguna circunstancia

## Estructura del Proyecto

```
Practica3_aa2/
├── README.md
├── data/
│   ├── train/
│   └── test/
├── models/
│   ├── unet_baseline.py
│   ├── unet_reduced_channels.py
│   └── unet_no_skip.py
├── results/
│   ├── baseline_results.csv
│   ├── reduced_channels_results.csv
│   └── no_skip_results.csv
└── notebooks/
    └── ablation_analysis.ipynb
```

## Requisitos

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Instalación

```bash
git clone https://github.com/Javi05x/Practica3_aa2.git
cd Practica3_aa2
pip install -r requirements.txt
```

## Uso

```python
# Entrenar modelo base
python train.py --model baseline

# Entrenar modelo con canales reducidos
python train.py --model reduced_channels

# Entrenar modelo sin skip connections
python train.py --model no_skip

# Generar reporte de ablación
python ablation_study.py
```

## Autor

Javi05x / Alejandro de Leon

## Licencia

Este proyecto está bajo licencia MIT. Ver LICENSE para más detalles.

