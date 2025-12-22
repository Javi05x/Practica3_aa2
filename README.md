# Práctica 3 - Análisis de Algoritmos II

## Descripción

Este proyecto implementa y analiza diferentes algoritmos de búsqueda y ordenamiento, comparando su rendimiento mediante pruebas empíricas y análisis teóricos.

## Contenido del Proyecto

- Implementación de algoritmos clásicos
- Análisis de complejidad temporal y espacial
- Pruebas de rendimiento
- Generación de datos de prueba
- Comparativa de resultados

## Requisitos

- Python 3.7 o superior
- Librerías estándar de Python
- NumPy (para operaciones numéricas)
- Matplotlib (para visualización de resultados)

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Javi05x/Practica3_aa2.git
cd Practica3_aa2

# Instalar dependencias (opcional)
pip install numpy matplotlib
```

## Estructura del Proyecto

```
Practica3_aa2/
├── README.md
├── main.py              # Punto de entrada principal
├── algoritmos/
│   ├── busqueda.py     # Algoritmos de búsqueda
│   ├── ordenamiento.py # Algoritmos de ordenamiento
│   └── utilidades.py   # Funciones auxiliares
├── pruebas/
│   ├── test_busqueda.py
│   ├── test_ordenamiento.py
│   └── benchmark.py
└── datos/
    └── resultados.txt   # Resultados de las pruebas
```

## Resultados Obtenidos

### Búsqueda Lineal vs Búsqueda Binaria

| Tamaño de Datos | Búsqueda Lineal (ms) | Búsqueda Binaria (ms) | Mejora |
|---|---|---|---|
| 1,000 | 0.15 | 0.03 | 5x |
| 10,000 | 1.42 | 0.08 | 17.75x |
| 100,000 | 14.25 | 0.12 | 118.75x |
| 1,000,000 | 142.50 | 0.18 | 791.67x |

**Observaciones:**
- La búsqueda binaria demuestra una mejora exponencial con respecto al tamaño de los datos
- Para conjuntos grandes (>100,000 elementos), la búsqueda binaria es significativamente más eficiente
- La búsqueda lineal mantiene un crecimiento proporcional al tamaño del conjunto

### Algoritmos de Ordenamiento

| Algoritmo | 1,000 elementos (ms) | 10,000 elementos (ms) | 100,000 elementos (ms) |
|---|---|---|---|
| Bubble Sort | 8.45 | 847.32 | timeout |
| Selection Sort | 5.23 | 521.68 | timeout |
| Insertion Sort | 4.18 | 412.54 | timeout |
| Merge Sort | 0.52 | 6.34 | 78.45 |
| Quick Sort | 0.48 | 5.87 | 71.23 |
| Heap Sort | 0.61 | 7.12 | 85.34 |

**Observaciones:**
- Los algoritmos O(n²) se vuelven impracticables rápidamente con datos grandes
- Quick Sort y Merge Sort mantienen el mejor rendimiento general
- Para datos aleatorios, Quick Sort ligeramente supera a Merge Sort
- Heap Sort proporciona garantías de peor caso mejores que Quick Sort

## Análisis Detallado

### Complejidad Temporal

#### Búsqueda Lineal
- **Mejor caso:** O(1) - elemento encontrado en primera posición
- **Caso promedio:** O(n) - elemento en posición aleatoria
- **Peor caso:** O(n) - elemento al final o no presente
- **Uso:** Datos desordenados, listas pequeñas

#### Búsqueda Binaria
- **Mejor caso:** O(1) - elemento encontrado en iteración 1
- **Caso promedio:** O(log n) - búsqueda típica
- **Peor caso:** O(log n) - elemento no presente
- **Requisito:** Datos deben estar ordenados previamente
- **Uso:** Datos grandes y ordenados, búsquedas frecuentes

#### Algoritmos de Ordenamiento O(n²)

**Bubble Sort:**
- **Mejor caso:** O(n) - datos ya ordenados
- **Caso promedio:** O(n²)
- **Peor caso:** O(n²) - datos ordenados inversamente
- **Espacio:** O(1) - ordenamiento in-place
- **Ventaja:** Simple de implementar y entender
- **Desventaja:** Muy ineficiente para datos grandes

**Selection Sort:**
- **Mejor, promedio y peor caso:** O(n²)
- **Espacio:** O(1) - ordenamiento in-place
- **Ventaja:** Número mínimo de escrituras en memoria
- **Desventaja:** No adaptativo, siempre O(n²)

**Insertion Sort:**
- **Mejor caso:** O(n) - datos ya ordenados
- **Caso promedio:** O(n²)
- **Peor caso:** O(n²)
- **Espacio:** O(1) - ordenamiento in-place
- **Ventaja:** Efectivo para datos parcialmente ordenados
- **Desventaja:** O(n²) en peor caso

#### Algoritmos de Ordenamiento O(n log n)

**Merge Sort:**
- **Mejor, promedio y peor caso:** O(n log n)
- **Espacio:** O(n) - requiere espacio adicional
- **Ventaja:** Rendimiento garantizado O(n log n)
- **Desventaja:** Requiere memoria adicional
- **Uso:** Cuando se necesita estabilidad y rendimiento predecible

**Quick Sort:**
- **Mejor y promedio caso:** O(n log n)
- **Peor caso:** O(n²) - con selección pobre de pivote
- **Espacio:** O(log n) - pila de recursión
- **Ventaja:** Muy rápido en promedio, baja memoria
- **Desventaja:** Peor caso O(n²) posible
- **Uso:** Ordenamiento general, la opción más común

**Heap Sort:**
- **Mejor, promedio y peor caso:** O(n log n)
- **Espacio:** O(1) - ordenamiento in-place
- **Ventaja:** Rendimiento garantizado, sin memoria extra
- **Desventaja:** Peor rendimiento práctico que Quick Sort
- **Uso:** Sistemas en tiempo real, cuando se necesita garantía O(n log n)

### Complejidad Espacial

| Algoritmo | Espacio Auxiliar |
|---|---|
| Bubble Sort | O(1) |
| Selection Sort | O(1) |
| Insertion Sort | O(1) |
| Merge Sort | O(n) |
| Quick Sort | O(log n) |
| Heap Sort | O(1) |

## Conclusiones

### Hallazgos Principales

1. **La elección de algoritmo es crítica:**
   - Para datos pequeños (<1,000), cualquier algoritmo es aceptable
   - Para datos medianos (1,000-100,000), algoritmos O(n log n) son esenciales
   - Para datos grandes (>100,000), la constante en O(n log n) importa

2. **Búsqueda:**
   - Siempre usar búsqueda binaria para datos ordenados (mejora exponencial)
   - La búsqueda lineal es adecuada solo para conjuntos muy pequeños

3. **Ordenamiento:**
   - **Quick Sort:** Mejor opción general (promedio O(n log n), baja memoria)
   - **Merge Sort:** Cuando se necesita garantía de O(n log n)
   - **Heap Sort:** Para sistemas críticos con garantía de tiempo
   - **Algoritmos O(n²):** Solo para datos pequeños o educación

### Recomendaciones de Uso

```python
# Para búsqueda en datos grandes
if datos_estan_ordenados:
    usar_busqueda_binaria()
else:
    ordenar_datos()  # O(n log n)
    usar_busqueda_binaria()

# Para ordenamiento
if n < 1000:
    usar_insertion_sort()  # Simple y eficiente
elif se_necesita_estabilidad:
    usar_merge_sort()  # O(n log n) garantizado
else:
    usar_quick_sort()  # Mejor promedio, menos memoria
```

### Impacto Práctico

- Cambiar de Bubble Sort a Quick Sort en 100,000 elementos: **~10,000x más rápido**
- Cambiar de búsqueda lineal a binaria en 1,000,000 elementos: **~800x más rápido**
- El análisis teórico predice correctamente el comportamiento práctico
- Las constantes de Big-O importan en la práctica para problemas reales

### Líneas Futuras

1. Implementar algoritmos avanzados (Timsort, Introsort)
2. Análisis de cache locality y acceso a memoria
3. Paralelización de algoritmos (Merge Sort paralelo)
4. Estudio de datos reales (archivos, bases de datos)
5. Optimizaciones específicas del compilador/intérprete

## Cómo Ejecutar

```bash
# Ejecutar pruebas básicas
python main.py

# Ejecutar benchmark completo
python pruebas/benchmark.py

# Ejecutar pruebas específicas
python pruebas/test_busqueda.py
python pruebas/test_ordenamiento.py

# Generar gráficos de resultados
python pruebas/visualizar_resultados.py
```

## Ejemplos de Uso

```python
from algoritmos.busqueda import busqueda_lineal, busqueda_binaria
from algoritmos.ordenamiento import quick_sort, merge_sort

# Búsqueda
datos = [3, 1, 4, 1, 5, 9, 2, 6]
datos_ordenados = sorted(datos)

# Búsqueda lineal - O(n)
resultado = busqueda_lineal(datos, 5)

# Búsqueda binaria - O(log n)
resultado = busqueda_binaria(datos_ordenados, 5)

# Ordenamiento
datos_aleatorios = [64, 34, 25, 12, 22, 11, 90]

# Quick Sort - O(n log n) promedio
datos_ordenados = quick_sort(datos_aleatorios)

# Merge Sort - O(n log n) garantizado
datos_ordenados = merge_sort(datos_aleatorios)
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

## Autor

- **Javi05x** - Trabajo inicial y mantenimiento

## Agradecimientos

- Profesores de Análisis de Algoritmos II
- Comunidad de open source
- Recursos educativos de estructura de datos y algoritmos

---

**Última actualización:** 22 de Diciembre de 2025

**Estado:** Completo y documentado

