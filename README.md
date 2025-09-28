# Dataset Sintético: Predicción de Abandono Estudiantil

Este dataset fue generado de manera sintética con el fin de simular un escenario universitario 
en el que se busca predecir si un estudiante abandonará sus estudios durante el primer año académico. 
El dataset contiene **500 registros** y está diseñado para ser utilizado en actividades de aprendizaje 
sobre Machine Learning (aprendizaje supervisado).

#  Descripción de las Variables

- **ID**: Identificador único del estudiante (numérico).
- **Edad**: Edad del estudiante (16 a 29 años).
- **Género**: M (Masculino), F (Femenino).
- **Ciudad**: Ciudad de origen del estudiante (Bogotá, Barranquilla, Cali, Medellín, Cartagena).
- **Promedio_Secundaria**: Promedio de calificaciones en la secundaria (2.5 a 5.0).
- **Nota_Admision**: Resultado del examen de admisión (150 a 400, con algunos valores atípicos en 999).
- **Promedio_1er_Semestre**: Promedio de notas del primer semestre (2.0 a 5.0, con algunos valores atípicos en -1).
- **Nivel_Socioeconómico**: Categoría (Bajo, Medio, Alto).
- **Beca**: Si el estudiante recibe beca (Sí/No).
- **Préstamo**: Si el estudiante tiene préstamo educativo (Sí/No).
- **Abandono**: Variable objetivo (Sí = abandonó, No = continuó).

#  Nulos y Outliers Introducidos

- Se introdujeron valores **nulos (NaN)** en algunas columnas: `Edad`, `Promedio_Secundaria`, `Nota_Admision`, `Promedio_1er_Semestre` (~5% de los datos).  
- Se introdujeron **valores atípicos**:
  - `Nota_Admision` con valor 999 en algunos registros (fuera del rango esperado).  
  - `Promedio_1er_Semestre` con valor -1 en algunos registros (inconsistente con el rango esperado).  

Esto permite practicar **limpieza de datos** y **preprocesamiento**.

#  Objetivo del Dataset

El objetivo es utilizar este dataset para aplicar **modelos de Machine Learning supervisados**, 
especialmente de **clasificación binaria**, con el fin de predecir si un estudiante **abandonará (Sí/No)** 
sus estudios durante el primer año.  

Ejemplos de algoritmos aplicables:  
- Regresión Logística  
- Árboles de Decisión  
- Random Forest  
- Redes Neuronales  

#  Instrucciones de Uso

1. Descargar el archivo `dataset_abandono_estudiantil.xlsx`.  
2. Cargarlo en Python (pandas), R o cualquier software estadístico.  
3. Realizar limpieza de datos (manejo de nulos, outliers, codificación de variables categóricas).  
4. Entrenar y evaluar un modelo supervisado de clasificación.  

---

 **Generado con fines académicos para la Universidad de la Costa - Curso de Data Mining.**


