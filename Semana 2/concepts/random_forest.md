# Random Forest: guía práctica y conceptual

## 1. ¿Qué es Random Forest?

Random Forest, o **bosque aleatorio**, es un algoritmo de aprendizaje supervisado que combina muchos árboles de decisión para producir una predicción más robusta que la de un solo árbol. Se usa tanto en:

- **Clasificación**: predecir categorías, por ejemplo fraude/no fraude.
- **Regresión**: predecir valores numéricos, por ejemplo el precio de una vivienda.

La idea central es sencilla: en lugar de confiar en un único árbol, construimos muchos árboles diferentes y combinamos sus respuestas.

## 2. Intuición: por qué funciona bien

Un árbol individual puede capturar relaciones no lineales y reglas complejas, pero también puede ser inestable: pequeños cambios en los datos pueden producir un árbol muy distinto. Random Forest reduce esa inestabilidad combinando muchos árboles entrenados sobre versiones distintas del conjunto de datos y obligándolos a explorar subconjuntos aleatorios de variables.

Esto suele producir un modelo con:

- buen poder predictivo,
- menor varianza que un árbol individual,
- menor riesgo de sobreajuste que un árbol profundo usado por sí solo.

## 3. Relación con bagging

Random Forest es una extensión de **bagging** (*bootstrap aggregating*).

En bagging:

1. Se toman muchas muestras *bootstrap* del conjunto de entrenamiento.
2. Se entrena un árbol sobre cada muestra.
3. Se combinan las predicciones de todos los árboles.

Random Forest agrega un elemento adicional: **en cada división del árbol, no se consideran todas las variables, sino solo un subconjunto aleatorio de ellas**.

Ese detalle hace que los árboles sean menos parecidos entre sí. Cuando los árboles están menos correlacionados, el promedio o la votación final suele mejorar.

## 4. ¿Qué es una muestra bootstrap?

Una muestra *bootstrap* es una muestra aleatoria tomada **con reemplazo** y del mismo tamaño que el conjunto original.

Si tenemos 100 observaciones, una muestra bootstrap también tendrá 100 observaciones, pero:

- algunas aparecerán varias veces,
- otras no aparecerán en esa muestra.

Las observaciones que no entran en una muestra bootstrap se llaman **out-of-bag (OOB)** para ese árbol. Más adelante veremos por qué esto es útil.

## 5. Cómo se construye un Random Forest

El proceso general es:

1. Tomar una muestra bootstrap del conjunto de entrenamiento.
2. Entrenar un árbol de decisión sobre esa muestra.
3. En cada nodo del árbol, seleccionar aleatoriamente `m` variables de las `p` variables disponibles.
4. Buscar la mejor división solo entre esas `m` variables.
5. Repetir este proceso para construir muchos árboles.
6. Combinar las predicciones de todos los árboles.

La combinación final depende del tipo de problema:

- **Clasificación**: votación mayoritaria.
- **Regresión**: promedio de predicciones.

## 6. El papel de `m`: cuántas variables se evalúan en cada división

Uno de los parámetros más importantes es `m`, es decir, la cantidad de variables candidatas en cada bifurcación.

Si tenemos `p` variables predictoras:

- en **clasificación**, una regla común es usar `m = sqrt(p)`,
- en **regresión**, suele probarse entre `p/3` y `p`.

Estas reglas son solo puntos de partida. En la práctica, el mejor valor depende del problema y conviene calibrarlo.

### ¿Por qué no usar siempre todas las variables?

Si cada árbol pudiera revisar siempre todas las variables, muchos árboles terminarían pareciéndose demasiado, especialmente cuando algunas variables son muy dominantes. Al limitar las variables candidatas en cada nodo:

- aumentamos la diversidad entre árboles,
- reducimos la correlación entre ellos,
- mejoramos el efecto del ensamblado.

## 7. Parámetros importantes del modelo

Además de `m`, en la práctica hay varios hiperparámetros relevantes:

| Parámetro | Qué controla | Efecto típico |
| --- | --- | --- |
| `n_estimators` | Número de árboles | Más árboles suelen estabilizar el desempeño, pero aumentan tiempo y memoria |
| `max_features` | Número de variables candidatas por división | Controla la diversidad entre árboles |
| `max_depth` | Profundidad máxima de cada árbol | Limita complejidad y puede reducir sobreajuste |
| `min_samples_split` | Mínimo de muestras para dividir un nodo | Evita divisiones demasiado específicas |
| `min_samples_leaf` | Mínimo de muestras por hoja | Suaviza predicciones y mejora generalización |
| `bootstrap` | Si se usan muestras bootstrap | Normalmente sí en Random Forest clásico |

Aunque en muchas explicaciones introductorias se enfatizan `n_estimators` y `max_features`, los parámetros de profundidad y tamaño mínimo de nodos también pueden influir bastante en el resultado final.

## 8. Error out-of-bag (OOB)

Una ventaja muy útil de Random Forest es la posibilidad de estimar desempeño sin separar explícitamente un conjunto de prueba para cada ajuste interno.

Como cada árbol se entrena con una muestra bootstrap, siempre quedan observaciones fuera de esa muestra. Entonces:

1. una observación se predice usando únicamente los árboles en los que quedó fuera,
2. se compara esa predicción con su valor real,
3. al repetir esto para todas las observaciones, obtenemos una estimación del **error OOB**.

El error OOB es especialmente útil para:

- obtener una señal rápida del desempeño,
- comparar configuraciones del modelo,
- evitar depender únicamente de una sola partición entrenamiento/prueba.

No reemplaza por completo una evaluación final bien diseñada, pero sí es una herramienta muy valiosa durante el modelado.

## 9. Ventajas de Random Forest

- **Buen desempeño predictivo** en muchos problemas tabulares.
- **Captura relaciones no lineales** e interacciones entre variables.
- **Requiere menos preprocesamiento** que otros modelos: no necesita normalización estricta.
- **Es relativamente robusto** frente a ruido y valores atípicos moderados.
- **Permite medir importancia de variables**.
- **Funciona para clasificación y regresión**.
- **Escala razonablemente bien** cuando se paraleliza el entrenamiento.

## 10. Desventajas y limitaciones

- **Menor interpretabilidad** que un árbol individual.
- Puede ser **más costoso en tiempo y memoria**.
- La importancia de variables basada en impureza puede estar sesgada hacia variables con muchas categorías o alta variabilidad.
- En problemas con muchísimas variables irrelevantes o con datos muy dispersos, puede requerir calibración cuidadosa.
- En tareas donde la interpretabilidad es prioritaria, puede ser preferible un modelo más simple.

## 11. Importancia de variables

Random Forest suele usarse también para identificar qué variables parecen aportar más al modelo. Hay dos enfoques comunes:

### 11.1. Importancia por reducción de impureza

Mide cuánto reduce cada variable la impureza (por ejemplo Gini o varianza) a lo largo de los árboles.

**Ventaja:** rápida de calcular.  
**Desventaja:** puede sesgarse a favor de ciertas variables.

### 11.2. Importancia por permutación

Consiste en mezclar aleatoriamente los valores de una variable y medir cuánto empeora el desempeño del modelo.

**Ventaja:** suele ser más confiable para interpretación.  
**Desventaja:** es más costosa computacionalmente.

Cuando el objetivo sea entender el papel de las variables, la importancia por permutación suele ser una mejor referencia.

## 12. ¿Por qué más árboles suelen ayudar?

Agregar árboles reduce la varianza del ensamblado y hace la predicción más estable. Normalmente, al aumentar `n_estimators`:

- el error baja al principio,
- luego se estabiliza,
- después de cierto punto, añadir más árboles ya no mejora casi nada.

Por eso, el objetivo no es usar "la mayor cantidad posible" sin límite, sino encontrar un punto donde el desempeño ya se estabiliza y el costo computacional siga siendo razonable.

## 13. Random Forest en clasificación y regresión

### Clasificación

Cada árbol vota por una clase y el bosque elige la clase con más votos. También puede estimarse la probabilidad de una clase como la proporción de árboles que votan por ella.

### Regresión

Cada árbol produce un valor numérico y la predicción final es el promedio de todos esos valores. Esto suaviza predicciones extremas de árboles individuales.

## 14. Ejemplo conceptual

Supongamos que queremos predecir si una persona tiene riesgo de enfermedad renal a partir de:

- edad,
- peso,
- latidos por minuto,
- presión arterial.

Un bosque aleatorio podría construir muchos árboles distintos. En uno, la primera división podría usar peso; en otro, presión arterial; en otro, edad. Además, cada árbol se entrenaría sobre una muestra bootstrap diferente.

Cuando llega un nuevo paciente:

- cada árbol emite su predicción,
- el modelo agrega todas esas decisiones,
- la clase final se obtiene por mayoría.

La fortaleza del método no está en que cada árbol sea perfecto, sino en que **muchos árboles diversos, al combinarse, producen una decisión más confiable**.

## 15. Buenas prácticas al usar Random Forest

1. Empezar con los valores por defecto como línea base.
2. Ajustar `n_estimators` hasta observar estabilidad en el desempeño.
3. Calibrar `max_features` según el tipo de problema.
4. Revisar también profundidad y tamaño mínimo de hojas si el modelo es muy complejo.
5. Evaluar con validación cruzada o con error OOB durante el ajuste.
6. Analizar importancia de variables con cautela, idealmente usando permutación.
7. Comparar contra modelos base más simples para justificar su uso.

## 16. Cuándo suele funcionar especialmente bien

Random Forest suele ser una muy buena opción cuando:

- los datos son tabulares,
- hay relaciones no lineales,
- existen interacciones entre variables,
- queremos un modelo fuerte sin una ingeniería de variables demasiado compleja.

En cambio, en datos de texto crudo, series de tiempo muy estructuradas o imágenes, otros enfoques pueden ser más apropiados dependiendo del problema.

## 17. Diferencia frente a otros métodos relacionados

### Árbol de decisión vs. Random Forest

- **Árbol individual**: más interpretable, pero más inestable.
- **Random Forest**: menos interpretable, pero generalmente más preciso y robusto.

### Bagging vs. Random Forest

- **Bagging**: usa muestras bootstrap.
- **Random Forest**: usa bootstrap y, además, selección aleatoria de variables en cada división.

### Random Forest vs. Boosting

- **Random Forest** construye árboles de manera independiente y luego los agrega.
- **Boosting** construye árboles secuencialmente, corrigiendo errores previos.

En general, boosting puede lograr resultados aún mejores en algunos casos, pero Random Forest suele ser más estable y más fácil de calibrar.

## 18. Idea clave para recordar

Random Forest funciona bien porque combina dos fuentes de aleatoriedad:

1. diferentes muestras de entrenamiento mediante *bootstrap*,
2. diferentes subconjuntos de variables en cada división.

Esa combinación genera árboles diversos y reduce la dependencia de cualquier árbol individual. El resultado es un modelo potente, flexible y muy útil como punto de partida para problemas predictivos con datos tabulares.
