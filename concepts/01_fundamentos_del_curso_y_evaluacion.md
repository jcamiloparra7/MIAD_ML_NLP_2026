# Semana 1: fundamentos del curso y evaluacion de modelos

Este documento resume y explica los conceptos introductorios que aparecen en las presentaciones de la semana 1. La idea no es solo listar temas, sino dejar claro **por que importan**, **cuando se usan** y **como se conectan con los laboratorios de la semana**.

## 1. De que trata el curso

El curso combina dos grandes areas:

- **Machine Learning (ML):** construccion de modelos que aprenden patrones a partir de datos.
- **Procesamiento de Lenguaje Natural (PLN):** uso de tecnicas de ML para analizar, interpretar y generar lenguaje humano.

Durante el curso se estudian modelos predictivos, implementacion en Python y aplicacion a problemas reales. La semana 1 introduce la logica general del curso y arranca con dos bloques tecnicos clave:

1. **Como evaluar bien un modelo.**
2. **Como trabajar con arboles de decision y ensamblajes.**

## 2. Por que dividir los datos

Uno de los mensajes mas importantes de la semana es que **no se debe entrenar y evaluar un modelo con exactamente los mismos datos**. Si se hace eso, el modelo puede parecer excelente aun cuando en realidad no generaliza bien.

### Train y test

La separacion mas basica es:

- **Train:** datos con los que el modelo aprende.
- **Test:** datos reservados para medir su desempeno final.

La idea es simple: el conjunto de prueba simula datos "nuevos". Si el modelo funciona bien alli, hay una mejor evidencia de que realmente aprendio patrones utiles y no solo memorizo.

### Train, validacion y test

Cuando ademas se quieren comparar varios modelos o ajustar hiperparametros, se necesita una tercera division:

- **Train:** para entrenar.
- **Validacion:** para comparar configuraciones y escoger la mejor.
- **Test:** para la evaluacion final.

La razon es evitar contaminar la medicion final. Si se usa el test para decidir entre modelos, en la practica ese conjunto deja de ser una prueba honesta.

### Validacion cruzada

La presentacion tambien introduce la **validacion cruzada**, una tecnica en la que el conjunto de entrenamiento se divide en varias partes y se rota cual parte se usa para validar. Esto ayuda a:

- aprovechar mejor los datos disponibles;
- obtener una estimacion mas estable del desempeno;
- reducir la dependencia de una sola particion train/val.

## 3. Matriz de confusion: la base para entender clasificacion

En problemas de clasificacion, muchas metricas salen de la **matriz de confusion**. Esta cruza:

- lo que el modelo **predijo**;
- con lo que era **real**.

Sus cuatro componentes son:

- **Verdaderos positivos (VP):** el modelo predijo positivo y era positivo.
- **Verdaderos negativos (VN):** el modelo predijo negativo y era negativo.
- **Falsos positivos (FP):** el modelo predijo positivo, pero era negativo.
- **Falsos negativos (FN):** el modelo predijo negativo, pero era positivo.

Entender estos cuatro casos es fundamental porque cada problema tiene costos distintos. En algunos contextos es mas grave un falso positivo; en otros, un falso negativo.

## 4. Accuracy: util, pero no suficiente

La semana enfatiza que el **accuracy** es:

\[
Accuracy = \frac{predicciones\ correctas}{total\ de\ observaciones}
\]

Su ventaja es que es facil de interpretar. Sin embargo, puede ser **enganoso cuando las clases estan desbalanceadas**.

### Ejemplo conceptual

Si 95 de 100 casos son negativos y solo 5 son positivos, un modelo que siempre predice "negativo" logra 95% de accuracy, pero no detecta ningun positivo. Es decir, **parece bueno en la metrica, pero es inutil para el objetivo real**.

Por eso el accuracy no debe usarse de forma aislada en problemas desbalanceados o cuando los errores tienen costos distintos.

## 5. Precision y recall

Estas dos metricas responden preguntas diferentes.

### Precision

La **precision** responde:

> De todos los casos que el modelo marco como positivos, ¿cuantos realmente lo eran?

\[
Precision = \frac{VP}{VP + FP}
\]

Es importante cuando un **falso positivo es costoso**. La presentacion usa como intuicion situaciones donde hay recursos limitados: si solo se puede ayudar a cierto numero de personas, conviene que las seleccionadas realmente necesiten esa ayuda.

### Recall

El **recall** responde:

> De todos los positivos reales, ¿cuantos logro detectar el modelo?

\[
Recall = \frac{VP}{VP + FN}
\]

Es importante cuando un **falso negativo es costoso**. Por ejemplo, si dejar pasar un caso positivo tiene consecuencias graves, interesa capturar la mayor cantidad posible de positivos, incluso si aparecen algunos falsos positivos.

## 6. El trade-off entre precision y recall

Normalmente no es posible maximizar ambas metricas al mismo tiempo sin compromisos. Si el modelo se vuelve mas estricto para declarar positivos:

- suele subir la **precision**;
- pero puede bajar el **recall**.

Si se vuelve mas flexible:

- suele subir el **recall**;
- pero puede bajar la **precision**.

Este equilibrio depende del problema de negocio, del costo de los errores y del umbral de decision usado para convertir probabilidades en clases.

## 7. F1-score

El **F1-score** busca un balance entre precision y recall. Es especialmente util cuando:

- hay desbalance de clases;
- interesa combinar ambas perspectivas en una sola medida;
- no se quiere optimizar solo una de las dos.

Conceptualmente, el F1 castiga los casos en los que una metrica es alta y la otra muy baja. Por eso es una medida mas exigente que el accuracy en muchos problemas de clasificacion.

## 8. ROC-AUC y el papel del umbral

Muchos modelos de clasificacion no producen directamente una clase, sino una **probabilidad**. Luego se usa un **umbral** para convertirla en 0 o 1.

La curva **ROC** compara:

- la **tasa de verdaderos positivos**;
- contra la **tasa de falsos positivos**;

para distintos umbrales.

El **AUC** resume esa curva en un solo numero. Su ventaja principal es que permite comparar clasificadores **sin depender de un unico umbral**.

Idea clave de la semana:

- si un modelo separa mejor las probabilidades de positivos y negativos, su ROC-AUC sera mayor;
- un modelo cercano al azar tendra un comportamiento cercano a la diagonal de 45 grados.

## 9. Metricas para regresion

La presentacion tambien cubre metricas para problemas donde la variable objetivo es numerica.

### RMSE

El **Root Mean Squared Error (RMSE)** penaliza mas fuertemente los errores grandes porque eleva al cuadrado las diferencias. Es util cuando los errores extremos importan mucho.

### MAE

El **Mean Absolute Error (MAE)** usa el valor absoluto de los errores. Es mas interpretable y menos sensible a outliers que el RMSE.

### MAPE

El **Mean Absolute Percentage Error (MAPE)** expresa el error en terminos porcentuales. Puede ser intuitivo para negocio, aunque hay que usarlo con cuidado si existen valores reales cercanos a cero.

### R2

El **R2** suele interpretarse como la proporcion de varianza explicada por el modelo. La presentacion hace una advertencia importante:

- en **modelos lineales**, la interpretacion clasica es mas directa;
- en **modelos no lineales**, el numero sigue siendo informativo, pero **no siempre debe interpretarse de la misma manera**.

Ese matiz es muy importante porque evita usar mecanicamente una metrica fuera de contexto.

## 10. Como escoger la metrica correcta

Una idea transversal de la semana es que **no existe una unica metrica "mejor" para todos los casos**. La metrica correcta depende de:

- si el problema es de **clasificacion** o **regresion**;
- si las clases estan balanceadas;
- si el costo de un falso positivo y un falso negativo es diferente;
- si importa mas castigar errores grandes o medir error promedio interpretable.

Una forma practica de pensar la eleccion es:

- **Accuracy:** util si las clases estan balanceadas.
- **Precision:** util si los falsos positivos son costosos.
- **Recall:** util si los falsos negativos son costosos.
- **F1:** util si se necesita balance entre precision y recall.
- **ROC-AUC:** util para comparar clasificadores a traves de varios umbrales.
- **RMSE:** util si los errores grandes deben penalizarse con fuerza.
- **MAE:** util si se quiere una medida directa del error absoluto.
- **MAPE:** util si el error porcentual es mas interpretable para el contexto.

## 11. Conexion con los notebooks de la semana

Estos conceptos no aparecen aislados. En los notebooks y el taller de la semana se aplican de inmediato:

- se separan datos en entrenamiento y prueba;
- se calibran parametros de modelos;
- se usan metricas de clasificacion como **accuracy** y **F1-score**;
- se trabajan tambien problemas de **regresion**, donde entran metricas como error cuadratico o absoluto.

En otras palabras, las presentaciones dan el lenguaje conceptual que luego se aterriza en codigo.

## 12. Ideas clave para recordar

Si tuviera que resumir la parte teorica de la semana 1 en pocas ideas, serian estas:

1. **Evaluar bien es tan importante como entrenar bien.**
2. **Nunca se debe juzgar un modelo solo con el conjunto de entrenamiento.**
3. **La metrica correcta depende del problema y del costo de los errores.**
4. **Accuracy no basta cuando hay desbalance.**
5. **Precision, recall, F1 y ROC-AUC permiten una lectura mas rica de clasificadores.**
6. **En regresion, RMSE, MAE, MAPE y R2 cuentan historias distintas sobre el error.**

Con esta base, la semana pasa naturalmente al estudio de modelos concretos: primero arboles de decision y luego metodos de ensamblaje.
