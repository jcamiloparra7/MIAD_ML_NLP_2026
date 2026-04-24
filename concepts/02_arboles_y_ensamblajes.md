# Semana 1: arboles de decision y metodos de ensamblaje

Este documento explica los conceptos tecnicos trabajados en los notebooks y el taller de la semana 1. El eje tematico es claro: **construir, entender, calibrar e implementar arboles de decision y ensamblajes**, tanto manualmente como con `sklearn`.

## 1. Que problema buscan resolver estos modelos

Los materiales de la semana muestran que un mismo tipo de modelo puede aplicarse a distintos contextos:

- **Clasificacion:** por ejemplo, predecir si un salario es alto o bajo, o si una noticia sera popular.
- **Regresion:** por ejemplo, pronosticar el precio de un automovil o el numero de bicicletas alquiladas.

La idea central es que los arboles y ensamblajes son modelos flexibles que capturan relaciones no lineales y pueden trabajar bien con interacciones entre variables.

## 2. Arbol de decision: intuicion basica

Un **arbol de decision** divide los datos en grupos cada vez mas homogeneos mediante preguntas del tipo:

- ¿la variable es menor que cierto valor?
- ¿la observacion pertenece a cierta categoria?

Cada division crea ramas y el proceso continua hasta llegar a nodos finales u hojas. En esas hojas se produce la prediccion:

- una **clase** en clasificacion;
- un **valor promedio** o estimado en regresion.

La gran ventaja pedagogica del arbol es que su logica puede seguirse paso a paso. Por eso la semana empieza construyendolo manualmente.

## 3. Preparacion de datos antes de entrenar

En el laboratorio de arboles se trabaja primero la preparacion del conjunto de datos:

- importacion de datos y librerias;
- identificacion de predictores y variable respuesta;
- codificacion de variables categoricas;
- exploracion inicial de la informacion.

Esto deja una leccion importante: **el modelo no empieza en el algoritmo**. Antes hay que definir claramente:

1. que variable se quiere predecir;
2. que variables se usaran para explicarla;
3. como convertir los datos a un formato util para el algoritmo.

## 4. Como se construye un arbol manualmente

El notebook desarrolla el arbol "a mano" para que el estudiante entienda su mecanismo interno.

### Paso 1: proponer puntos de corte

Para cada variable numerica se consideran posibles umbrales de separacion. Por ejemplo:

> dividir observaciones con `Hits <= c` y `Hits > c`.

### Paso 2: medir que tan buena es la division

Luego se calcula una medida de impureza o ganancia. En el material aparece el **indice Gini** como criterio para clasificacion.

La idea es:

- una division buena deja grupos mas "puros";
- es decir, grupos donde la variable objetivo queda menos mezclada.

### Paso 3: escoger la mejor variable y el mejor corte

Se comparan todas las variables candidatas y todos sus puntos de corte para elegir la division que mas reduce la impureza.

### Paso 4: crecer el arbol de forma recursiva

Una vez hecha la primera division, se repite el mismo proceso dentro de cada rama:

- volver a evaluar variables;
- volver a buscar el mejor corte;
- seguir hasta cumplir una condicion de parada.

### Paso 5: predecir con el arbol construido

Para hacer una prediccion, una observacion recorre el arbol desde la raiz hasta una hoja, siguiendo las reglas definidas en cada nodo.

Este enfoque manual es muy valioso porque muestra que un arbol no es una "caja negra": es una secuencia de reglas elegidas para mejorar la separacion de los datos.

## 5. Criterios de parada y calibracion del arbol

Si un arbol crece demasiado, puede ajustarse excesivamente al entrenamiento. Eso se conoce como **sobreajuste**.

Por eso los materiales muestran la necesidad de **calibrar** el modelo, es decir, ajustar parametros que controlan su complejidad. Entre las ideas que aparecen en la semana estan:

- profundidad maxima del arbol;
- ganancia minima requerida para dividir;
- tamano minimo de nodos u observaciones;
- numero de candidatos de corte evaluados.

La intuicion es:

- un arbol muy pequeño puede quedar corto y subajustar;
- un arbol muy grande puede memorizar ruido;
- la calibracion busca un punto medio con mejor generalizacion.

## 6. Arboles con `sklearn`

Despues de entender el mecanismo manual, el curso muestra como usar la implementacion de `sklearn`. Esto permite:

- entrenar mas rapido;
- calibrar hiperparametros con mayor facilidad;
- aplicar el modelo a problemas reales sin reescribir toda la logica desde cero.

La ensenanza importante aqui es doble:

1. **primero entender el modelo por dentro**;
2. **luego usar librerias para implementarlo de manera eficiente**.

Ese orden ayuda a no usar herramientas de forma mecanica.

## 7. Importancia de variables

El notebook de arboles tambien introduce la **importancia de variables**. Este concepto busca responder:

> ¿que predictores contribuyeron mas a las divisiones utiles del modelo?

En arboles, una variable suele ser mas importante si participa repetidamente en particiones que mejoran mucho la pureza o reducen el error.

Esto es util porque:

- ayuda a interpretar el modelo;
- permite identificar predictores relevantes;
- puede guiar decisiones de negocio o futuras limpiezas de datos.

## 8. Que es un ensamblaje

Un **metodo de ensamblaje** combina varios modelos para producir una prediccion final. La intuicion es poderosa:

- un modelo individual puede cometer errores inestables;
- varios modelos combinados pueden reducir varianza y mejorar robustez.

En la semana se trabajan dos grandes ideas:

1. **Bagging**
2. **Combinacion de clasificadores mediante votacion**

## 9. Bagging

**Bagging** significa entrenar multiples modelos sobre diferentes muestras bootstrap del conjunto de entrenamiento.

### Bootstrap

Una muestra bootstrap se construye:

- tomando observaciones del entrenamiento;
- con reemplazo;
- hasta formar una nueva muestra del mismo tamano aproximado.

Eso implica que:

- algunas observaciones aparecen repetidas;
- otras quedan por fuera de una muestra particular.

### Por que funciona

Si se entrenan muchos modelos sobre muestras ligeramente distintas y luego se combinan:

- en clasificacion se puede votar;
- en regresion se puede promediar.

Con esto se reduce la inestabilidad de modelos sensibles a cambios en los datos, como los arboles.

## 10. Error out-of-bag

Como en cada muestra bootstrap quedan datos por fuera, esos datos pueden usarse para una evaluacion aproximada del desempeno. Esa idea aparece como **error out-of-bag (OOB)**.

Su valor practico es que permite medir generalizacion sin necesitar una particion adicional para cada modelo individual del ensamblaje.

## 11. Combinacion de modelos y votacion

El segundo notebook trabaja ensamblajes heterogeneos, es decir, combinaciones de modelos distintos.

### Votacion mayoritaria

Cada clasificador emite una prediccion y gana la clase con mas votos.

Esta estrategia es simple y efectiva cuando:

- los modelos individuales aportan perspectivas distintas;
- sus errores no son exactamente los mismos.

### Votacion ponderada

No todos los modelos tienen que pesar igual. En la **votacion ponderada**, algunos reciben mas influencia que otros segun su calidad.

La intuicion es:

- si un modelo es mas confiable, su voto deberia valer mas;
- si otro es mas debil, su efecto deberia ser menor.

Esto introduce una capa adicional de diseno: no solo se combinan modelos, tambien se decide **como** combinarlos.

## 12. Implementacion manual vs librerias

Los notebooks repiten un patron didactico muy valioso:

- primero construir la idea manualmente;
- luego replicarla con `sklearn`.

Eso ocurre con:

- arboles de decision;
- bagging;
- votacion mayoritaria;
- votacion ponderada.

La meta pedagogica es que el estudiante comprenda tanto la teoria como la implementacion practica.

## 13. Que aporta el taller

El taller de la semana traslada estos conceptos a ejercicios aplicados.

### Parte A: arboles de decision

Se usa un problema de alquiler de bicicletas para:

- hacer analisis descriptivo;
- interpretar graficos;
- ajustar una regresion lineal base;
- construir un arbol manual;
- entrenar un arbol con libreria y calibrarlo.

Esto muestra algo importante: antes de usar modelos mas complejos, conviene comparar contra baselines sencillos y entender la estructura del problema.

### Parte B: ensamblajes

Se usa un problema de popularidad de noticias para:

- comparar un arbol de decision con una regresion logistica;
- medir desempeno con **accuracy** y **F1-score**;
- construir ensamblajes con **votacion mayoritaria**;
- construir ensamblajes con **votacion ponderada**;
- analizar ventajas y desventajas de cada enfoque.

El taller no solo pide programar modelos; tambien exige **interpretar resultados**, que es donde realmente se consolida el aprendizaje.

## 14. Diferencia entre regresion y clasificacion en estos materiales

La semana muestra que las mismas herramientas pueden adaptarse a objetivos distintos:

- en **regresion**, la salida es numerica y se evalua con metricas de error;
- en **clasificacion**, la salida es una clase y se evalua con metricas como accuracy y F1.

Esto ayuda a entender que el nombre del algoritmo no basta. Siempre hay que preguntarse:

- cual es la variable objetivo;
- si el problema es continuo o discreto;
- con que metrica se va a juzgar el resultado.

## 15. Ventajas y limites de los arboles

### Ventajas

- Son faciles de interpretar.
- Capturan no linealidades e interacciones.
- Requieren poca transformacion conceptual para explicarlos.

### Limites

- Pueden ser inestables.
- Tienden a sobreajustar si no se calibran.
- Un solo arbol puede tener menor capacidad predictiva que un buen ensamblaje.

Precisamente por esas limitaciones surgen los metodos de ensamblaje como complemento natural.

## 16. Ventajas y limites de los ensamblajes

### Ventajas

- Mejoran estabilidad.
- Suelen aumentar desempeno predictivo.
- Reducen la dependencia de un unico modelo.

### Limites

- Son menos interpretables que un arbol individual.
- Requieren mas computo.
- Exigen decisiones adicionales sobre numero de modelos, pesos y configuracion.

## 17. Ideas clave para recordar

La semana 1 deja varias lecciones de fondo:

1. **Un arbol de decision es una secuencia de particiones optimizadas sobre los datos.**
2. **Construirlo manualmente ayuda a entender impureza, cortes y recursion.**
3. **Calibrar parametros es esencial para evitar sobreajuste.**
4. **La importancia de variables ayuda a interpretar el modelo.**
5. **Bagging mejora robustez al combinar muchos modelos entrenados sobre muestras bootstrap.**
6. **La votacion mayoritaria y ponderada son formas de fusionar clasificadores.**
7. **El valor de un modelo no se mide solo por implementarlo, sino por evaluarlo e interpretarlo correctamente.**

En conjunto, la carpeta de la semana 1 construye una base muy solida: primero ensena a **evaluar**, luego a **modelar con arboles**, y finalmente a **mejorar esos modelos mediante ensamblajes**.
