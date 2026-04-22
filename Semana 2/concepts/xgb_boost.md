# Boosting y XGBoost: guía práctica y conceptual

## 1. ¿Qué es boosting?

**Boosting** es una familia de métodos de ensamble en la que varios modelos simples, usualmente árboles pequeños, se construyen **de manera secuencial**. Cada nuevo modelo intenta corregir los errores cometidos por los modelos anteriores.

A diferencia de métodos como bagging o Random Forest, donde los árboles se entrenan en paralelo e independientemente, en boosting hay una dependencia entre iteraciones:

1. se entrena un primer modelo,
2. se identifican sus errores,
3. se entrena un nuevo modelo para corregirlos,
4. se repite el proceso varias veces.

La idea central es que la suma de muchos modelos débiles puede producir un predictor muy poderoso.

## 2. Intuición: por qué funciona

Un modelo individual pequeño puede no capturar toda la complejidad del problema. Boosting mejora el desempeño construyendo una secuencia de modelos donde cada uno se enfoca en los casos más difíciles.

En términos intuitivos:

- los primeros árboles capturan patrones generales,
- los siguientes corrigen errores residuales,
- la predicción final combina el aporte de todos.

Por eso, boosting suele lograr un desempeño muy alto en datos tabulares, especialmente cuando hay relaciones no lineales e interacciones entre variables.

## 3. Boosting vs. bagging

Aunque ambos son métodos de ensamble, tienen lógicas distintas:

| Método | Cómo construye los modelos | Objetivo principal |
| --- | --- | --- |
| **Bagging** | Modelos en paralelo | Reducir varianza |
| **Random Forest** | Bagging + selección aleatoria de variables | Reducir varianza y correlación entre árboles |
| **Boosting** | Modelos secuenciales | Reducir sesgo y corregir errores |

En general:

- **bagging** busca estabilidad,
- **boosting** busca corrección progresiva.

## 4. De AdaBoost a Gradient Boosting

Históricamente, boosting aparece en varias formas.

### 4.1. AdaBoost

AdaBoost aumenta el peso de las observaciones mal clasificadas para que el siguiente modelo les preste más atención.

### 4.2. Gradient Boosting

Gradient Boosting generaliza la idea y, en lugar de reasignar pesos de forma manual, entrena cada nuevo árbol para aproximar el **gradiente negativo** de una función de pérdida.

Esto permite usar boosting no solo en clasificación, sino también en regresión y en otros problemas con diferentes funciones objetivo.

## 5. ¿Qué es XGBoost?

**XGBoost** significa **Extreme Gradient Boosting**. Es una implementación optimizada de gradient boosting que se hizo muy popular por su combinación de:

- alto desempeño predictivo,
- velocidad,
- regularización,
- flexibilidad para calibrar hiperparámetros,
- buen manejo de datos tabulares.

No es solamente “otro boosting”; incorpora mejoras de ingeniería y de formulación matemática que lo hacen muy competitivo en la práctica.

## 6. Cómo funciona XGBoost a alto nivel

XGBoost construye una secuencia de árboles, donde cada nuevo árbol se ajusta para corregir el error acumulado de los anteriores.

En regresión, una forma intuitiva de verlo es:

1. se hace una predicción inicial,
2. se calcula el error o residuo,
3. se entrena un árbol para explicar ese residuo,
4. la predicción se actualiza,
5. se repite el proceso.

En clasificación la lógica es similar, aunque la corrección se formula en términos de una función de pérdida apropiada, como la log-loss.

La predicción final es la suma de los aportes de todos los árboles:

`predicción final = árbol 1 + árbol 2 + ... + árbol M`

## 7. ¿Por qué “gradient” boosting?

Se llama así porque el nuevo árbol no corrige errores de manera arbitraria, sino siguiendo la dirección que más reduce la función de pérdida, es decir, el gradiente.

En términos simples:

- si el modelo se equivoca mucho en algunas observaciones,
- el gradiente indica hacia dónde debe moverse para mejorar,
- el siguiente árbol aprende esa corrección.

XGBoost aprovecha esta idea de forma muy eficiente y usa además información de primer y segundo orden para optimizar mejor el ajuste.

## 8. Componentes clave que diferencian a XGBoost

### 8.1. Regularización

XGBoost incorpora penalizaciones para controlar la complejidad del modelo. Esto ayuda a evitar sobreajuste, algo importante porque boosting puede volverse muy agresivo si se dejan crecer demasiados árboles o árboles muy complejos.

### 8.2. Shrinkage o learning rate

Cada árbol nuevo corrige solo una parte del error, no todo de una vez. El parámetro **learning rate** reduce el tamaño del aporte de cada árbol.

Esto suele hacer el entrenamiento más estable:

- learning rate pequeño: aprendizaje más lento pero más controlado,
- learning rate grande: aprendizaje más rápido pero con mayor riesgo de sobreajuste.

### 8.3. Submuestreo de filas y columnas

XGBoost puede usar solo una fracción de observaciones y variables en cada iteración. Esto introduce aleatoriedad útil y puede mejorar generalización y velocidad.

### 8.4. Manejo eficiente del cómputo

Su implementación está diseñada para ser rápida y escalable, lo que explica buena parte de su popularidad en competencias y proyectos aplicados.

## 9. Hiperparámetros importantes

En la práctica, XGBoost tiene varios hiperparámetros clave:

| Parámetro | Qué controla | Efecto típico |
| --- | --- | --- |
| `n_estimators` | Número de árboles | Más árboles pueden mejorar desempeño hasta estabilizarse |
| `learning_rate` | Tamaño del aporte de cada árbol | Menor valor suele requerir más árboles |
| `max_depth` | Profundidad máxima de cada árbol | Controla complejidad |
| `min_child_weight` | Mínimo peso/soporte en nodos hijos | Evita divisiones demasiado específicas |
| `subsample` | Fracción de observaciones usada por árbol | Reduce sobreajuste y costo |
| `colsample_bytree` | Fracción de variables usada por árbol | Reduce correlación y complejidad |
| `gamma` | Mejora mínima requerida para dividir | Hace el árbol más conservador |
| `reg_alpha` | Regularización L1 | Favorece modelos más parsimoniosos |
| `reg_lambda` | Regularización L2 | Suaviza pesos y reduce sobreajuste |

En el material del curso, se destaca especialmente la calibración de:

- `learning_rate`,
- `gamma`,
- `colsample_bytree`.

## 10. Cómo interpretar algunos parámetros importantes

### 10.1. `learning_rate`

Controla cuánto corrige cada nuevo árbol.

- **Bajo**: el modelo aprende despacio, pero suele generalizar mejor.
- **Alto**: aprende rápido, pero puede ajustarse demasiado al entrenamiento.

Suele haber una relación práctica con `n_estimators`: si el learning rate es bajo, normalmente se necesitan más árboles.

### 10.2. `gamma`

Define cuánta mejora debe lograr una división para que valga la pena hacerla.

- **Gamma alto**: el modelo se vuelve más conservador.
- **Gamma bajo**: el árbol puede crecer más fácilmente.

Es útil para controlar complejidad y limitar divisiones poco útiles.

### 10.3. `colsample_bytree`

Indica qué proporción de variables se usa al construir cada árbol.

- valores menores introducen más aleatoriedad,
- pueden ayudar a generalizar mejor,
- también reducen costo computacional.

Si el valor es demasiado bajo, el modelo puede perder información relevante.

## 11. Clasificación y regresión con XGBoost

XGBoost puede emplearse tanto en:

- **clasificación**, con modelos como `XGBClassifier`,
- **regresión**, con modelos como `XGBRegressor`.

La estructura general es la misma, pero cambia la función de pérdida y, por lo tanto, la manera en que se calculan las correcciones en cada iteración.

## 12. Ventajas de XGBoost

- **Muy alto desempeño predictivo** en problemas tabulares.
- **Captura no linealidades** e interacciones complejas.
- **Incluye regularización**, algo no siempre tan explícito en otras implementaciones.
- **Permite calibración fina** del sesgo-varianza.
- **Tolera bien distintos tipos de relaciones entre variables**.
- **Suele funcionar muy bien como modelo competitivo de referencia**.

## 13. Desventajas y limitaciones

- Tiene **muchos hiperparámetros**, por lo que puede requerir más calibración.
- Es **menos interpretable** que modelos lineales o árboles simples.
- Puede **sobreajustar** si se usan árboles muy profundos o demasiadas iteraciones.
- El entrenamiento puede ser **más costoso** que el de modelos más simples.
- En conjuntos pequeños o problemas muy simples, su complejidad puede no justificarse.

## 14. Comparación con Random Forest

Aunque ambos usan árboles, su filosofía es diferente:

| Aspecto | Random Forest | XGBoost |
| --- | --- | --- |
| Construcción | Árboles independientes | Árboles secuenciales |
| Meta principal | Reducir varianza | Corregir errores y reducir sesgo |
| Calibración | Suele ser más simple | Suele ser más sensible a hiperparámetros |
| Interpretación | Difícil | También difícil, a veces más |
| Desempeño | Muy sólido | Frecuentemente superior si se calibra bien |

En muchos problemas tabulares, XGBoost puede superar a Random Forest, pero normalmente exige una calibración más cuidadosa.

## 15. Ejemplo conceptual

Supongamos que queremos predecir el precio de una vivienda.

1. El primer árbol hace una predicción inicial.
2. Observamos los errores entre predicción y valor real.
3. El segundo árbol aprende patrones asociados a esos errores.
4. El tercer árbol corrige lo que aún falta.
5. La suma de todos los árboles produce una predicción más precisa.

Cada árbol no intenta resolver el problema desde cero, sino mejorar lo ya construido.

## 16. Buenas prácticas al usar XGBoost

1. Empezar con una línea base sencilla.
2. Ajustar primero `n_estimators`, `learning_rate` y `max_depth`.
3. Revisar `gamma`, `subsample` y `colsample_bytree` para controlar sobreajuste.
4. Usar validación cruzada o un conjunto de validación separado.
5. Monitorear si el desempeño mejora realmente al agregar más árboles.
6. Evitar profundidades excesivas si el conjunto no es grande.
7. Comparar con Random Forest y modelos más simples antes de concluir que es la mejor opción.

## 17. Cuándo suele funcionar especialmente bien

XGBoost suele destacar cuando:

- el problema usa datos tabulares,
- hay interacciones complejas entre variables,
- el objetivo es maximizar desempeño predictivo,
- existe tiempo para calibrar hiperparámetros.

En problemas donde la interpretabilidad es prioritaria o el conjunto es pequeño y simple, un modelo menos complejo puede ser una mejor decisión.

## 18. Idea clave para recordar

XGBoost construye árboles **uno tras otro**, y cada nuevo árbol corrige una parte del error de los anteriores. Su fortaleza está en combinar:

1. la lógica del gradient boosting,
2. regularización,
3. control fino de la complejidad,
4. una implementación muy eficiente.

Por eso es uno de los algoritmos más importantes en machine learning aplicado a datos tabulares y una referencia habitual para tareas de clasificación y regresión.

## 19. ¿Cómo aprende exactamente de los errores de los árboles anteriores?

La idea más importante de gradient boosting es que **cada nuevo árbol no intenta predecir la variable objetivo desde cero**, sino aprender la **corrección** que debe hacerse sobre la predicción actual del modelo.

Supongamos que en la iteración `m-1` ya tenemos un modelo:

`F_{m-1}(x)`

Ese modelo produce una predicción para cada observación. El siguiente árbol se entrena para aprender qué ajuste debe hacerse sobre esa predicción.

### 19.1. Caso intuitivo: regresión con error cuadrático

Si estamos en un problema de regresión con pérdida cuadrática, el “error” que aprende el siguiente árbol es simplemente el **residuo**:

`r_i = y_i - F_{m-1}(x_i)`

donde:

- `y_i` es el valor real,
- `F_{m-1}(x_i)` es la predicción actual,
- `r_i` es la corrección necesaria.

Entonces el proceso es:

1. se empieza con una predicción inicial, por ejemplo la media;
2. se calculan los residuos;
3. se entrena un árbol pequeño para predecir esos residuos a partir de las variables;
4. se suma ese árbol al modelo actual.

La actualización del modelo puede escribirse como:

`F_m(x) = F_{m-1}(x) + eta * h_m(x)`

donde:

- `h_m(x)` es el nuevo árbol,
- `eta` es el *learning rate*.

Esto significa que el nuevo árbol aporta una corrección:

- positiva, si el modelo estaba subestimando;
- negativa, si el modelo estaba sobreestimando.

### 19.2. Ejemplo simple

Supongamos que queremos predecir tres valores reales:

| x | y |
| --- | --- |
| 1 | 10 |
| 2 | 12 |
| 3 | 20 |

Si el modelo inicial predice la media:

`F_0(x) = 14`

entonces los residuos son:

- para `x=1`: `10 - 14 = -4`,
- para `x=2`: `12 - 14 = -2`,
- para `x=3`: `20 - 14 = 6`.

El siguiente árbol no aprende directamente `y`, sino estos residuos. En otras palabras, aprende:

- correcciones negativas para valores bajos de `x`,
- correcciones positivas para valores altos de `x`.

Luego el modelo actualizado sería:

`F_1(x) = 14 + eta * h_1(x)`

Por eso se dice que el nuevo árbol aprende de los errores anteriores: porque su objetivo es modelar **la parte que todavía falta explicar**.

### 19.3. Caso general: gradiente y pseudo-residuos

En problemas más generales, especialmente en clasificación, el siguiente árbol no aprende necesariamente el residuo simple. Aprende el **gradiente negativo de la función de pérdida**, también llamado **pseudo-residuo**:

`r_i = - dL(y_i, F(x_i)) / dF(x_i)`

Esto significa que el árbol nuevo se ajusta en la dirección que más reduce la pérdida.

Por eso el nombre **gradient boosting**:

- se define una función de pérdida,
- se calcula cómo debería cambiar la predicción para reducirla,
- se entrena un árbol para aproximar esa corrección,
- se actualiza el modelo,
- y el proceso se repite.

### 19.4. La idea clave

En resumen:

- en regresión simple, los nuevos árboles aprenden **residuos**;
- en el caso general, aprenden **pseudo-residuos** o gradientes negativos;
- la predicción final es la suma de muchas pequeñas correcciones sucesivas.

Esa es la razón por la que boosting puede construir modelos muy potentes: cada árbol se especializa en corregir lo que el ensamble anterior todavía no ha aprendido bien.
