# Prototipo de API de popularidad de canciones en Spotify sobre ECS Fargate

Este directorio contiene un prototipo para entrenar, empaquetar y servir un modelo de prediccion de popularidad de canciones de Spotify. La aplicacion esta pensada para ejecutarse localmente en Flask y desplegarse en AWS usando una imagen publicada en **Amazon ECR** y un servicio en **Amazon ECS con launch type Fargate**, corriendo en una **subred publica**.

## Arquitectura objetivo en AWS

La configuracion objetivo del despliegue es la siguiente:

- **ECR** almacena la imagen Docker de la API.
- **ECS** orquesta la ejecucion del contenedor.
- **Fargate** ejecuta la tarea sin administrar servidores.
- La tarea corre en una **subred publica** con IP publica asignada.
- El contenedor expone el puerto **8000**.

## Estructura del proyecto

- `train_model.py`: entrena y serializa el pipeline de prediccion.
- `model_features.py`: concentra la logica de transformacion y generacion de variables.
- `inference.py`: carga el modelo serializado, valida entradas y arma el `DataFrame` para inferencia.
- `app.py`: API Flask/Flask-RESTX con documentacion Swagger.
- `csv_to_request_json.py`: convierte un CSV con el esquema de entrenamiento al JSON esperado por la API.
- `spotify_popularity_rf.pkl`: artefacto serializado del modelo entrenado.
- `Dockerfile`: definicion de la imagen para ejecucion local o en ECS.
- `requirements.txt`: dependencias de Python.

## Procedimiento seguido para entrenar el pipeline

### 1. Fuente de datos

El entrenamiento parte del archivo CSV definido en `TRAIN_URL` dentro de `train_model.py`. Ese dataset contiene las variables de entrada de la cancion y la variable objetivo `popularity`.

### 2. Variables de entrada y variable objetivo

La variable objetivo es:

- `popularity`

Las variables base de entrada incluyen:

- texto/categoricas: `track_name`, `album_name`, `artists`, `track_genre`
- numericas: `duration_ms`, `explicit`, `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `time_signature`

### 3. Procesamiento y construccion de variables

La clase `SpotifyFeatureBuilder` en `model_features.py` implementa la transformacion previa al modelo. El flujo seguido es:

1. **Normalizacion de artistas**: separa listas de artistas y obtiene el artista principal.
2. **Limpieza de texto**: estandariza nombres de track y album para crear indicadores.
3. **Features de suavizamiento por agregacion**:
   - popularidad por artista
   - popularidad por genero
   - popularidad por album
   - interaccion artista principal-genero
   - interaccion album-genero
4. **Features de soporte y orden**:
   - soporte del artista con mayor score
   - score del artista principal
   - medias ponderadas por orden de artistas
   - diferencia entre artista principal y mejor artista
5. **Features regex sobre `track_name` y `album_name`**:
   - presencia de remix, live, feat, version, remaster, christmas, acoustic, etc.
6. **Consolidacion del set final de variables** en `engineered_numeric_columns`.

Durante inferencia, el contrato externo recibe `explicit` como booleano, pero antes de pasar al modelo se normaliza internamente a `0/1`, que es el formato usado en entrenamiento.

#### Logica detallada de las features generadas

##### a. Parsing y normalizacion de artistas

La columna `artists` puede contener varios artistas en una sola cadena. El transformador:

- reemplaza `;` por `,`
- separa la cadena por comas
- elimina espacios sobrantes
- conserva la lista ordenada de artistas
- define como **artista principal** al primer elemento de la lista

Esto permite diferenciar entre:

- el mejor artista disponible segun historial
- el artista principal declarado
- el efecto del orden de aparicion de los artistas

##### b. Scores suavizados por artista, genero y album

El transformador aprende promedios historicos de popularidad usando el set de entrenamiento. En lugar de usar un promedio crudo, aplica **suavizamiento** para evitar que grupos con muy pocas observaciones produzcan scores extremos.

La idea general es:

```text
score_suavizado = (conteo * promedio_grupo + alpha * promedio_global) / (conteo + alpha)
```

donde:

- `promedio_grupo` es la popularidad promedio del artista, genero o album
- `conteo` es el numero de observaciones del grupo
- `promedio_global` es la media global de `popularity`
- `alpha` controla que tanto se "jala" el score hacia la media global

De esta forma se construyen:

- `artist_popularity_dict_`
- `genre_popularity_dict_`
- `album_popularity_dict_`

y luego se exponen en features como:

- `artist_popularity_score`
- `artist_popularity_primary`
- `genre_popularity_score`
- `album_popularity_score`

##### c. Score del mejor artista y soporte asociado

Para una cancion con varios artistas, el transformador calcula el score historico de cada uno y toma el mayor.

De ahi salen dos variables:

- `artist_popularity_score`: el mejor score de popularidad entre todos los artistas asociados a la cancion
- `artist_support_for_max`: la cantidad de observaciones historicas del artista que obtuvo ese mejor score

La intuicion es separar dos cosas:

- **que tan prometedor es el mejor artista asociado**
- **que tanta evidencia historica existe para confiar en ese score**

##### d. Score del artista principal

No siempre el artista con mayor historial coincide con el primero listado. Por eso se calcula:

- `artist_popularity_primary`: score del primer artista en `artists`

Esta variable captura el peso del artista principal declarado, incluso cuando haya colaboraciones con artistas mas populares.

##### e. Ordered scores y ordered supports

Cuando hay varios artistas, el orden puede contener senal. El transformador asigna pesos decrecientes segun posicion:

```text
peso_i = 1 / (i + 1)
```

Es decir:

- primer artista: peso `1`
- segundo artista: peso `1/2`
- tercer artista: peso `1/3`
- etc.

Con esos pesos calcula:

- `artist_popularity_ordered_mean`: promedio ponderado de los scores de artistas segun orden
- `artist_support_ordered_mean`: promedio ponderado de los soportes historicos segun orden

Estas features retienen mas informacion que solo quedarse con el mejor artista o con el artista principal, porque resumen la composicion completa del credito artistico respetando la jerarquia del orden.

##### f. Gap entre artista principal y mejor artista

La feature:

- `artist_gap_primary_vs_best`

se define como:

```text
score_artista_principal - mejor_score_entre_todos_los_artistas
```

Su comportamiento es:

- cercano a `0` cuando el artista principal ya es el mas fuerte
- negativo cuando otro artista secundario tiene mejor historial que el principal

Esto ayuda a detectar colaboraciones donde el "arrastre" comercial puede venir de un feat o de un artista invitado mas fuerte que el principal.

##### g. Scores jerarquicos por interaccion

Ademas de los scores simples por artista, genero y album, se construyen interacciones:

- `primary_artist_genre_popularity_score`
- `album_genre_popularity_score`

Estas no usan solo el promedio observado del par. Antes de suavizar, construyen un **prior jerarquico**:

- para artista principal-genero:
  `(score_artista + score_genero) / 2`
- para album-genero:
  `(score_album + score_genero) / 2`

Luego ese prior se combina con la evidencia observada del par usando otra vez una formula de suavizamiento. Esto hace que las interacciones:

- aprovechen informacion especifica cuando el par aparece muchas veces
- retrocedan a una estimacion razonable cuando el par es raro o nuevo

##### h. Deltas contra referencias base

Despues de calcular los scores de interaccion, se generan dos diferencias:

- `primary_artist_genre_popularity_delta_vs_genre`
- `album_genre_popularity_delta_vs_album`

Estas features miden el valor incremental de la interaccion frente a una referencia mas simple:

- artista principal + genero versus solo genero
- album + genero versus solo album

Si el delta es positivo, la combinacion especifica parece aportar mas que la referencia base. Si es negativo, la combinacion rinde por debajo de lo esperable para ese genero o album.

##### i. Flags de texto sobre `track_name`

El transformador convierte `track_name` a minusculas y genera variables binarias con expresiones regulares. Cada una vale `1.0` si el patron aparece y `0.0` en caso contrario.

Los flags actuales son:

- `track_name_has_dash`: detecta texto tipo " - "
- `track_name_has_bracketed_context`: detecta parentesis, corchetes o comillas
- `track_name_has_numbers`: detecta numeros u ordinales
- `track_name_has_feat`: detecta `feat`, `ft` o `featuring`
- `track_name_has_live`: detecta `live`, `en vivo` o `ao vivo`
- `track_name_has_from`: detecta la palabra `from`
- `track_name_has_mix_or_remix`: detecta `mix`, `remix` o `rmx`
- `track_name_has_version`: detecta `version`
- `track_name_has_remastered`: detecta `remaster` o `remastered`
- `track_name_has_original`: detecta `original`
- `track_name_has_christmas`: detecta `christmas`, `navidad` o `xmas`
- `track_name_has_acoustic`: detecta variantes de `acoustic`

Estos flags buscan capturar contextos editoriales o comerciales del titulo: versiones en vivo, remixes, reediciones, ediciones especiales o contenido estacional.

##### j. Flags de texto sobre `album_name`

Sobre `album_name` se repite la misma idea de binarizacion con patrones especificos del contexto editorial del album:

- `album_name_has_bracketed_context`
- `album_name_has_numbers`
- `album_name_has_live`
- `album_name_has_volume`
- `album_name_has_christmas`
- `album_name_has_soundtrack`
- `album_name_has_edition`
- `album_name_has_deluxe`
- `album_name_has_remaster`
- `album_name_has_version`
- `album_name_has_acoustic`
- `album_name_has_ep`
- `album_name_has_anniversary`

Estas variables ayudan a separar albums estandar de:

- ediciones deluxe
- remasters
- soundtracks
- EPs
- volumenes o aniversarios
- albums en vivo

##### k. Resultado final del feature engineering

El `DataFrame` transformado conserva:

- las columnas numericas originales
- los scores agregados
- los ordered scores y ordered supports
- los deltas de interaccion
- los flags binarios de track y album

Ese conjunto final se lista en `engineered_numeric_columns` y es exactamente lo que consume el `ColumnTransformer` antes de pasar al `RandomForestRegressor`.

### 4. Entrenamiento del modelo

En `train_model.py` se construye un pipeline de `scikit-learn` con estas etapas:

1. `SpotifyFeatureBuilder`
2. `ColumnTransformer` para conservar las variables numericas ya procesadas
3. `RandomForestRegressor`

Configuracion actual del modelo:

- `n_estimators=250`
- `random_state=42`
- `n_jobs=-1`
- `max_features='log2'`

Los hiperparametros de suavizamiento usados por el transformador estan definidos en `BEST_ALPHA_CONFIG`.

### 5. Serializacion del artefacto

Una vez entrenado, el pipeline completo se guarda con `joblib.dump(...)` en:

- `spotify_popularity_rf.pkl`

Eso permite servir exactamente el mismo pipeline en inferencia, sin reentrenar al iniciar la API.

## Como se sirve el modelo con Flask

La aplicacion se expone desde `app.py` usando **Flask** y **Flask-RESTX**.

### Endpoints

- `GET /`: interfaz Swagger generada por Flask-RESTX
- `GET /status`: pagina HTML simple de estado
- `GET /health`: healthcheck JSON
- `POST /predict`: endpoint de prediccion batch

### Flujo de inferencia

1. El cliente envia un JSON con una lista de registros en `instances`.
2. `app.py` valida el contrato del request.
3. `inference.py`:
   - carga el modelo una sola vez usando `@lru_cache`
   - convierte `instances` a `pandas.DataFrame`
   - valida que esten todas las columnas requeridas
   - transforma `explicit` de booleano a entero
   - ejecuta `model.predict(...)`
4. La API responde con:

```json
{
  "predictions": [83.94],
  "count": 1
}
```

### Configuracion de la aplicacion

- Framework web: `Flask`
- Documentacion de API: `Flask-RESTX`
- Servidor de produccion en contenedor: `gunicorn`
- Puerto del contenedor: `8000`
- Variable de entorno soportada: `PORT`

El `Dockerfile` arranca la aplicacion con:

```bash
gunicorn --workers 2 --bind 0.0.0.0:8000 app:app
```

## Contrato del API

El endpoint `POST /predict` espera un JSON con esta estructura:

```json
{
  "instances": [
    {
      "track_name": "Blinding Lights",
      "album_name": "After Hours",
      "artists": "The Weeknd",
      "track_genre": "pop",
      "duration_ms": 200040,
      "explicit": false,
      "danceability": 0.514,
      "energy": 0.73,
      "key": 1,
      "loudness": -5.934,
      "mode": 1,
      "speechiness": 0.0598,
      "acousticness": 0.00146,
      "instrumentalness": 0.000095,
      "liveness": 0.0897,
      "valence": 0.334,
      "tempo": 171.005,
      "time_signature": 4
    }
  ]
}
```

## Ejecucion local

### 1. Crear entorno e instalar dependencias

```bash
cd ecs_fargate_prototype
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Reentrenar el modelo si es necesario

```bash
python train_model.py
```

### 3. Ejecutar la API

```bash
python app.py
```

### 4. Probar endpoints

```bash
curl http://127.0.0.1:5000/
curl http://127.0.0.1:5000/status
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "track_name": "Blinding Lights",
        "album_name": "After Hours",
        "artists": "The Weeknd",
        "track_genre": "pop",
        "duration_ms": 200040,
        "explicit": false,
        "danceability": 0.514,
        "energy": 0.73,
        "key": 1,
        "loudness": -5.934,
        "mode": 1,
        "speechiness": 0.0598,
        "acousticness": 0.00146,
        "instrumentalness": 0.000095,
        "liveness": 0.0897,
        "valence": 0.334,
        "tempo": 171.005,
        "time_signature": 4
      }
    ]
  }'
```

## Generacion de JSON de prueba desde CSV

Si se parte de un CSV con el mismo formato usado en entrenamiento, puede generarse automaticamente el payload del API con:

```bash
python csv_to_request_json.py \
  https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2026/main/datasets/dataTrain_Spotify.csv \
  sample_request.json \
  --rows 5
```

Ese script:

- toma solo las columnas requeridas por la API
- ignora columnas extra como `popularity`
- convierte `explicit` de `0/1` a `true/false`
- escribe un archivo JSON con la clave `instances`

## Contenerizacion

Para construir y ejecutar la imagen localmente:

```bash
docker build -t spotify-popularity-api:latest ecs_fargate_prototype
docker run --rm -p 8000:8000 spotify-popularity-api:latest
```

Luego la API queda disponible en:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/status`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/predict`

## Procedimiento de despliegue en AWS

El flujo esperado de despliegue es:

1. construir la imagen Docker localmente
2. publicar la imagen en un repositorio de **Amazon ECR**
3. crear una **task definition** en ECS usando esa imagen
4. crear un **service** en **ECS Fargate**
5. asociar la tarea a una **subred publica** y asignar IP publica
6. abrir el puerto del contenedor en el security group correspondiente

En esta aproximacion, Flask se sirve dentro del contenedor y Gunicorn atiende las solicitudes HTTP dentro de la tarea Fargate.

## Dependencias principales

- `Flask`
- `flask-restx`
- `gunicorn`
- `joblib`
- `pandas`
- `scikit-learn`

## Consideraciones del prototipo

- el despliegue en subred publica simplifica la prueba inicial, pero no es la mejor opcion para produccion
- no incluye balanceador, dominio propio ni terminacion TLS
- para un ambiente productivo conviene agregar ALB, observabilidad y una estrategia de seguridad de red mas robusta
