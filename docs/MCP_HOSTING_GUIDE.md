# Guía de hosting — servidor MCP de MPCR

---

## 1. Resumen: qué necesita llave y qué no

| Nivel | Herramientas | ¿Llave? | Qué hace falta |
|---|---|---|---|
| **L0** | `get_vocabulary`, `search_species`, `get_species`, `resolve_common_name` | No | SQLite del bundle |
| **L1** | `parse_distribution_text`, `get_occurrences`, `get_occurrence_points`, `render_species_map`, `get_species_map_image`, `where_to_see` | No | Bundle + shapefiles + DEM |
| **L2a** | `semantic_search`, `search_by_description` | No* | Índice pgvector local |
| **L2b** | `answer_question` | **Sí** | `OPENROUTER_API_KEY` |

\* El backend vectorial por defecto es **pgvector local** (`MPCR_VECTOR_BACKEND=pgvector`):
embeddings calculados en la máquina, Postgres embebido, cero llamadas remotas.
**Nota:** hoy el chequeo de llaves en `_check_l2()` pide `OPENROUTER_API_KEY` también para
estas dos, aunque no la usen. Es un bug conocido; mientras no se corrija, hay que poner
la llave igual o parchear esa función.

Solo `answer_question` necesita de verdad una llave, y **puede ser gratuita** (ver §6).

---

## 2. Requisitos

- **Python 3.12**
- **~8 GB de disco**: 1.5 GB de dependencias, 115 MB de datos, 1.1 GB de índice pgvector,
  2.5 GB del modelo de embeddings (se descarga solo la primera vez)
- **8 GB de RAM** recomendados. El modelo de embeddings pesa ~2 GB en memoria
- Red solo para instalar y para consultar GBIF de especies que no estén en caché

No hace falta instalar Postgres, GDAL ni QGIS: Postgres viene embebido (`pgserver`) y
GDAL/GEOS vienen dentro de las ruedas de `rasterio` y `fiona`.

---

## 3. Instalación

### 3.1 Código

```
git clone <URL-del-repo> CR-BioLM
cd CR-BioLM
git checkout mpcr-rag
```

La rama `mpcr-rag` todavía no está fusionada en `main`.

### 3.2 Dependencias

**No usar `requirements.txt` de la raíz**: está en UTF-16 y trae todo el stack de
desarrollo (torch, jupyter, shap). **Tampoco alcanza `requirements_app.txt`**: ese es
para la app web y no incluye geopandas, rasterio ni matplotlib, que el servidor MCP
importa al arrancar.

```
python -m venv venv
venv\Scripts\activate
```

En Linux o macOS: `source venv/bin/activate`

```
pip install "mcp[cli]" geopandas rasterio shapely matplotlib pandas numpy pygbif python-dotenv rapidfuzz requests
```

Solo si van a usar búsqueda semántica (`semantic_search`, `search_by_description`):

```
pip install psycopg2-binary pgserver sentence-transformers
```

`sentence-transformers` arrastra PyTorch (~2 GB). Si no necesitan búsqueda semántica,
omitan este segundo bloque y ahorran bastante espacio y tiempo de instalación.

### 3.3 Datos

Los datos geoespaciales están en `.gitignore`. Se entregan aparte, en un bundle
generado con:

```
python utils/make_handoff_bundle.py -o <carpeta-destino>
```

Produce `mpcr_rag_data_bundle.zip` (~67 MB comprimido, ~108 MB extraído). **Se
descomprime en la raíz del repo**: las rutas ya vienen relativas a ella.

```
data_raw/regiones_botanicas/        Regiones Botánicas (José Araya, 2026)
data_raw/topography/altitud_cr.tif  DEM ~1 km, de aquí sale la máscara cian
data_raw/vectors/                   Áreas protegidas (SINAC)
data_raw/Cartografia/               Provincias, cantones, distritos (IGN)
mpcr_rag/data/fichas.sqlite         Catálogo, 6,946 especies
mpcr_rag/data/gbif_cache/           Ocurrencias ya descargadas (519 especies)
```

Los shapefiles necesitan todos sus archivos hermanos (`.shx`, `.dbf`, `.prj`, `.cpg`).
El script los incluye; copiándolos a mano es fácil olvidarlos y el error que da después
no es obvio.

Verificación:

```
python -c "import sqlite3; from mpcr_rag import config as C; print(sqlite3.connect(C.SQLITE_PATH).execute('SELECT COUNT(*) FROM fichas').fetchone())"
```

Debe imprimir `(6946,)`.

---

## 4. Índice vectorial (solo para búsqueda semántica)

El índice pgvector **no se entrega**: pesa 1.1 GB y se reconstruye desde el SQLite.

```
python -m mpcr_rag.store.pg_store sync
```

Arranca un Postgres embebido, calcula los embeddings de las 6,946 fichas (20,587
chunks) y los guarda. La primera vez descarga el modelo `intfloat/multilingual-e5-large`
(~2.5 GB) y tarda: con GPU unos minutos, en CPU puede ser una hora o más. Se hace una
sola vez.

Comandos de control:

```
python -m mpcr_rag.store.pg_store status
python -m mpcr_rag.store.pg_store start
python -m mpcr_rag.store.pg_store stop
```

**Importante:** `MPCR_PGDATA` no debe apuntar a una carpeta sincronizada con Google
Drive, OneDrive o Dropbox. Una base de datos viva más sincronización da conflictos de
lock y corrupción. Por defecto va a `~/Documents/Tesis/pgdata/mpcr`, fuera del repo.

Si prefieren un Postgres propio en vez del embebido, definan `MPCR_PG_URL` y el
servidor lo usa en lugar de levantar el suyo.

---

## 5. Levantar el servidor

El servidor habla **stdio**, no HTTP: no se levanta en un puerto, lo lanza el cliente
MCP. El registro va en `.mcp.json`, en la raíz del repo:

```json
{
  "mcpServers": {
    "mpcr": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "mpcr_rag.mcp.server"],
      "cwd": "${workspaceFolder}",
      "env": {}
    }
  }
}
```

Prueba directa, sin cliente MCP:

```
python -c "import asyncio, json, mpcr_rag.mcp.server as S; asyncio.run(S.server.list_tools())"
```

Explorador interactivo con los esquemas de cada herramienta (necesita Node.js):

```
mcp dev mpcr_rag/mcp/server.py
```

Desde Claude Code: reiniciar y correr `/mcp`. Debe aparecer `mpcr` conectado con 13
herramientas. La primera vez pide aprobar el servidor del proyecto.

---

## 6. Llave de API y elección de modelo

Solo `answer_question` la necesita de verdad. **Puede ser gratuita**, pero igual hay
que registrarse.

### 6.1 Obtener la llave

1. Crear cuenta en https://openrouter.ai
2. Ir a **Keys** y crear una nueva
3. Guardarla en un archivo `.env` en la raíz del repo:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

No piden tarjeta si solo se usan modelos gratuitos.

### 6.2 Dónde se cambia el modelo

Por variable de entorno, sin tocar código:

```
MPCR_ENRICH_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

Definida en `mpcr_rag/config.py`. El valor por defecto es `openai/gpt-4o-mini`, que
**es de pago**: para hosting conviene fijar uno gratuito desde el principio.

**No hace falta un modelo con visión.** El servidor MCP solo manda texto. Las imágenes
son cosa del pipeline de experimentos (`main.py`), que usa otra variable
(`CRBIOLM_MULTIMODAL_MODELS`) y queda fuera de esta guía.

### 6.3 Qué modelos gratuitos hay disponibles

La lista cambia seguido, así que conviene consultarla en vez de copiarla:

```
python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('https://openrouter.ai/api/v1/models'))['data']; [print(m['id'], m.get('context_length')) for m in d if float(m.get('pricing',{}).get('prompt',1))==0]"
```

En octubre de 2026 había 19 modelos con precio cero. Los `:free` tienen límites de tasa
más estrictos y a veces desaparecen: si `answer_question` empieza a fallar, lo primero
que hay que revisar es si el modelo configurado sigue existiendo.

---

## 7. Ver la base de datos en VSCode

El puerto está **fijado en 5433** precisamente para esto: `pgserver` por su cuenta elige
un puerto aleatorio en cada arranque, lo que rompería cualquier conexión guardada.

1. Instalar la extensión **PostgreSQL** (Microsoft), o **SQLTools** con el driver
   **SQLTools PostgreSQL**
2. Crear una conexión nueva con estos datos:

| Campo | Valor |
|---|---|
| Host | `127.0.0.1` |
| Puerto | `5433` |
| Base de datos | `postgres` |
| Usuario | `postgres` |
| Contraseña | *(vacía)* |

URL equivalente: `postgresql://postgres@127.0.0.1:5433/postgres`

El servidor tiene que estar corriendo (`python -m mpcr_rag.store.pg_store start`). Para
ver los valores reales en cualquier momento, incluido el conteo de fichas y chunks:

```
python -m mpcr_rag.store.pg_store status
```

Los mismos datos sirven para DBeaver y pgAdmin. El puerto 5432 se dejó libre a propósito
por si tienen un Postgres de sistema instalado.

---

## 8. El formato de respuesta

Toda herramienta devuelve un sobre de procedencia. No es decoración: es el mecanismo que
hace verificable cada dato, y la interfaz que lo consuma debería mostrarlo.

```json
{
  "value": { "...": "el dato" },
  "source": "MPCR",
  "citation": "Manual de Plantas de Costa Rica, Tomo IV, p. 320",
  "confidence": "exact",
  "caveat": ""
}
```

- `source`: `MPCR`, `GBIF`, `DEM`, `SINAC`, o combinaciones con `+`
- `confidence`: `exact` (textual del Manual), `estimated` (derivado por reglas, GBIF o
  DEM), `insufficient` (no hay datos suficientes; `value` puede venir `null`)
- `caveat`: **si no está vacío, se muestra siempre.** Avisa cuando el dato es sospechoso

Dos reglas que conviene respetar en cualquier cliente:

1. `confidence: "insufficient"` **no es un error**. Significa "no sabemos", no un 500
2. `citation` se muestra. Es el punto entero del proyecto: cada afirmación es rastreable
   al Manual, con tomo y página

---

## 9. Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: mpcr_rag` | Se ejecutó fuera de la raíz del repo | `cd` a la raíz, o definir `PYTHONPATH` |
| Arranque lento (varios segundos) | Importa geopandas, rasterio, matplotlib | Normal. Es costo de importación, una vez por proceso |
| `semantic_search` responde "API key not configured" | El bug de `_check_l2()` (ver §1) | Poner `OPENROUTER_API_KEY` igual, o parchear la función |
| `sync` tarda horas | Embeddings calculados en CPU | Normal sin GPU. Se corre una vez y el pgdata queda |
| Postgres no levanta | `MPCR_PGDATA` en carpeta sincronizada | Moverlo a disco local |
| Mapa sin capa cian | El Manual da una elevación de un solo punto | Esperado. Llega explicado en `caveat`. Ejemplo: *Talamancaster minusculus* |
| Especie no encontrada | Ortografía o sinónimo taxonómico | Probar con `search_species`. El Manual usa la nomenclatura vigente a su fecha |
| Primera consulta GBIF lenta | Especie fuera del caché | Normal, queda cacheada. GBIF da 503 intermitentes; hay reintentos con espera |

---

## 10. Créditos y licencias

Las capas tienen licencias distintas:

- **Manual de Plantas de Costa Rica** — Hammel, Grayum, Herrera y Zamora (eds.),
  Missouri Botanical Garden. Texto con derechos reservados; citar tomo y página
- **Ocurrencias GBIF** — snapshot citable, DOI 10.15468/dl.8yhee8
- **Cartografía base** — Instituto Geográfico Nacional (IGN), Costa Rica
- **Áreas protegidas** — SINAC
- **Regiones botánicas** — José Araya, 2026

Antes de publicar algo con estos mapas (tesis, presentación, sitio web), consultar sobre
atribución: las condiciones de redistribución de las capas base siguen en confirmación.
