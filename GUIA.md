# Caso BreastDCEDL — guía completa

Todo lo que necesitas, en un solo documento. El enunciado oficial es
[`documentation/caso_breastdcedl.pdf`](documentation/caso_breastdcedl.pdf); esta
guía lo complementa y te da el código.

**Índice**

- [Parte A. Empezar](#parte-a-empezar)
  - [A1. El problema en una frase](#a1-el-problema-en-una-frase)
  - [A2. Descargar los datos](#a2-descargar-los-datos)
  - [A3. Qué hay en la carpeta](#a3-qué-hay-en-la-carpeta)
  - [A4. Mira las imágenes antes de programar](#a4-mira-las-imágenes-antes-de-programar)
- [Parte B. Los datos](#parte-b-los-datos)
  - [B1. El modelo mental](#b1-el-modelo-mental)
  - [B2. Las imágenes](#b2-las-imágenes)
  - [B3. `samples.csv`](#b3-samplescsv)
  - [B4. `patients.csv`](#b4-patientscsv)
  - [B5. Cargar los datos en Python](#b5-cargar-los-datos-en-python)
- [Parte C. El trabajo](#parte-c-el-trabajo)
  - [C1. Separación por paciente](#c1-separación-por-paciente)
  - [C2. Tu CNN desde cero](#c2-tu-cnn-desde-cero)
  - [C3. Desbalance de clases](#c3-desbalance-de-clases)
  - [C4. Evaluar por paciente](#c4-evaluar-por-paciente)
  - [C5. La evaluación final](#c5-la-evaluación-final)
  - [C6. La aplicación web](#c6-la-aplicación-web)
  - [C7. Entregables y defensa](#c7-entregables-y-defensa)
- [Parte D. Referencia](#parte-d-referencia)
  - [D1. Reglas que invalidan el trabajo](#d1-reglas-que-invalidan-el-trabajo)
  - [D2. Errores que más caros salen](#d2-errores-que-más-caros-salen)
  - [D3. Problemas técnicos](#d3-problemas-técnicos)
  - [D4. Diccionario de columnas](#d4-diccionario-de-columnas)
  - [D5. Preguntas para el informe](#d5-preguntas-para-el-informe)
  - [D6. Licencia y ética](#d6-licencia-y-ética)

---
---

# Parte A. Empezar

## A1. El problema en una frase

> A partir de una resonancia de mama con contraste tomada **antes** de empezar la
> quimioterapia, predecir si esa paciente alcanzará **pCR**: que tras el
> tratamiento no quede tumor invasivo.

No estás detectando un tumor que ya se ve. Estás anticipando **cómo va a
responder** a un tratamiento que aún no ha empezado. Es genuinamente difícil, y
por eso no esperes métricas de detección de imagen clásica.

| | train | test | total |
|---|---|---|---|
| pacientes | 1.097 | 176 | **1.273** |
| cortes | 10.945 | 1.758 | **12.703** |
| imágenes PNG | 32.835 | 5.274 | **38.109** |
| cortes con pCR=1 | 3.216 | 530 | 3.746 |
| cortes con pCR=0 | 7.729 | 1.228 | 8.957 |

La etiqueta está desbalanceada: **el 70,6 % de los casos son pCR=0**.

## A2. Descargar los datos

Los datos están en un bucket público. **No necesitas cuenta de Google ni
credenciales.**

Descarga primero el script, que es un único fichero:

```
https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl/descargar_datos.py
```

Y ejecútalo:

```bash
pip install requests
python descargar_datos.py
```

Baja la documentación, los metadatos y las 38.109 imágenes en paralelo, con
progreso. Tarda unos minutos según tu conexión.

**Es reanudable**: si se corta, vuelve a lanzarlo y solo descargará lo que falte.
Al terminar verifica que no falte ninguna imagen.

Opciones útiles:

```bash
python descargar_datos.py --destino C:\bdcedl    # ruta corta (Windows, ver D3)
python descargar_datos.py --pacientes 5          # solo 5 pacientes, para probar
python descargar_datos.py --solo-test            # solo el conjunto de prueba
python descargar_datos.py --hilos 32             # más paralelismo
```

**En Google Colab**, que es lo más rápido porque la red de Google va a Google:

```python
!curl -L -O https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl/descargar_datos.py
!python descargar_datos.py
import sys; sys.path.insert(0, "/content/breastdcedl")
```

> Colab borra el disco al desconectar. Puedes copiar la carpeta a tu Drive para
> no repetir la descarga, pero **entrena desde `/content/`**: leer desde Drive es
> mucho más lento.

**Si tienes el SDK de Google Cloud**, esto es aún más rápido:

```bash
gcloud storage rsync -r gs://usecasesf-breastdcedl-alumnos-8264/breastdcedl ./breastdcedl
```

**Sin descargar nada**, si solo quieres explorar los metadatos:

```python
import pandas as pd
BASE = "https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl"

samples  = pd.read_csv(f"{BASE}/metadata/samples.csv")     # 2,6 MB
patients = pd.read_csv(f"{BASE}/metadata/patients.csv")    # 208 KB
```

Incluso una imagen suelta, directamente desde su URL:

```python
import io, requests, numpy as np
from PIL import Image

r = requests.get(f"{BASE}/{samples.iloc[0].path_early}", timeout=30)
early = np.asarray(Image.open(io.BytesIO(r.content)).convert("L"), np.float32) / 255.0
```

**Comprobar que la descarga está completa:**

```python
from pathlib import Path
import pandas as pd

RAIZ = Path("breastdcedl")
samples = pd.read_csv(RAIZ / "metadata" / "samples.csv")
presentes = sum(1 for _ in (RAIZ / "dataset").rglob("*.png"))
print(f"{presentes} de {len(samples)*3}")        # debe dar 38109 de 38109
```

## A3. Qué hay en la carpeta

```
breastdcedl/
├── GUIA.md                        ← este documento: TODO está aquí
├── descargar_datos.py             ← descarga el dataset
├── utils_caso.py                  ← partición y agregación, ya resueltas
├── ver_muestras.py                ← visor de imágenes
├── LICENSE                        ← CC BY-NC 4.0
├── dataset/
│   ├── train/<paciente>/*.png     ← 1.097 pacientes
│   └── test/<paciente>/*.png      ← 176 pacientes
├── metadata/
│   ├── samples.csv                ← 1 fila por corte
│   ├── patients.csv               ← 1 fila por paciente
│   ├── excluded_patients.csv      ← 2 pacientes excluidas y por qué
│   └── statistics.json            ← recuentos por split y clase
└── documentation/
    ├── caso_breastdcedl.pdf       ← EL ENUNCIADO OFICIAL
    └── figures/
```

Son **dos documentos y tres scripts**, nada más. `GUIA.md` (este) y el enunciado
en PDF; el resto es código que puedes usar tal cual.

Necesitas Python con `numpy`, `pandas`, `pillow`, `matplotlib`, `scikit-learn` y
`torch`. Sin GPU, usa Colab.

## A4. Mira las imágenes antes de programar

**Media hora, y no te la saltes.** Abre una carpeta de paciente en el explorador
de archivos: son PNG normales, doble clic y los ves. Para eso están en ese formato.

Abre `_PRE` y `_EARLY` del mismo corte y alterna entre las dos. **Lo que se
ilumina es donde entra el contraste.** Ahí está la señal del problema.

```bash
python ver_muestras.py muestra ISPY1_1001_z012    # 3 fases + mapa de realce
python ver_muestras.py paciente ISPY1_1001        # los 10 cortes
python ver_muestras.py comparar                   # un pCR=0 frente a un pCR=1
```

Quien empieza a programar sin haber mirado los datos depura a ciegas durante días.

---
---

# Parte B. Los datos

## B1. El modelo mental

> **Un corte = 3 ficheros `.png`, uno por fase DCE.**
> Las imágenes están en `dataset/`; **todo lo demás está en los CSV**.

```
patients.csv            samples.csv                 dataset/<split>/<paciente>/
------------            -----------                 --------------------------
pid ─────────────────►  patient_id                        (imágenes)
(1 fila por paciente)   sample_id                              ▲
 pCR, age, HER2,        split, slice_index, pCR               │
 dataset, tum_vol...    path_pre, path_early, path_late ──────┘
                        (1 fila por corte, 3 rutas)
```

Cada paciente tiene su carpeta y aporta **unos 10 cortes** (mínimo 5):

```
dataset/train/ISPY1_1001/
├── ISPY1_1001_z012_PRE.png
├── ISPY1_1001_z012_EARLY.png
├── ISPY1_1001_z012_LATE.png
├── ISPY1_1001_z013_PRE.png
└── ...                          (10 cortes × 3 fases = 30 ficheros)
```

**Cómo se lee un nombre.** Es la confusión más habitual:

```
ISPY1_1001 _ z012 _ EARLY .png
^^^^^^^^^^   ^^^^   ^^^^^
 paciente    dónde  cuándo
             (alto) (fase)
```

`z012` y `z013` **no** son dos instantes de tiempo: son dos alturas distintas de
la mama, como dos rebanadas de pan. El tiempo lo indica el sufijo. Un mismo corte
`z012` existe en las tres fases.

> **1.273 pacientes, no 12.703 casos.** Los 12.703 cortes son más datos de
> entrenamiento, pero no más pacientes independientes. Todo lo demás sale de aquí.

## B2. Las imágenes

Cada fase es un **PNG en escala de grises de 8 bits, 256×256**:

| Sufijo | Fase | Qué es |
|---|---|---|
| `_PRE` | PRE | Antes de inyectar el contraste |
| `_EARLY` | EARLY | Post-contraste temprano |
| `_LATE` | LATE | Post-contraste tardío |

Las tres comparten una única ventana de normalización, así que sus intensidades
**son comparables entre sí**. Puedes verificar que lees bien los ficheros: en el
tejido debe cumplirse **PRE < EARLY**, porque el contraste ilumina. Se cumple en
el 100 % de las pacientes.

Que `LATE` sea a veces menor que `EARLY` **no es un error**: es el lavado del
contraste (*washout*), un patrón clínicamente relevante, real en el 17 % de las
pacientes.

**La operación que hace visible el tumor.** Las tres fases se parecen mucho a
simple vista. La señal está en su **diferencia**:

```python
realce = imagen[1] - imagen[0]        # EARLY - PRE
```

## B3. `samples.csv`

Una fila por corte. **Es el índice del dataset: itera sobre él, no sobre la
carpeta.**

| Campo | Para qué sirve |
|---|---|
| `sample_id` | Identificador del corte: `{patient_id}_z{slice_index:03d}` |
| `patient_id` | **Clave de agrupación.** Une con `patients.pid` |
| `split` | `train` o `test` |
| `path_pre` | Ruta del PNG antes del contraste, relativa a `breastdcedl/` |
| `path_early` | Ruta del PNG postcontraste temprano |
| `path_late` | Ruta del PNG postcontraste tardío |
| `slice_index` | Posición axial del corte |
| `pCR` | **Objetivo**: `1` respuesta completa, `0` enfermedad residual |
| `fold` | Pliegue de validación ya calculado: `0`–`4` en train, `-1` en test |

Tres cosas comprobadas, que te ahorran trabajo:

- **No hay nulos en `pCR`.** Todas las muestras están etiquetadas.
- **La etiqueta es constante dentro de cada paciente.** Es una propiedad de la
  paciente, no del corte.
- **Ninguna paciente aparece en dos splits.**

## B4. `patients.csv`

Una fila por paciente, 30 columnas, se une por `pid`. Ver [D4](#d4-diccionario-de-columnas).

Reparto por cohorte, que conviene mirar antes de sacar conclusiones:

| cohorte | train | test |
|---|---|---|
| `spy2` | 784 | 99 |
| `duke` | 209 | 42 |
| `spy1` | 104 | 35 |

Las cohortes **se procesaron de forma distinta**: I-SPY1 e I-SPY2 seleccionan
cortes por superficie de máscara tumoral 3D; Duke no tiene máscara completa y usa
los cortes centrales entre `mask_start` y `mask_end`. Es una fuente de sesgo real
y el enunciado espera que la discutas.

## B5. Cargar los datos en Python

Tienes `utils_caso.py` en la raíz. Resuelve lo que es fácil hacer mal:

```python
import utils_caso as uc

samples = uc.cargar_samples()
imagen  = uc.cargar_imagen(samples.iloc[0])    # (3, 256, 256) float32 en [0,1]
```

A mano, si prefieres entender qué pasa por dentro:

```python
import numpy as np
from PIL import Image

def cargar_fase(ruta):
    with Image.open(ruta) as png:
        return np.asarray(png.convert("L"), dtype=np.float32) / 255.0

base  = "dataset/train/ISPY1_1001/ISPY1_1001_z012"
pre   = cargar_fase(f"{base}_PRE.png")
early = cargar_fase(f"{base}_EARLY.png")
late  = cargar_fase(f"{base}_LATE.png")

imagen = np.stack([pre, early, late])          # (3, 256, 256)
```

El orden del apilado es **siempre PRE, EARLY, LATE**, y ese es el orden de canales
que recibe la CNN.

**El `Dataset` de PyTorch ya está hecho:**

```python
from torch.utils.data import DataLoader
import utils_caso as uc

samples = uc.cargar_samples()
entrenamiento, validacion = uc.particion(samples, fold_val=0)

dl = DataLoader(uc.BreastDCEDataset(entrenamiento), batch_size=32, shuffle=True)
x, y = next(iter(dl))
print(x.shape, x.dtype)    # torch.Size([32, 3, 256, 256]) torch.float32
```

> **Los 3 canales NO son colores.** Son tres instantes temporales. No apliques
> normalización de ImageNet, ni `ColorJitter`, ni `ImageFolder`. Y cualquier
> transformación geométrica debe aplicarse **al tensor completo**, nunca canal a
> canal: si volteas uno y no los otros, desalineas las fases y destruyes el realce.

---
---

# Parte C. El trabajo

## C1. Separación por paciente

**Es donde más gente suspende.** La unidad estadística es la paciente. Si dos
cortes de la misma paciente caen en entrenamiento y en validación, el modelo
reconoce anatomía ya vista y tus métricas suben sin significar nada.

Para que no puedas equivocarte, **la partición ya viene hecha** en la columna
`fold`, calculada por paciente y estratificada por pCR:

```python
entrenamiento, validacion = uc.particion(samples, fold_val=0)
```

La función comprueba internamente que no haya solape y lanza error si lo hubiera.

| fold | cortes | pacientes | proporción pCR |
|---|---|---|---|
| 0 | 2.187 | 219 | 0,2926 |
| 1 | 2.186 | 219 | 0,2928 |
| 2 | 2.191 | 220 | 0,2948 |
| 3 | 2.195 | 220 | 0,2961 |
| 4 | 2.186 | 219 | 0,2928 |

Cubren las 1.097 pacientes de train sin repetir ninguna. Para validación cruzada
completa, recorre `fold_val` de 0 a 4 y promedia.

> Si prefieres calcularla tú, usa
> `StratifiedGroupKFold(..., groups=train.patient_id)`. Lo que **no** puedes hacer
> es repartir los 10.945 cortes al azar.

## C2. Tu CNN desde cero

**Obligatorio: arquitectura propia.** Nada de ResNet, EfficientNet, VGG, DenseNet,
pesos preentrenados ni transfer learning. Una única salida (un *logit*).

Empieza pequeña —tres o cuatro bloques convolucionales— y crece solo si lo
necesitas. La primera capa lleva `in_channels=3`, pero por las tres fases DCE, no
por color.

Documenta y justifica: arquitectura, inicialización, regularización, optimizador,
tasa de aprendizaje, tamaño de lote, épocas, semilla y criterio de parada.

## C3. Desbalance de clases

En train hay 7.729 cortes pCR=0 frente a 3.216 pCR=1. El enunciado pide
**comparar pérdida normal y ponderada**:

```python
import torch

criterio_normal    = torch.nn.BCEWithLogitsLoss()
criterio_ponderado = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(uc.pos_weight(entrenamiento))     # 2,40
)
```

Entrena las dos y compara. No basta con decir cuál va mejor: explica **por qué** y
qué le ocurre a la sensibilidad en cada caso.

> Un modelo que prediga siempre pCR=0 acierta el **70,6 %**. Si tu accuracy ronda
> esa cifra, tu modelo no ha aprendido nada. Mira la matriz de confusión.

## C4. Evaluar por paciente

El modelo puntúa cortes, pero pCR es una propiedad de la paciente: sus ~10 cortes
comparten etiqueta. Hay que combinar esas probabilidades en una sola.

**Genera las probabilidades con `shuffle=False`** para que el orden coincida:

```python
dl = DataLoader(uc.BreastDCEDataset(validacion), batch_size=64, shuffle=False)

modelo.eval()
probs = []
with torch.no_grad():
    for x, _ in dl:
        probs.append(torch.sigmoid(modelo(x).squeeze(1)).numpy())
probs = np.concatenate(probs)

resultados = uc.evaluar_por_paciente(probs, validacion, umbral=0.5, metodo="mean")
```

Devuelve matriz de confusión, accuracy, sensibilidad, especificidad y AUC, **a
nivel de paciente**.

| `metodo` | Qué hace | Efecto |
|---|---|---|
| `mean` | Media de las probabilidades | Robusto. El punto de partida sensato |
| `max` | La más alta | Sensible: basta un corte sospechoso. Más falsos positivos |
| `median` | Mediana | Como la media, resistente a un corte atípico |
| `voto` | Fracción de cortes sobre el umbral | Voto mayoritario |

**El método y el umbral son decisiones de diseño que debes justificar.** Elígelos
con tu validación interna, **nunca con test**.

## C5. La evaluación final

Una sola vez, al final, con el modelo ya cerrado:

```python
test = uc.conjunto_test(samples)
```

De aquí salen las métricas y la matriz de confusión del informe. Si vuelves atrás
a tocar el modelo después de mirar test, tus resultados dejan de ser válidos.

## C6. La aplicación web

Con Streamlit, Gradio o equivalente, **desplegada y accesible por URL**. No vale
que funcione solo en tu portátil.

Debe permitir cargar una muestra o elegir un ejemplo, mostrar PRE, EARLY y LATE
por separado, y tras validar la entrada mostrar:

- Probabilidad de pCR y clase según el umbral.
- Umbral utilizado y versión o *checksum* del modelo.
- Dispositivo de inferencia y tiempo de respuesta.
- Parámetros del entrenamiento y métricas internas.
- Mensajes claros ante ficheros corruptos o valores no finitos.
- Aviso visible de uso educativo y de que **no tiene validez clínica**.

Requisitos: `model.eval()`, sin gradientes, resultado reproducible, sin aceptar
rutas arbitrarias del sistema, sin ejecutar contenido subido, con límite de
tamaño y formato.

> **La normalización de la app debe ser idéntica a la del entrenamiento.** Es el
> fallo más común y se detecta en la defensa.

## C7. Entregables y defensa

| Entregable | Detalle |
|---|---|
| Repositorio reproducible | Código, configuración, semillas |
| Pesos del modelo | El modelo entrenado |
| Informe | Decisiones, métricas, matriz de confusión, limitaciones |
| URL de la aplicación | Desplegada y funcionando |
| Cinco diapositivas | Ni una más |

**Las cinco diapositivas:**

1. Problema clínico, qué es pCR, objetivo.
2. Preparación de datos, separación por paciente, análisis de clases.
3. Arquitectura de la CNN, con tamaños y número de parámetros.
4. Entrenamiento, decisiones y resultados internos.
5. Aplicación web, demostración, limitaciones y conclusiones.

No son capturas de código: son tu razonamiento.

**En la defensa** el profesor elegirá **cinco muestras al azar de cinco pacientes
distintas**, de un conjunto privado que no has visto. Se cargarán una a una en tu
aplicación **sin que toques el código, sin reentrenar y sin reiniciar el
despliegue**.

No se comprueba solo si aciertas: se verifica que la predicción coincida con la de
tu pipeline, que no haya normalización distinta entre entrenamiento e inferencia,
y que los errores se gestionen bien.

---
---

# Parte D. Referencia

## D1. Reglas que invalidan el trabajo

1. **Mezclar cortes de una misma paciente** entre entrenamiento y validación.
2. **Usar test para ajustar** el modelo o los hiperparámetros.
3. **Usar modelos preentrenados** o transfer learning.
4. **Publicar la validación privada** o sus etiquetas.
5. El trabajo es **individual**.

## D2. Errores que más caros salen

**Tratar los 3 canales como RGB.** Son tres momentos temporales.

**Aumentado de datos canal a canal.** Desalinea las fases y destruye la señal.

**Confundir `z012` con un instante de tiempo.** Es una altura. El tiempo es el
sufijo.

**Reportar solo accuracy.** Con este desbalance no dice nada.

**Olvidar `shuffle=False` al evaluar.** Las probabilidades dejan de corresponder
con las filas.

**Normalizar distinto en la app que en el entrenamiento.** Se detecta en la
defensa.

## D3. Problemas técnicos

**`FileNotFoundError` al descomprimir en Windows, en ficheros que sí existen**
Es el límite de 260 caracteres de ruta. Las rutas del dataset son largas, así que
si descomprimes en una carpeta profunda (típico en OneDrive corporativo) la
extracción falla a medias. Usa una ruta corta como `C:\bdcedl`, o activa las rutas
largas como administrador y reinicia:

```powershell
Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
```

**El entrenamiento va lentísimo**
No entrenes leyendo desde OneDrive, Dropbox o Google Drive. Copia a disco local.

**`ModuleNotFoundError: No module named 'utils_caso'`**
Tu script no está en `breastdcedl/`:
```python
import sys; sys.path.insert(0, "/ruta/a/breastdcedl")
```

**`ValueError: Recibidas N probabilidades para M cortes`**
Usaste `shuffle=True` al evaluar, o no recorriste el `DataLoader` entero.

**Accuracy ~70 % y una columna vacía en la matriz de confusión**
El modelo predice siempre la clase mayoritaria. Ver [C3](#c3-desbalance-de-clases).

**Métricas de validación sospechosamente altas**
Casi siempre es fuga por paciente. Ver [C1](#c1-separación-por-paciente).

**La descarga se corta**
`curl` reanuda con `-C -`.

## D4. Diccionario de columnas

### `samples.csv`

Ver la tabla de [B3](#b3-samplescsv).

### `patients.csv`

**Identificación y objetivo**

| Campo | Significado |
|---|---|
| `pid` | Identificador de la paciente. Une con `samples.patient_id` |
| `pCR` | Etiqueta objetivo binaria |
| `split` | Conjunto público: `train` o `test` |
| `test` | Partición original: 0 train, 1 test, 2 validation |
| `dataset` | Cohorte: `spy1`, `spy2`, `duke` |

**Adquisición de la resonancia** — útiles para analizar sesgo entre cohortes

| Campo | Significado |
|---|---|
| `n_xy` | Tamaño original en el plano axial |
| `n_z` | Número de cortes del volumen |
| `n_times` | Adquisiciones temporales DCE disponibles |
| `pre`, `post_early`, `post_late` | Índices de las adquisiciones elegidas |
| `slice_thick` | Grosor del corte |
| `xy_spacing` | Separación de píxeles en el plano axial |

**Localización del tumor**

| Campo | Significado |
|---|---|
| `mask_start`, `mask_end` | Límites axiales de la región tumoral |
| `sraw`, `eraw` | Límites de fila del recorte |
| `scol`, `ecol` | Límites de columna del recorte |
| `tum_vol` | Volumen tumoral |

**Clínicas y demográficas**

| Campo | Significado |
|---|---|
| `age` | Edad |
| `menopause` | Estado menopáusico codificado por la fuente |
| `race_white`, `race_black` | Indicadores raciales binarios |
| `HR` | Estado de receptores hormonales |
| `HER2` | Estado del receptor HER2 |
| `HR_HER2_STATUS` | Subtipo combinado |
| `TripleNeg`, `HER2pos`, `HRposHER2neg` | Indicadores de subtipo |

> Los campos vacíos son datos **no disponibles** en la fuente, no ceros. El
> dataset no incluye peso, IMC, tabaquismo ni alcohol. Las variables raciales
> están porque vienen en la fuente; tratarlas como variables causales es un error
> metodológico, no una opción de diseño.

## D5. Preguntas para el informe

- ¿Qué error es más grave aquí, un falso positivo o un falso negativo? ¿Cambia la
  respuesta según el uso previsto?
- ¿La red aprende biología tumoral o diferencias entre cohortes?
- ¿Cómo afecta tener diez observaciones correlacionadas por paciente?
- ¿Está calibrada la probabilidad que devuelve tu modelo?
- ¿Qué evidencia faltaría para estudiar utilidad clínica, equidad y
  generalización externa?

Una reflexión honesta sobre las limitaciones vale más que una métrica alta sin
contexto. Con 176 pacientes en test, cualquier intervalo de confianza sobre la
sensibilidad va a ser ancho: dilo.

## D6. Licencia y ética

Imágenes desidentificadas derivadas de **BreastDCEDL**, licencia
**CC BY-NC 4.0**. Uso docente y de investigación exclusivamente.

No se permite reidentificación, diagnóstico ni recomendación terapéutica. Puedes
publicar tu trabajo en un repositorio personal conservando licencia y atribución,
pero **nunca la validación privada ni sus etiquetas**.

Un buen resultado retrospectivo no demuestra seguridad, causalidad ni utilidad
clínica prospectiva. **Esto no es un dispositivo médico.**
