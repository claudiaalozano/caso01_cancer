# Contexto del proyecto: CNN desde cero para predecir pCR (caso BreastDCEDL)

> Documento de traspaso. Resume todo lo trabajado hasta ahora para que otro asistente
> pueda continuar sin perder contexto. Autor del trabajo: Álvaro (4º Ingeniería
> Matemática, UAX), asignatura **Aprendizaje Automático**. Trabajo **individual**.

---

## 1. Cómo quiere trabajar Álvaro

- Parte de **poco conocimiento de redes neuronales** y quiere **entender cada decisión** para
  poder defenderla en clase. Prefiere ir **paso a paso**, con explicaciones conceptuales
  (sin código cuando pide "entender"), y una pregunta de comprobación al final de cada paso.
- Las **decisiones las toma él**; el código se implementa a partir de ellas.
- Sus profesores **no quieren notebooks como entrega**: el código final debe ser `.py`.
  Usa un notebook solo para visualizar datos (`src/eda.ipynb`).
- Trabaja en **Windows**, con **VS Code** y la terminal PowerShell, dentro de un entorno
  virtual `.venv`. **Su portátil no tiene GPU**: hace pruebas pequeñas en CPU y el
  entrenamiento real lo hará en un ordenador de la universidad con GPU.
- **No quiere usar Google Colab.**
- Tuvo que **desactivar "Control inteligente de aplicaciones"** de Windows (bloqueaba las DLL
  de pandas/matplotlib/torch). Ya está resuelto y Python funciona en local.

---

## 2. El enunciado (resumen)

**Reto:** construir una **CNN 2D desde cero** en PyTorch que, a partir de una resonancia
DCE-MRI tomada **antes** del tratamiento neoadyuvante, estime **P(pCR = 1)**, y convertirla en
una **aplicación web desplegada**.

- **pCR** (*pathological Complete Response*): tras el tratamiento no queda cáncer invasivo
  residual en mama ni ganglios (se confirma con anatomía patológica tras la cirugía).
  `y = 1` ⇔ pCR, `y = 0` ⇔ no pCR. Una probabilidad no es certeza ni recomendación clínica.
- **Prohibido:** ResNet, EfficientNet, VGG, DenseNet, pesos preentrenados, transfer learning.
- **Salida:** un único **logit**. Pérdida `BCEWithLogitsLoss`. Hay que **comparar pérdida
  normal y ponderada** con `pos_weight = N0/N1`.
- Justificar y guardar: arquitectura, inicialización, regularización, optimizador, lr,
  batch, épocas, semilla y criterio de parada. Si se usa GPU: dispositivo, versión del
  entorno, memoria máxima, tiempo por época, tiempo total, batch (opcional speedup CPU/GPU).
- **Datos:** la unidad estadística es la **paciente** (nunca mezclar cortes de una paciente
  entre particiones: data leakage). Test solo para métricas finales y matriz de confusión,
  **nunca** para ajustar. El umbral no se elige mirando test.
- **App web** (Streamlit/Gradio), desplegada con URL: cargar las 3 fases de un corte o elegir
  ejemplo; mostrar PRE, EARLY, LATE y mapa de realce; probabilidad y clase según umbral;
  umbral y versión/checksum del modelo; dispositivo y tiempo de respuesta; parámetros y
  métricas; mensajes claros ante archivos corruptos, no PNG 256×256, fases ausentes o
  duplicadas; aviso de uso educativo sin validez clínica. Sin rutas arbitrarias, sin
  ejecutar contenido subido, límite de tamaño, `model.eval()` y sin gradientes, resultado
  reproducible. Normalización idéntica a la del entrenamiento.
- **Defensa:** el profesor carga en la app 5 muestras de su **validación privada** (5 pacientes
  distintas) sin tocar código ni reiniciar; comprueba que la predicción coincide con el
  pipeline y que los errores se gestionan bien.
- **Entregables:** repositorio reproducible, pesos, configuración, URL de la app, informe y
  **exactamente 5 diapositivas**: (1) problema clínico y pCR; (2) datos, separación por
  paciente y clases; (3) arquitectura con tamaños y parámetros; (4) entrenamiento, decisiones
  y resultados; (5) app, demo, limitaciones y conclusiones.
- **Preguntas para discutir:** ¿qué error es más grave (FP/FN)?; ¿aprende biología o
  diferencias entre cohortes?; efecto de 10 cortes correlacionados por paciente;
  ¿probabilidad calibrada?; qué falta para utilidad clínica, equidad y generalización.
- Licencia de los datos **CC BY-NC 4.0**; no publicar la validación privada ni sus etiquetas.

---

## 3. El dataset BreastDCEDL (versión docente)

- Descargado del bucket público
  `https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl` con el
  script `descarga_datos.py`. Está en `data/breastdcedl/` dentro del repo (ignorado por Git).
- **1.273 pacientes, 12.703 cortes, 38.109 PNG.** Train: 1.097 pacientes (775 no pCR, 322 pCR;
  10.945 cortes: 7.729 / 3.216). Test: 176 pacientes (123 / 53; 1.758 cortes). La validación
  privada (177 pacientes) la tiene el profesor.
- Cada **corte** = 3 PNG en gris 8 bits de 256×256: `_PRE`, `_EARLY`, `_LATE`
  (antes del contraste, poscontraste temprano y tardío). Ruta:
  `dataset/<split>/<patient_id>/<sample_id>_<FASE>.png`. `z012`… indica la **altura** del corte,
  no el tiempo. ~10 cortes por paciente (7 pacientes tienen 5-8).
- Intensidades recortadas a percentiles 1-99 **con una ventana común a las 3 fases** y
  escaladas a 0-255: por eso EARLY − PRE representa el **realce** (la señal del problema).
  PRE < EARLY en el 100 % de las pacientes; LATE < EARLY (lavado/washout) en ~17 %.
- `metadata/samples.csv`: una fila por corte (`sample_id, patient_id, split, path_pre,
  path_early, path_late, slice_index, pCR, fold`). `fold` 0-4 en train (precalculado por
  paciente y estratificado), −1 en test.
- `metadata/patients.csv`: una fila por paciente con variables clínicas y técnicas
  (cohorte `dataset`, `HR_HER2_STATUS`, `age`, `tum_vol`, `menopause`, `n_times`,
  `slice_thick`, `xy_spacing`, etc.). **No son entradas de la red**, solo para interpretar.
- **Cohortes:** I-SPY1 (`spy1`, ensayo, `ISPY1_…`), I-SPY2 (`spy2`, ensayo adaptativo con
  fármacos nuevos, publicado como ACRIN 6698, `ACRIN-6698-…`), Duke (`duke`, serie
  retrospectiva de un hospital, `Breast_MRI_…`). Duke no tenía máscara 3D completa y se
  eligieron cortes centrales: posible sesgo.
- El profesor da `utils_caso.py` (cargar_samples, cargar_imagen, particion por fold,
  BreastDCEDataset, agregar/evaluar_por_paciente, pos_weight), `ver_muestras.py` (visor) y
  `GUIA.md`. Están en `data/breastdcedl/` y **deben quedarse ahí** (buscan los datos en su
  propia carpeta).

---

## 4. Hallazgos del EDA

- **Desbalance ≈ 70/30** (29 % pCR en train, 30 % en test). Es la tasa real de pCR, no un
  defecto. Predecir siempre "no pCR" da **69,9 % de accuracy** en test → la accuracy no sirve
  como métrica principal.
- `pos_weight` = N0/N1 en cortes de train = 7.729/3.216 = **2,40** (con fold 0 de validación:
  6.183/2.576 = 2,40).
- Separación correcta: ninguna paciente en dos splits ni en dos folds.
- **Los folds no están tan equilibrados** como dice la guía: pCR por fold (paciente)
  29,2 %, 28,3 %, **26,8 %**, **34,1 %**, 28,3 %.
- **Subtipo** (train): pCR 43 % en HER2+, 38 % en triple negativo, **14 % en HR+/HER2−**.
- **Cohortes** (train): Duke 21 % pCR (209 pac.), I-SPY1 25 % (104), I-SPY2 32 % (784).
  Duke tiene 51 % de HR+/HER2− → parte de su menor pCR es por composición clínica.
- Protocolos de adquisición muy distintos entre cohortes (nº de tiempos DCE, grosor de
  corte, resolución original) → riesgo de que la red aprenda la cohorte en vez de biología.
- Edad y volumen tumoral apenas difieren entre clases.
- Faltantes: `menopause` falta en el 100 % de I-SPY1 (13,5 % global); subtipo falta en 5
  pacientes de I-SPY1; resto < 0,5 %. **No se excluye ninguna paciente** (pCR e imágenes
  completas; las 2 problemáticas ya las excluyeron los autores).
- Pendiente: Álvaro debe ejecutar el notebook completo, guardar con resultados y rellenar
  las conclusiones de la sección 11 (realce medio por clase, % de lavado, aspecto de
  cohortes).

---

## 5. Estructura del repositorio

Repo: `https://github.com/claudiaalozano/caso01_cancer.git`, local en `C:\Dev\caso01_cancer`.

```
caso01_cancer/
├── .gitignore            ← ignora data/, .venv, etc.
├── README.md
├── requirements.txt      ← numpy, pandas, pillow, matplotlib, scikit-learn, requests, torch
├── descarga_datos.py     ← descarga el dataset
├── data/breastdcedl/     ← dataset + GUIA.md, LICENSE, utils_caso.py, ver_muestras.py (no en Git)
├── data/cache/cortes_uint8.npy   ← caché (no en Git)
└── src/
    ├── eda.ipynb         ← exploración visual (secciones 0-11, incl. 1b calidad de variables)
    ├── preprocessing.py  ← carga, validación, aumento, DataLoaders, caché
    ├── model.py          ← la CNN + verificaciones
    └── train.py          ← entrenamiento + evaluación final opcional en test
```

Estructura final prevista: añadir `configs/`, `models/` (pesos), `resultados/`, `app/`
(Streamlit/Gradio), `informe/` (informe + 5 diapositivas).

---

## 6. Preprocesado (`src/preprocessing.py`) — HECHO y verificado

- **Única normalización: PNG / 255**, apilado en orden **PRE, EARLY, LATE** → tensor
  `(3, 256, 256)` en [0, 1]. Idéntico a `utils_caso.cargar_imagen` (verificado).
  No se normaliza cada fase por separado (destruiría el realce), ni ImageNet.
- `validar_subida()`: para la app; rechaza no-PNG, no gris, no 256×256, >1 MB, corruptos,
  fases ausentes o duplicadas (fase deducida del sufijo del nombre).
- `AumentoGeometrico` (solo train): volteo horizontal p=0,5, rotación ±10°, traslación ±8 px,
  **misma transformación para las 3 fases** (verificado: error de alineación del realce
  ≈ 1e-7). Sin volteo vertical ni cambios de brillo/contraste.
- `crear_dataloaders(fold_val=0, batch_size=32, max_pacientes=None, ...)`: separación por
  paciente con `uc.particion` (fold 0 validación, 1-4 entrenamiento), `pos_weight` calculado
  solo con la parte de entrenamiento, train con shuffle+aumento, validación sin ninguno,
  semilla fija. `crear_dataloader_test()` aparte.
- **Caché** (`--construir-cache`): todos los cortes en `data/cache/cortes_uint8.npy`
  (uint8, ~2,5 GB, memmap). Solo acelera; resultado idéntico. Ya creada en el portátil;
  hay que recrearla en el ordenador de la universidad.

---

## 7. Arquitectura (`src/model.py`) — HECHA y verificada

Diseño siguiendo el **método de 6 pasos del profesor** (diapositivas *CNN II*):

1. **Entrada/salida:** `3 × 256 × 256` (3 fases como canales; no son colores) → **1 logit**.
   Probabilidad = sigmoide(logit), aplicada fuera de la red (la pérdida la incluye).
2. **Etapas:** **6**, porque 256 → 128 → 64 → 32 → 16 → 8 → 4 (regla: dividir entre 2 hasta
   ~4×4). Campo receptivo final ≈ 190×190 px (con 4 etapas serían solo ~76 px).
   El nº de etapas depende del tamaño de la imagen, **no** del hardware.
3. **Bloque:** `Conv2d 3×3 (padding=1, bias=False) → BatchNorm2d → ReLU → MaxPool2d 2×2`.
   Canales **16 → 32 → 64 → 128 → 128 → 128** (duplicar al reducir resolución, con tope en 128
   para no pasar de ~400 K parámetros con ~880 pacientes de entrenamiento; la regla
   estricta 32→512 daría 3,9 M).
4. **Reducción:** MaxPool 2×2 (sin parámetros, conserva lo más fuerte, tolera desplazamientos).
5. **Cabeza:** Global Average Pooling → Dropout(0,3) → `Linear(128, 1)`.
6. **Verificación:** (A) tensor falso `[2,3,256,256]` → `[2,1]`; (B) sobreajustar 10 cortes
   reales (pérdida 0,69 → ~0,002, 10/10 aciertos).

| Capa | Salida | Parámetros |
|---|---|---|
| Etapa 1 | 16×128×128 | 464 |
| Etapa 2 | 32×64×64 | 4.672 |
| Etapa 3 | 64×32×32 | 18.560 |
| Etapa 4 | 128×16×16 | 73.984 |
| Etapa 5 | 128×8×8 | 147.712 |
| Etapa 6 | 128×4×4 | 147.712 |
| GAP + Dropout | 128 | 0 |
| Linear | 1 | 129 |
| **Total** | | **393.233** |

Fórmula: conv sin sesgo = 3·3·C_in·C_out; BatchNorm = 2·C_out (γ, β). Inicialización
**Kaiming** (He) para ReLU. Sin preentrenamiento. La mayor parte del **cálculo** está en las
etapas 1-4 (imágenes grandes); la mayor parte de los **parámetros**, en las etapas 5-6.

---

## 8. Entrenamiento (`src/train.py`) — HECHO, probado solo con datos de prueba

- Uso: `python src/train.py --rapido` (prueba en CPU: 10 pacientes/clase en train, 5 en
  validación). Real: `python src/train.py` (en GPU). Añadir `--test` para evaluar al final
  en test **una sola vez**, con la mejor época y el umbral elegidos en validación.
- Por defecto: **10 épocas máx.**, early stopping con **paciencia 5** sobre el **AUC por
  paciente** en validación (media de las probabilidades de sus cortes), guarda la **mejor
  época**. AdamW, lr 1e-3, weight decay 1e-4, batch 32, dropout 0,3, semilla 42, fold 0.
  `--perdida ponderada|normal` (obligatorio comparar ambas).
- Umbral: se elige en validación con **Youden** (máx. sensibilidad + especificidad − 1).
- Guarda `models/<nombre>.pt` y en `resultados/entrenamiento/<nombre>/`: `config.json`
  (hiperparámetros, entorno, tiempos, memoria GPU, métricas, SHA-256 de los pesos),
  `historial.csv`, `curvas.png`, `val_pacientes.csv`, y con `--test`
  `matriz_confusion_test.png` y `test_pacientes.csv`.
- Consejo dado: comparar experimentos **sin** `--test`; usar `--test` solo en el modelo final.
- Estimación en CPU: `--rapido` ~15-20 s/época; datos completos ~10-20 min/época (→ GPU).

---

## 9. Dónde nos quedamos (explicación paso a paso de la arquitectura)

Álvaro pidió entender la arquitectura **paso a paso**. Ya se han explicado los pasos 1-4:

- **Paso 1 (entrada/salida):** respondió que juntar fases permite ver cómo "se colorean"
  zonas. Se le corrigieron dos matices: lo que ilumina es el **contraste** (las 3 fases son
  de la misma resonancia, antes del tratamiento), y no se predice si tiene cáncer (todas lo
  tienen) sino **si responderá con pCR**.
- **Paso 2 (etapas):** confundió nº de etapas con capacidad del hardware; se aclaró que
  depende del tamaño de la imagen (128×128 → 5 etapas) y que CPU y GPU usan **la misma red**
  (en CPU solo se reducen datos/épocas).
- **Paso 3 (bloque):** se explicó por qué sin ReLU la red colapsaría en una función lineal
  (composición de funciones lineales es lineal; matiz: MaxPool aporta algo de no linealidad).
- **Paso 4 (reducción):** MaxPool frente a convolución con stride 2.
- **Pregunta pendiente para Álvaro:** el cuadradito 2×2 `[[0,1; 0,7], [0,3; 0,2]]` → ¿en qué
  número se convierte con MaxPool? ¿Y con average pooling? (Respuesta: 0,7 y 0,325.)
- **Siguiente:** **Paso 5 (la cabeza:** GAP frente a Flatten, dropout, logit, sigmoide) y
  **Paso 6 (verificación)**. Después: explicar el entrenamiento (pérdida, pos_weight,
  optimizador, épocas, early stopping, umbral).

---

## 10. Próximos pasos del trabajo

1. Terminar la explicación de la arquitectura (pasos 5 y 6).
2. Ejecutar en el portátil `python src/model.py` y `python src/train.py --rapido`.
3. En la GPU de la universidad: descargar datos, crear caché y entrenar de verdad;
   comparar **pérdida normal vs ponderada** (y opcionalmente dropout, aumento, canales).
4. Elegir modelo final con validación (opcional: validación cruzada en los 5 folds), y solo
   entonces evaluar en test (`--test`). Analizar resultados por cohorte y por subtipo.
5. Construir y desplegar la app (Streamlit/Gradio) usando `preprocessing.validar_subida` y
   `model.CNNpCR`; mismos pesos y umbral; checksum visible.
6. Informe y 5 diapositivas; respuestas a las preguntas de discusión.

Material de clase disponible: diapositivas del profesor *CNN I* (fundamentos: convolución,
padding/stride, parámetros, campo receptivo, pooling, activaciones, BatchNorm, dropout),
*CNN II* (método de 6 pasos, cabeza GAP, verificación, entrenamiento, augmentation) y
*CNN III* (LeNet, AlexNet, VGG).
