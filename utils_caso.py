"""Utilidades del caso BreastDCEDL.

Resuelven las dos partes del problema que son fáciles de hacer mal y que no
forman parte de lo que se evalúa: la separación por paciente y el paso de
predicciones por corte a predicción por paciente.

Lo que SÍ tienes que construir tú: la arquitectura de la CNN, el bucle de
entrenamiento, la regularización, el umbral de decisión y su justificación.

Uso típico desde la carpeta `breastdcedl/`:

    import utils_caso as uc

    samples = uc.cargar_samples()
    tr, va = uc.particion(samples, fold_val=0)

    ds_tr = uc.BreastDCEDataset(tr, uc.RAIZ)
    ds_va = uc.BreastDCEDataset(va, uc.RAIZ)
    ...
    resultados = uc.evaluar_por_paciente(probabilidades, va, umbral=0.5)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent

FASES = ("PRE", "EARLY", "LATE")
METODOS_AGREGACION = ("mean", "max", "median", "voto")


# --------------------------------------------------------------------------- #
# Carga de metadatos
# --------------------------------------------------------------------------- #

def cargar_samples(raiz: Path | str = RAIZ) -> pd.DataFrame:
    """Índice del dataset: una fila por corte."""
    return pd.read_csv(Path(raiz) / "metadata" / "samples.csv")


def cargar_patients(raiz: Path | str = RAIZ) -> pd.DataFrame:
    """Metadatos clínicos: una fila por paciente."""
    return pd.read_csv(Path(raiz) / "metadata" / "patients.csv")


def cargar_fase(ruta_relativa: str, raiz: Path | str = RAIZ) -> np.ndarray:
    """Carga UNA fase: devuelve (256, 256) en float32, valores en [0, 1].

    Cada fase DCE es un PNG en escala de grises independiente, con la fase
    escrita en el nombre del fichero:

        ISPY1_1001_z016_PRE.png      antes del contraste
        ISPY1_1001_z016_EARLY.png    postcontraste temprano
        ISPY1_1001_z016_LATE.png     postcontraste tardio

    Pillow los entrega como enteros de 0 a 255; dividir entre 255 los lleva al
    rango que espera la red.
    """
    with Image.open(Path(raiz) / ruta_relativa) as png:
        return np.asarray(png.convert("L"), dtype=np.float32) / 255.0


def cargar_imagen(fila, raiz: Path | str = RAIZ) -> np.ndarray:
    """Devuelve el array (3, 256, 256) en float32 en [0,1], apilando las 3 fases.

    `fila` es una fila de `samples.csv` (la que devuelve `.itertuples()` o
    `.iloc[i]`). De ella se leen las tres rutas, una por fase.

    El orden del apilado es siempre PRE, EARLY, LATE, y ese es el orden de los
    canales que recibe la CNN. Puedes comprobarlo tu mismo: al inyectar el
    contraste el tejido se ilumina, asi que la media de EARLY supera a la de PRE
    (medido: en el 100 % de las pacientes).

        x = uc.cargar_imagen(fila)
        tejido = x[0] > 0.1                 # descartar el aire del fondo
        pre, early, late = [float(c[tejido].mean()) for c in x]
        assert pre < early

    Que LATE sea a veces menor que EARLY no es un error: es el lavado del
    contraste (washout), real en el 17 % de las pacientes.
    """
    rutas = [fila.path_pre, fila.path_early, fila.path_late]
    return np.stack([cargar_fase(ruta, raiz) for ruta in rutas])   # (3, 256, 256)


# --------------------------------------------------------------------------- #
# Separación por paciente
# --------------------------------------------------------------------------- #

def particion(samples: pd.DataFrame, fold_val: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (entrenamiento, validación) usando la columna `fold`.

    Los pliegues vienen calculados por paciente y estratificados por pCR, así que
    ninguna paciente puede aparecer en los dos lados. `fold_val` elige cuál de los
    cinco pliegues (0 a 4) se reserva para validar.

    El conjunto `test` queda fuera de ambos: se usa una sola vez, al final, para
    la evaluación que entregas. Nunca para ajustar el modelo.
    """
    if fold_val not in range(5):
        raise ValueError(f"fold_val debe estar entre 0 y 4, recibido {fold_val}")

    train = samples[samples.split == "train"]
    entrenamiento = train[train.fold != fold_val]
    validacion = train[train.fold == fold_val]

    solapan = set(entrenamiento.patient_id) & set(validacion.patient_id)
    if solapan:                                   # no debería ocurrir nunca
        raise RuntimeError(f"Fuga por paciente: {len(solapan)} pacientes en ambos lados")

    return entrenamiento, validacion


def conjunto_test(samples: pd.DataFrame) -> pd.DataFrame:
    """El conjunto de prueba. Úsalo una sola vez, al final."""
    return samples[samples.split == "test"]


# --------------------------------------------------------------------------- #
# Dataset de PyTorch
# --------------------------------------------------------------------------- #

try:
    import torch
    from torch.utils.data import Dataset

    class BreastDCEDataset(Dataset):
        """Devuelve (imagen[3,256,256] float32, etiqueta float32).

        `transform` recibe el tensor completo de 3 canales. Cualquier
        transformación geométrica debe aplicarse a los tres a la vez: si rotas o
        volteas un canal por separado, desalineas las fases DCE y destruyes la
        información de realce, que es justo la señal del problema.
        """

        def __init__(self, filas: pd.DataFrame, raiz: Path | str = RAIZ, transform=None):
            self.filas = filas.reset_index(drop=True)
            self.raiz = Path(raiz)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.filas)

        def __getitem__(self, i: int):
            fila = self.filas.iloc[i]
            x = torch.from_numpy(cargar_imagen(fila, self.raiz))
            if self.transform is not None:
                x = self.transform(x)
            y = torch.tensor(float(fila.pCR), dtype=torch.float32)
            return x, y

except ImportError:                               # PyTorch no instalado
    BreastDCEDataset = None


# --------------------------------------------------------------------------- #
# De predicción por corte a predicción por paciente
# --------------------------------------------------------------------------- #

def agregar_por_paciente(
    patient_id, probabilidad, metodo: str = "mean", umbral: float = 0.5
) -> pd.DataFrame:
    """Combina las probabilidades de los cortes de cada paciente en una sola.

    El modelo puntúa cortes, pero pCR es una propiedad de la paciente: hay que
    agregar. Métodos disponibles:

      mean   media de las probabilidades. Robusto, es el punto de partida sensato.
      max    la más alta. Sensible: basta un corte sospechoso. Sube la tasa de
             falsos positivos.
      median mediana. Como la media pero más resistente a un corte atípico.
      voto   fracción de cortes que superan `umbral` (voto mayoritario).

    Devuelve un DataFrame indexado por paciente con la columna `prob`.
    El método que elijas y por qué es una decisión tuya que debes justificar.
    """
    if metodo not in METODOS_AGREGACION:
        raise ValueError(f"metodo debe ser uno de {METODOS_AGREGACION}, recibido {metodo!r}")

    datos = pd.DataFrame({
        "patient_id": np.asarray(patient_id),
        "prob": np.asarray(probabilidad, dtype=float),
    })

    if metodo == "voto":
        agregado = datos.assign(voto=(datos.prob >= umbral)).groupby("patient_id").voto.mean()
    else:
        agregado = getattr(datos.groupby("patient_id").prob, metodo)()

    return agregado.rename("prob").to_frame()


def evaluar_por_paciente(
    probabilidad, filas: pd.DataFrame, umbral: float = 0.5, metodo: str = "mean"
) -> dict:
    """Métricas a nivel de paciente, que es la unidad que cuenta.

    `probabilidad` es la salida del modelo para cada corte de `filas`, en el mismo
    orden. `filas` es el DataFrame que alimentó el DataLoader (con shuffle=False).

    Devuelve un diccionario con la matriz de confusión y las métricas. Mira la
    sensibilidad, no solo la accuracy: con este desbalance, predecir siempre
    pCR=0 ya acierta el 70 % de los casos.
    """
    probabilidad = np.asarray(probabilidad, dtype=float)
    if len(probabilidad) != len(filas):
        raise ValueError(
            f"Recibidas {len(probabilidad)} probabilidades para {len(filas)} cortes. "
            "¿Usaste shuffle=False en el DataLoader de evaluación?"
        )

    agregado = agregar_por_paciente(filas.patient_id.values, probabilidad, metodo, umbral)
    verdad = filas.groupby("patient_id").pCR.first()

    tabla = agregado.join(verdad)
    y_real = tabla.pCR.values
    y_pred = (tabla.prob.values >= umbral).astype(int)

    vp = int(((y_pred == 1) & (y_real == 1)).sum())
    vn = int(((y_pred == 0) & (y_real == 0)).sum())
    fp = int(((y_pred == 1) & (y_real == 0)).sum())
    fn = int(((y_pred == 0) & (y_real == 1)).sum())

    def division(a, b):
        return a / b if b else float("nan")

    resultado = {
        "pacientes": len(tabla),
        "metodo": metodo,
        "umbral": umbral,
        "matriz_confusion": {"VP": vp, "VN": vn, "FP": fp, "FN": fn},
        "accuracy": division(vp + vn, vp + vn + fp + fn),
        "sensibilidad": division(vp, vp + fn),      # recall de la clase pCR=1
        "especificidad": division(vn, vn + fp),
        "precision": division(vp, vp + fp),
    }

    try:                                            # el AUC no depende del umbral
        from sklearn.metrics import roc_auc_score
        resultado["auc"] = float(roc_auc_score(y_real, tabla.prob.values))
    except Exception:
        resultado["auc"] = float("nan")

    return resultado


def pos_weight(filas: pd.DataFrame) -> float:
    """N0/N1 para BCEWithLogitsLoss, calculado sobre las filas que le pases."""
    n1 = int((filas.pCR == 1).sum())
    if n1 == 0:
        raise ValueError("No hay muestras positivas en este subconjunto")
    return float((filas.pCR == 0).sum()) / n1
