#!/usr/bin/env python3
"""Preprocesado de BreastDCEDL.

Incluye:
- Carga de PRE, EARLY y LATE.
- Diagnóstico y registro opcional de traslaciones pequeñas.
- Recorte común del fondo negro externo.
- Normalización PNG / 255.
- Aumento geométrico conjunto.
- Dataset, caché y DataLoaders.

Registro y recorte están desactivados por defecto.
El registro es aproximado y requiere revisión visual.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------
# Rutas y constantes
# ---------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RAIZ_DATOS = REPO / "data" / "breastdcedl"
RUTA_CACHE = REPO / "data" / "cache" / "cortes_uint8.npy"

FASES = ("PRE", "EARLY", "LATE")
TAMANO = 256
ESCALA = 255.0
MAX_BYTES_PNG = 1_000_000

sys.path.insert(0, str(RAIZ_DATOS))

try:
    import utils_caso as uc
except ModuleNotFoundError as error:
    if error.name != "utils_caso":
        raise
    uc = None


class ErrorEntrada(ValueError):
    """Error de validación de imágenes."""


def _requiere_utils():
    if uc is None:
        raise RuntimeError(
            f"No encuentro utils_caso.py en {RAIZ_DATOS}"
        )


# ---------------------------------------------------------------------
# Configuración espacial
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigEspacial:
    registrar: bool = False
    recortar: bool = False
    max_desplazamiento: int = 8
    mejora_minima: float = 0.02
    correlacion_minima: float = 0.4
    margen: int = 8

    def __post_init__(self):
        if type(self.registrar) is not bool:
            raise ValueError("registrar debe ser booleano")

        if type(self.recortar) is not bool:
            raise ValueError("recortar debe ser booleano")

        for nombre in ("max_desplazamiento", "margen"):
            valor = getattr(self, nombre)
            if type(valor) is not int or not 1 <= valor <= 32:
                raise ValueError(
                    f"{nombre} debe ser un entero entre 1 y 32"
                )

        if not 0 <= self.mejora_minima <= 1:
            raise ValueError("mejora_minima debe estar entre 0 y 1")

        if not 0 <= self.correlacion_minima <= 1:
            raise ValueError(
                "correlacion_minima debe estar entre 0 y 1"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def desde_json(cls, ruta: str | Path):
        datos = json.loads(
            Path(ruta).read_text(encoding="utf-8")
        )
        return cls(**datos)


# ---------------------------------------------------------------------
# Diagnóstico de alineación
# ---------------------------------------------------------------------

def _validar_fases(fases: np.ndarray) -> np.ndarray:
    fases = np.asarray(fases)

    if fases.shape != (3, TAMANO, TAMANO):
        raise ErrorEntrada(
            "Se esperaba una matriz (3, 256, 256)"
        )

    if fases.dtype != np.uint8:
        raise ErrorEntrada("Las fases deben ser uint8")

    return fases


def _bordes(imagen: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    suavizada = gaussian_filter(
        imagen.astype(np.float64) / ESCALA,
        sigma=1,
    )
    gradiente_y, gradiente_x = np.gradient(suavizada)

    return np.hypot(gradiente_y, gradiente_x)


def _mover(
    imagen: np.ndarray,
    desplazamiento_y: int,
    desplazamiento_x: int,
) -> np.ndarray:
    """Traslada píxeles enteros y rellena con cero, sin interpolar."""

    alto, ancho = imagen.shape
    salida = np.zeros_like(imagen)

    origen_y = slice(
        max(0, -desplazamiento_y),
        min(alto, alto - desplazamiento_y),
    )
    origen_x = slice(
        max(0, -desplazamiento_x),
        min(ancho, ancho - desplazamiento_x),
    )

    destino_y = slice(
        max(0, desplazamiento_y),
        min(alto, alto + desplazamiento_y),
    )
    destino_x = slice(
        max(0, desplazamiento_x),
        min(ancho, ancho + desplazamiento_x),
    )

    salida[destino_y, destino_x] = imagen[origen_y, origen_x]
    return salida


def _estimar_desplazamiento(
    referencia: np.ndarray,
    movil: np.ndarray,
    config: ConfigEspacial,
) -> dict:
    """Propone una traslación mediante correlación de gradientes.

    No certifica correspondencia anatómica.
    No corrige rotaciones, deformaciones ni movimiento fuera del plano.
    """

    from scipy.signal import correlate

    a = _bordes(referencia)
    b = _bordes(movil)

    resultado = {
        "desplazamiento_yx": [0, 0],
        "corr_antes": None,
        "corr_candidata": None,
        "aplicado": False,
        "motivo": "textura insuficiente",
    }

    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return resultado

    unos = np.ones_like(a)

    def correlacion(x, y):
        return correlate(
            x,
            y,
            mode="full",
            method="fft",
        )

    n = correlacion(unos, unos)
    suma_a = correlacion(a, unos)
    suma_b = correlacion(unos, b)

    covarianza = (
        correlacion(a, b)
        - suma_a * suma_b / np.maximum(n, 1)
    )

    varianza_a = np.maximum(
        correlacion(a * a, unos)
        - suma_a**2 / np.maximum(n, 1),
        0,
    )
    varianza_b = np.maximum(
        correlacion(unos, b * b)
        - suma_b**2 / np.maximum(n, 1),
        0,
    )

    puntuaciones = covarianza / np.maximum(
        np.sqrt(varianza_a * varianza_b),
        1e-12,
    )

    centro_y, centro_x = np.array(a.shape) - 1
    limite = config.max_desplazamiento

    region = puntuaciones[
        centro_y - limite:centro_y + limite + 1,
        centro_x - limite:centro_x + limite + 1,
    ]

    indice_y, indice_x = np.unravel_index(
        np.argmax(region),
        region.shape,
    )

    dy = int(indice_y - limite)
    dx = int(indice_x - limite)

    antes = float(
        np.clip(puntuaciones[centro_y, centro_x], -1, 1)
    )
    despues = float(
        np.clip(region[indice_y, indice_x], -1, 1)
    )

    resultado.update({
        "desplazamiento_yx": [dy, dx],
        "corr_antes": antes,
        "corr_candidata": despues,
        "motivo": "sin mejora suficiente",
    })

    if dy == 0 and dx == 0:
        resultado["motivo"] = "mejor candidato sin desplazamiento"
        return resultado

    if abs(dy) == limite or abs(dx) == limite:
        resultado["motivo"] = (
            "candidato en el límite; revisar manualmente"
        )
        return resultado

    if despues < config.correlacion_minima:
        resultado["motivo"] = (
            "correlación insuficiente; revisar manualmente"
        )
        return resultado

    if despues - antes < config.mejora_minima:
        return resultado

    candidato = _mover(movil, dy, dx)

    if np.count_nonzero(candidato) != np.count_nonzero(movil):
        resultado["motivo"] = (
            "se perdería tejido en el borde; revisar manualmente"
        )
        return resultado

    resultado["motivo"] = "candidato aceptable para revisión"
    resultado["aplicado"] = config.registrar

    return resultado


# ---------------------------------------------------------------------
# Recorte común y procesamiento espacial
# ---------------------------------------------------------------------

def procesar_espacial(
    fases: np.ndarray,
    config: ConfigEspacial | None = None,
    diagnostico: bool = False,
) -> tuple[np.ndarray, dict]:
    fases = _validar_fases(fases)
    config = config or ConfigEspacial()

    informe = {
        "config": config.to_dict(),
        "registro": {},
        "recorte": None,
        "advertencia": (
            "Diagnóstico heurístico; no garantiza "
            "correspondencia anatómica."
        ),
    }

    if not (config.registrar or config.recortar or diagnostico):
        return fases, informe

    salida = fases.copy()

    if config.registrar or diagnostico:
        for indice, nombre in ((1, "EARLY"), (2, "LATE")):
            resultado = _estimar_desplazamiento(
                fases[0],
                fases[indice],
                config,
            )

            informe["registro"][nombre] = resultado

            if resultado["aplicado"]:
                salida[indice] = _mover(
                    fases[indice],
                    *resultado["desplazamiento_yx"],
                )

    if config.recortar:
        # Unión de todos los píxeles no negros de las tres fases.
        # No se elimina tejido oscuro mediante un umbral arbitrario.
        mascara = np.any(salida > 0, axis=0)
        filas, columnas = np.where(mascara)

        if len(filas):
            y0 = int(filas.min())
            y1 = int(filas.max()) + 1
            x0 = int(columnas.min())
            x1 = int(columnas.max()) + 1

            lado = max(y1 - y0, x1 - x0) + 2 * config.margen

            if lado < TAMANO:
                superior = (y0 + y1 - lado) // 2
                izquierda = (x0 + x1 - lado) // 2

                superior = max(
                    0, min(superior, TAMANO - lado)
                )
                izquierda = max(
                    0, min(izquierda, TAMANO - lado)
                )

                recortes = []

                for imagen in salida:
                    recorte = imagen[
                        superior:superior + lado,
                        izquierda:izquierda + lado,
                    ]

                    redimensionada = Image.fromarray(
                        recorte
                    ).resize(
                        (TAMANO, TAMANO),
                        Image.Resampling.BILINEAR,
                    )

                    recortes.append(np.asarray(redimensionada))

                salida = np.stack(recortes)

                informe["recorte"] = {
                    "top": superior,
                    "left": izquierda,
                    "lado": lado,
                    "salida": [TAMANO, TAMANO],
                    "escala": TAMANO / lado,
                }

    return salida, informe


# ---------------------------------------------------------------------
# Lectura y normalización
# ---------------------------------------------------------------------

def _abrir_png(
    fuente: str | Path | bytes,
    nombre: str,
) -> np.ndarray:
    try:
        if isinstance(fuente, (bytes, bytearray)):
            if len(fuente) > MAX_BYTES_PNG:
                raise ErrorEntrada(
                    f"{nombre}: supera "
                    f"{MAX_BYTES_PNG // 1000} KB"
                )

            fuente = io.BytesIO(fuente)

        with Image.open(fuente) as imagen:
            if imagen.format != "PNG":
                raise ErrorEntrada(
                    f"{nombre}: debe ser un PNG"
                )

            if imagen.mode != "L":
                raise ErrorEntrada(
                    f"{nombre}: debe ser gris de 8 bits, modo L"
                )

            if imagen.size != (TAMANO, TAMANO):
                raise ErrorEntrada(
                    f"{nombre}: debe medir {TAMANO}x{TAMANO}"
                )

            imagen.load()
            return np.array(imagen, dtype=np.uint8)

    except ErrorEntrada:
        raise
    except FileNotFoundError as error:
        raise ErrorEntrada(
            f"{nombre}: no existe el fichero"
        ) from error
    except Exception as error:
        raise ErrorEntrada(
            f"{nombre}: fichero corrupto o ilegible "
            f"({type(error).__name__})"
        ) from error


def a_tensor(
    fases_uint8: np.ndarray,
    config_espacial: ConfigEspacial | None = None,
) -> torch.Tensor:
    fases, _ = procesar_espacial(
        fases_uint8,
        config_espacial,
    )

    x = torch.from_numpy(
        np.array(fases, dtype=np.uint8)
    ).float() / ESCALA

    if not torch.isfinite(x).all():
        raise ErrorEntrada("La imagen contiene valores no finitos")

    return x


def cargar_corte(
    pre,
    early,
    late,
    config_espacial: ConfigEspacial | None = None,
) -> torch.Tensor:
    fases = np.stack([
        _abrir_png(fuente, nombre)
        for fuente, nombre in zip(
            (pre, early, late),
            FASES,
        )
    ])

    return a_tensor(fases, config_espacial)


def cargar_corte_uint8(
    fila,
    raiz: Path = RAIZ_DATOS,
) -> np.ndarray:
    rutas = (
        fila.path_pre,
        fila.path_early,
        fila.path_late,
    )

    return np.stack([
        _abrir_png(raiz / ruta, fase)
        for ruta, fase in zip(rutas, FASES)
    ])


# ---------------------------------------------------------------------
# Validación de subidas
# ---------------------------------------------------------------------

def validar_subida(
    ficheros: dict[str, bytes],
    config_espacial: ConfigEspacial | None = None,
) -> torch.Tensor:
    por_fase = {fase: [] for fase in FASES}
    identificadores = []

    for nombre in ficheros:
        base = Path(nombre).name

        if not base.upper().endswith(".PNG"):
            raise ErrorEntrada(
                f"{nombre}: solo se aceptan archivos PNG"
            )

        encontrada = False

        for fase in FASES:
            sufijo = f"_{fase}.PNG"

            if base.upper().endswith(sufijo):
                por_fase[fase].append(nombre)
                identificadores.append(base[:-len(sufijo)])
                encontrada = True
                break

        if not encontrada:
            raise ErrorEntrada(
                f"{nombre}: falta el sufijo PRE, EARLY o LATE"
            )

    faltan = [
        fase for fase, nombres in por_fase.items()
        if not nombres
    ]
    duplicadas = [
        fase for fase, nombres in por_fase.items()
        if len(nombres) > 1
    ]

    if faltan:
        raise ErrorEntrada(
            f"Faltan fases: {', '.join(faltan)}"
        )

    if duplicadas:
        raise ErrorEntrada(
            f"Fases duplicadas: {', '.join(duplicadas)}"
        )

    if len(ficheros) != 3:
        raise ErrorEntrada("Hay que subir exactamente tres archivos")

    if len(set(identificadores)) != 1:
        raise ErrorEntrada(
            "Las tres fases deben pertenecer al mismo corte"
        )

    return cargar_corte(
        *(ficheros[por_fase[fase][0]] for fase in FASES),
        config_espacial=config_espacial,
    )


# ---------------------------------------------------------------------
# Aumento: después del procesamiento espacial, solo en train
# ---------------------------------------------------------------------

class AumentoGeometrico:
    def __init__(
        self,
        p_volteo: float = 0.5,
        max_grados: float = 10.0,
        max_desplazamiento: int = 8,
    ):
        self.p_volteo = p_volteo
        self.max_grados = max_grados
        self.max_desplazamiento = max_desplazamiento

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p_volteo:
            x = torch.flip(x, dims=[2])

        angulo = math.radians(
            (torch.rand(1).item() * 2 - 1)
            * self.max_grados
        )

        _, alto, ancho = x.shape

        dx = (
            (torch.rand(1).item() * 2 - 1)
            * self.max_desplazamiento * 2 / ancho
        )
        dy = (
            (torch.rand(1).item() * 2 - 1)
            * self.max_desplazamiento * 2 / alto
        )

        coseno = math.cos(angulo)
        seno = math.sin(angulo)

        theta = torch.tensor(
            [[coseno, -seno, dx], [seno, coseno, dy]],
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0)

        malla = F.affine_grid(
            theta,
            [1, *x.shape],
            align_corners=False,
        )

        return F.grid_sample(
            x.unsqueeze(0),
            malla,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)

    def __repr__(self):
        return (
            f"AumentoGeometrico(p_volteo={self.p_volteo}, "
            f"max_grados={self.max_grados}, "
            f"max_desplazamiento={self.max_desplazamiento})"
        )


# ---------------------------------------------------------------------
# Dataset y caché de imágenes originales
# ---------------------------------------------------------------------

class DatasetCortes(Dataset):
    def __init__(
        self,
        filas: pd.DataFrame,
        transform=None,
        cache: np.ndarray | None = None,
        indice_cache: dict[str, int] | None = None,
        raiz: Path = RAIZ_DATOS,
        config_espacial: ConfigEspacial | None = None,
    ):
        self.filas = filas.reset_index(drop=True)
        self.transform = transform
        self.cache = cache
        self.indice_cache = indice_cache
        self.raiz = raiz
        self.config_espacial = config_espacial

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, indice):
        fila = self.filas.iloc[indice]

        if self.cache is None:
            fases = cargar_corte_uint8(fila, self.raiz)
        else:
            fases = self.cache[
                self.indice_cache[fila.sample_id]
            ]

        x = a_tensor(fases, self.config_espacial)

        if self.transform is not None:
            x = self.transform(x)

        y = torch.tensor(
            float(fila.pCR),
            dtype=torch.float32,
        )

        return x, y


def construir_cache(
    samples: pd.DataFrame,
    ruta: Path = RUTA_CACHE,
):
    """Guarda fases originales, sin registro ni recorte."""

    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".tmp.npy")

    array = np.lib.format.open_memmap(
        temporal,
        mode="w+",
        dtype=np.uint8,
        shape=(len(samples), 3, TAMANO, TAMANO),
    )

    inicio = time.time()

    for indice, fila in enumerate(
        samples.itertuples(index=False)
    ):
        array[indice] = cargar_corte_uint8(fila)

        if (indice + 1) % 500 == 0:
            print(
                f"Caché: {indice + 1}/{len(samples)}",
                flush=True,
            )

    array.flush()
    del array

    temporal.replace(ruta)

    # Conserva el orden exacto para validar la correspondencia.
    ruta.with_suffix(".ids.json").write_text(
        json.dumps(samples.sample_id.astype(str).tolist()),
        encoding="utf-8",
    )

    print(f"Caché creada en {time.time() - inicio:.0f} s")


def cargar_cache(
    samples: pd.DataFrame,
    ruta: Path = RUTA_CACHE,
):
    if not ruta.exists():
        return None, None

    ruta_ids = ruta.with_suffix(".ids.json")

    if not ruta_ids.exists():
        print(
            "AVISO: caché sin índice verificable; se leerán PNG. "
            "Puedes reconstruirla con --construir-cache."
        )
        return None, None

    ids = json.loads(ruta_ids.read_text(encoding="utf-8"))
    ids_actuales = samples.sample_id.astype(str).tolist()

    if ids != ids_actuales:
        print(
            "AVISO: el orden de la caché no coincide "
            "con samples.csv; se leerán PNG."
        )
        return None, None

    array = np.load(ruta, mmap_mode="r")

    forma = (len(samples), 3, TAMANO, TAMANO)

    if array.shape != forma or array.dtype != np.uint8:
        print("AVISO: caché incompatible; se leerán PNG.")
        return None, None

    return array, {
        sample_id: indice
        for indice, sample_id in enumerate(ids)
    }


# ---------------------------------------------------------------------
# Separación y DataLoaders
# ---------------------------------------------------------------------

def _semilla_worker(worker_id):
    semilla = torch.initial_seed() % 2**32
    np.random.seed(semilla)


def crear_dataloaders(
    fold_val: int = 0,
    batch_size: int = 32,
    aumentar: bool = True,
    usar_cache: bool = True,
    num_workers: int = 0,
    semilla: int = 42,
    max_pacientes: int | None = None,
    config_espacial: ConfigEspacial | None = None,
) -> dict:
    _requiere_utils()

    samples = uc.cargar_samples(RAIZ_DATOS)
    filas_tr, filas_va = uc.particion(
        samples,
        fold_val=fold_val,
    )

    if max_pacientes:
        def limitar(filas, cantidad):
            pacientes = (
                filas.drop_duplicates("patient_id")
                .groupby("pCR")
                .head(cantidad)
                .patient_id
            )
            return filas[
                filas.patient_id.isin(pacientes)
            ]

        filas_tr = limitar(filas_tr, max_pacientes)
        filas_va = limitar(
            filas_va,
            max(2, max_pacientes // 2),
        )

    solape = (
        set(filas_tr.patient_id)
        & set(filas_va.patient_id)
    )

    if solape:
        raise ValueError(
            "Hay pacientes compartidas entre train y validación"
        )

    cache, indice_cache = (
        cargar_cache(samples)
        if usar_cache else (None, None)
    )

    aumento = AumentoGeometrico() if aumentar else None

    dataset_train = DatasetCortes(
        filas_tr,
        transform=aumento,
        cache=cache,
        indice_cache=indice_cache,
        config_espacial=config_espacial,
    )

    dataset_val = DatasetCortes(
        filas_va,
        cache=cache,
        indice_cache=indice_cache,
        config_espacial=config_espacial,
    )

    # Evaluación de train sin aumento y sin barajar.
    dataset_train_eval = DatasetCortes(
        filas_tr,
        cache=cache,
        indice_cache=indice_cache,
        config_espacial=config_espacial,
    )

    generador = torch.Generator().manual_seed(semilla)
    pin_memory = torch.cuda.is_available()

    train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generador,
        worker_init_fn=_semilla_worker,
        pin_memory=pin_memory,
    )

    val = DataLoader(
        dataset_val,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    train_eval = DataLoader(
        dataset_train_eval,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train,
        "val": val,
        "train_eval": train_eval,
        "filas_train": filas_tr.reset_index(drop=True),
        "filas_val": filas_va.reset_index(drop=True),
        "pos_weight": uc.pos_weight(filas_tr),
        "usa_cache": cache is not None,
        "aumento": repr(aumento),
        "preprocesado_espacial": (
            config_espacial or ConfigEspacial()
        ).to_dict(),
    }


def crear_dataloader_test(
    batch_size: int = 64,
    usar_cache: bool = True,
    num_workers: int = 0,
    config_espacial: ConfigEspacial | None = None,
):
    _requiere_utils()

    samples = uc.cargar_samples(RAIZ_DATOS)
    filas = uc.conjunto_test(samples).reset_index(drop=True)

    cache, indice_cache = (
        cargar_cache(samples)
        if usar_cache else (None, None)
    )

    dataset = DatasetCortes(
        filas,
        cache=cache,
        indice_cache=indice_cache,
        config_espacial=config_espacial,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return loader, filas


# ---------------------------------------------------------------------
# Comprobación rápida
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--construir-cache",
        action="store_true",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON con la configuración espacial",
    )
    parser.add_argument(
        "--max-pacientes",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    _requiere_utils()

    config = (
        ConfigEspacial.desde_json(args.config)
        if args.config else ConfigEspacial()
    )

    if args.construir_cache:
        construir_cache(uc.cargar_samples(RAIZ_DATOS))

    datos = crear_dataloaders(
        fold_val=0,
        batch_size=8,
        max_pacientes=args.max_pacientes,
        config_espacial=config,
    )

    x, y = next(iter(datos["train"]))

    print("Configuración espacial:", config.to_dict())
    print("Forma de las imágenes:", tuple(x.shape))
    print("Forma de las etiquetas:", tuple(y.shape))
    print("Rango:", float(x.min()), float(x.max()))
    print("Pos weight:", datos["pos_weight"])
    print("Usa caché:", datos["usa_cache"])

    assert x.shape[1:] == (3, TAMANO, TAMANO)
    assert torch.isfinite(x).all()
    assert 0 <= x.min() and x.max() <= 1

    print("Comprobación completada.")


if __name__ == "__main__":
    main()