#!/usr/bin/env python3
"""Preprocesado del caso BreastDCEDL: de los PNG al tensor que recibe la CNN.

Este módulo es la ÚNICA fuente de verdad sobre cómo se cargan las imágenes.
Lo usan el entrenamiento, la evaluación y la aplicación web, para garantizar
que la normalización es idéntica en todos los casos (lo comprueban en la defensa).

Qué hace:
  1. Carga y escala   3 PNG (PRE, EARLY, LATE) -> tensor (3, 256, 256) en [0, 1]
  2. Validación       comprueba formato, modo gris, tamaño y fases (para la app)
  3. Aumento          volteo horizontal + rotación/traslación pequeñas, IGUALES
                      para las 3 fases y SOLO en entrenamiento
  4. Separación       folds por paciente (uc.particion) y pos_weight del train
  5. Caché opcional   todos los cortes en un .npy uint8 para acelerar el entrenamiento

Qué NO hace, a propósito:
  - No normaliza cada canal por separado (media 0 / varianza 1 por fase): las
    tres fases comparten una ventana de intensidad y normalizarlas por separado
    destruiría el realce EARLY - PRE, que es la señal del problema.
  - No usa la normalización de ImageNet: los canales no son colores RGB.
  - No aplica cambios de brillo, contraste ni ruido: alterarían el realce.
  - No re-muestrea las clases: el desbalance se trata con pos_weight en la pérdida.

Uso desde la raíz del repositorio:

    python src/preprocessing.py                    # comprobaciones rápidas
    python src/preprocessing.py --construir-cache  # crea data/cache/cortes_uint8.npy
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Rutas y constantes
# --------------------------------------------------------------------------- #

REPO = Path(__file__).resolve().parents[1]
RAIZ_DATOS = REPO / "data" / "breastdcedl"
RUTA_CACHE = REPO / "data" / "cache" / "cortes_uint8.npy"

FASES = ("PRE", "EARLY", "LATE")          # orden de los canales: SIEMPRE este
TAMANO = 256
ESCALA = 255.0                            # única normalización: dividir entre 255
MAX_BYTES_PNG = 1_000_000                 # límite de tamaño por fichero (para la app)

sys.path.insert(0, str(RAIZ_DATOS))
import utils_caso as uc                   # noqa: E402  (particion, pos_weight)


# =========================================================================== #
# 1. CARGA Y ESCALA
# =========================================================================== #

class ErrorEntrada(ValueError):
    """Error de validación con un mensaje claro para el usuario de la app."""


def _abrir_png(fuente: str | Path | bytes, nombre: str) -> np.ndarray:
    """Abre UN PNG y valida que sea gris 8 bits de 256x256. Devuelve uint8 (256, 256).

    `fuente` puede ser una ruta (entrenamiento) o los bytes de un fichero subido (app).
    """
    try:
        if isinstance(fuente, (bytes, bytearray)):
            if len(fuente) > MAX_BYTES_PNG:
                raise ErrorEntrada(f"{nombre}: el fichero supera {MAX_BYTES_PNG // 1000} KB")
            fuente = io.BytesIO(fuente)
        with Image.open(fuente) as png:
            png.load()                                   # fuerza la lectura: detecta corruptos
            if png.format != "PNG":
                raise ErrorEntrada(f"{nombre}: no es un PNG (es {png.format})")
            if png.mode != "L":
                raise ErrorEntrada(f"{nombre}: debe ser escala de grises (modo 'L'), es '{png.mode}'")
            if png.size != (TAMANO, TAMANO):
                raise ErrorEntrada(f"{nombre}: debe medir {TAMANO}x{TAMANO}, mide {png.size[0]}x{png.size[1]}")
            return np.asarray(png, dtype=np.uint8)
    except ErrorEntrada:
        raise
    except FileNotFoundError:
        raise ErrorEntrada(f"{nombre}: no existe el fichero")
    except Exception as e:                               # PNG corrupto o ilegible
        raise ErrorEntrada(f"{nombre}: fichero corrupto o ilegible ({type(e).__name__})")


def a_tensor(fases_uint8: np.ndarray) -> torch.Tensor:
    """(3, 256, 256) uint8 -> tensor float32 en [0, 1]. LA normalización del proyecto."""
    x = torch.from_numpy(np.array(fases_uint8, dtype=np.uint8)).float() / ESCALA   # copia
    if not torch.isfinite(x).all():                      # por seguridad (app)
        raise ErrorEntrada("la imagen contiene valores no finitos")
    return x


def cargar_corte(pre, early, late) -> torch.Tensor:
    """Carga las 3 fases de un corte (rutas o bytes) -> tensor (3, 256, 256) en [0, 1].

    Es la función que debe usar también la aplicación web.
    """
    fases = np.stack([_abrir_png(f, n) for f, n in zip((pre, early, late), FASES)])
    return a_tensor(fases)


def cargar_corte_uint8(fila, raiz: Path = RAIZ_DATOS) -> np.ndarray:
    """Las 3 fases de una fila de samples.csv, sin escalar: uint8 (3, 256, 256)."""
    rutas = (fila.path_pre, fila.path_early, fila.path_late)
    return np.stack([_abrir_png(raiz / r, f) for r, f in zip(rutas, FASES)])


# =========================================================================== #
# 2. VALIDACIÓN DE UNA SUBIDA (para la app)
# =========================================================================== #

def validar_subida(ficheros: dict[str, bytes]) -> torch.Tensor:
    """Valida lo que sube un usuario a la app y devuelve el tensor listo para la red.

    `ficheros` es {nombre_de_fichero: bytes}. La fase se deduce del sufijo del
    nombre (_PRE.png, _EARLY.png, _LATE.png). Errores claros si falta una fase,
    está duplicada, o algún fichero no es válido.
    """
    por_fase: dict[str, list[str]] = {f: [] for f in FASES}
    for nombre in ficheros:
        base = Path(nombre).name.upper()
        if not base.endswith(".PNG"):
            raise ErrorEntrada(f"{nombre}: solo se aceptan ficheros .png")
        for fase in FASES:
            if base.endswith(f"_{fase}.PNG"):
                por_fase[fase].append(nombre)
    faltan = [f for f, v in por_fase.items() if not v]
    duplicadas = [f for f, v in por_fase.items() if len(v) > 1]
    if faltan:
        raise ErrorEntrada(f"faltan fases: {', '.join(faltan)}")
    if duplicadas:
        raise ErrorEntrada(f"fases duplicadas: {', '.join(duplicadas)}")
    if len(ficheros) != 3:
        raise ErrorEntrada("hay que subir exactamente 3 ficheros (PRE, EARLY y LATE)")
    return cargar_corte(*(ficheros[por_fase[f][0]] for f in FASES))


# =========================================================================== #
# 3. AUMENTO DE DATOS (solo entrenamiento)
# =========================================================================== #

class AumentoGeometrico:
    """Volteo horizontal + rotación y traslación pequeñas.

    La MISMA transformación se aplica a los 3 canales a la vez: si se moviera
    una fase y no las otras, se desalinearían y se destruiría el realce.

    - Volteo horizontal: una mama izquierda y una derecha son simétricas.
    - NO volteo vertical: la orientación cabeza-pies sí tiene sentido anatómico.
    - Rotación ±10° y traslación ±8 px: pequeñas variaciones de posición.
    """

    def __init__(self, p_volteo: float = 0.5, max_grados: float = 10.0,
                 max_desplazamiento: int = 8):
        self.p_volteo = p_volteo
        self.max_grados = max_grados
        self.max_desplazamiento = max_desplazamiento

    def __call__(self, x: torch.Tensor) -> torch.Tensor:        # x: (3, H, W)
        if torch.rand(1).item() < self.p_volteo:
            x = torch.flip(x, dims=[2])                          # eje W: izquierda-derecha

        angulo = math.radians((torch.rand(1).item() * 2 - 1) * self.max_grados)
        _, h, w = x.shape
        dx = (torch.rand(1).item() * 2 - 1) * self.max_desplazamiento * 2 / w
        dy = (torch.rand(1).item() * 2 - 1) * self.max_desplazamiento * 2 / h
        cos, sen = math.cos(angulo), math.sin(angulo)
        theta = torch.tensor([[cos, -sen, dx], [sen, cos, dy]], dtype=x.dtype).unsqueeze(0)
        malla = F.affine_grid(theta, [1, *x.shape], align_corners=False)
        # Una sola malla para los 3 canales -> las fases siguen alineadas.
        # Fuera de la imagen se rellena con 0 (fondo negro, como el aire).
        return F.grid_sample(x.unsqueeze(0), malla, mode="bilinear",
                             padding_mode="zeros", align_corners=False).squeeze(0)

    def __repr__(self):
        return (f"AumentoGeometrico(p_volteo={self.p_volteo}, max_grados={self.max_grados}, "
                f"max_desplazamiento={self.max_desplazamiento})")


# =========================================================================== #
# 4. DATASET, SEPARACIÓN Y DATALOADERS
# =========================================================================== #

class DatasetCortes(Dataset):
    """Devuelve (x, y) con x = tensor (3, 256, 256) en [0, 1] e y = pCR (float).

    Lee de la caché .npy si se le pasa; si no, de los PNG. El resultado es idéntico.
    """

    def __init__(self, filas: pd.DataFrame, transform=None, cache: np.ndarray | None = None,
                 indice_cache: dict[str, int] | None = None, raiz: Path = RAIZ_DATOS):
        self.filas = filas.reset_index(drop=True)
        self.transform = transform
        self.cache = cache
        self.indice_cache = indice_cache
        self.raiz = raiz

    def __len__(self) -> int:
        return len(self.filas)

    def __getitem__(self, i: int):
        fila = self.filas.iloc[i]
        if self.cache is not None:
            fases = self.cache[self.indice_cache[fila.sample_id]]
        else:
            fases = cargar_corte_uint8(fila, self.raiz)
        x = a_tensor(fases)
        if self.transform is not None:
            x = self.transform(x)
        return x, torch.tensor(float(fila.pCR))


def construir_cache(samples: pd.DataFrame, ruta: Path = RUTA_CACHE) -> None:
    """Guarda TODOS los cortes (train y test) en un .npy uint8 (N, 3, 256, 256), ~2,5 GB.

    El orden es el de samples.csv. Se guarda sin escalar (uint8) para ocupar 4 veces
    menos; la división entre 255 se hace al leer, igual que con los PNG.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                    shape=(len(samples), 3, TAMANO, TAMANO))
    t0 = time.time()
    for i, fila in enumerate(samples.itertuples(index=False)):
        arr[i] = cargar_corte_uint8(fila)
        if (i + 1) % 500 == 0 or i + 1 == len(samples):
            print(f"\r  caché: {i + 1}/{len(samples)} cortes", end="", flush=True)
    arr.flush()
    del arr
    tmp.replace(ruta)
    print(f"   [{time.time() - t0:.0f} s] -> {ruta}")


def cargar_cache(samples: pd.DataFrame, ruta: Path = RUTA_CACHE):
    """Abre la caché sin cargarla entera en RAM (memmap). Devuelve (array, índice) o (None, None)."""
    if not ruta.exists():
        return None, None
    arr = np.load(ruta, mmap_mode="r")
    if arr.shape[0] != len(samples):
        print(f"  AVISO: la caché tiene {arr.shape[0]} cortes y samples.csv {len(samples)}; se ignora.")
        return None, None
    return arr, {sid: i for i, sid in enumerate(samples.sample_id)}


def _semilla_worker(worker_id: int):
    """Cada worker del DataLoader con su propia semilla derivada: reproducible."""
    semilla = torch.initial_seed() % 2**32
    np.random.seed(semilla)


def crear_dataloaders(fold_val: int = 0, batch_size: int = 32, aumentar: bool = True,
                      usar_cache: bool = True, num_workers: int = 0, semilla: int = 42,
                      max_pacientes: int | None = None) -> dict:
    """Prepara todo lo que necesita el entrenamiento para un fold de validación.

    Devuelve un diccionario con:
      train, val        DataLoaders (train con shuffle y aumento; val sin ninguno)
      filas_train/val   DataFrames en el MISMO orden que los DataLoaders de evaluación
      pos_weight        N0/N1 calculado SOLO sobre la parte de entrenamiento
    `max_pacientes` reduce el problema (modo prueba en casa): N pacientes por clase.
    """
    samples = uc.cargar_samples(RAIZ_DATOS)
    filas_tr, filas_va = uc.particion(samples, fold_val=fold_val)   # comprueba fuga

    if max_pacientes:                                   # modo prueba: pocos pacientes
        def recortar(filas, n):
            pac = filas.drop_duplicates("patient_id").groupby("pCR").head(n).patient_id
            return filas[filas.patient_id.isin(pac)]
        filas_tr = recortar(filas_tr, max_pacientes)
        filas_va = recortar(filas_va, max(2, max_pacientes // 2))

    cache, indice = cargar_cache(samples) if usar_cache else (None, None)
    transform = AumentoGeometrico() if aumentar else None

    gen = torch.Generator().manual_seed(semilla)        # orden de barajado reproducible
    dl_tr = DataLoader(DatasetCortes(filas_tr, transform, cache, indice), batch_size=batch_size,
                       shuffle=True, num_workers=num_workers, generator=gen,
                       worker_init_fn=_semilla_worker, pin_memory=torch.cuda.is_available())
    # Validación: SIN aumento y SIN shuffle, para que las probabilidades coincidan
    # fila a fila con filas_val al agregar por paciente.
    dl_va = DataLoader(DatasetCortes(filas_va, None, cache, indice), batch_size=batch_size * 2,
                       shuffle=False, num_workers=num_workers,
                       pin_memory=torch.cuda.is_available())

    return {
        "train": dl_tr, "val": dl_va,
        "filas_train": filas_tr.reset_index(drop=True),
        "filas_val": filas_va.reset_index(drop=True),
        "pos_weight": uc.pos_weight(filas_tr),
        "usa_cache": cache is not None,
        "aumento": repr(transform),
    }


def crear_dataloader_test(batch_size: int = 64, usar_cache: bool = True,
                          num_workers: int = 0) -> tuple[DataLoader, pd.DataFrame]:
    """Test: úsalo UNA sola vez, al final, con el modelo ya cerrado."""
    samples = uc.cargar_samples(RAIZ_DATOS)
    filas = uc.conjunto_test(samples).reset_index(drop=True)
    cache, indice = cargar_cache(samples) if usar_cache else (None, None)
    return DataLoader(DatasetCortes(filas, None, cache, indice), batch_size=batch_size,
                      shuffle=False, num_workers=num_workers), filas


# =========================================================================== #
# COMPROBACIONES (python src/preprocessing.py)
# =========================================================================== #

def comprobar(max_pacientes: int):
    print("1) Carga y escala")
    samples = uc.cargar_samples(RAIZ_DATOS)
    fila = samples.iloc[0]
    x = cargar_corte(*(RAIZ_DATOS / p for p in (fila.path_pre, fila.path_early, fila.path_late)))
    x_prof = torch.from_numpy(uc.cargar_imagen(fila, RAIZ_DATOS))
    print(f"   {fila.sample_id}: forma {tuple(x.shape)}, {x.dtype}, rango [{x.min():.3f}, {x.max():.3f}]")
    assert x.shape == (3, TAMANO, TAMANO) and 0 <= x.min() and x.max() <= 1
    assert torch.equal(x, x_prof), "difiere de utils_caso.cargar_imagen"
    print("   idéntico a utils_caso.cargar_imagen: OK")

    print("2) Validación de subidas")
    bytes_ok = {Path(p).name: (RAIZ_DATOS / p).read_bytes()
                for p in (fila.path_pre, fila.path_early, fila.path_late)}
    assert torch.equal(validar_subida(bytes_ok), x)
    casos = {
        "falta una fase": dict(list(bytes_ok.items())[:2]),
        "fichero corrupto": {**bytes_ok, Path(fila.path_late).name: b"no soy un png"},
        "no es PNG": {**dict(list(bytes_ok.items())[:2]), "x_LATE.jpg": b"..."},
    }
    for caso, fich in casos.items():
        try:
            validar_subida(fich)
            raise AssertionError(f"no detectó: {caso}")
        except ErrorEntrada as e:
            print(f"   {caso:17s} -> rechazado: {e}")

    print("3) Aumento: misma transformación en las 3 fases")
    torch.manual_seed(0)
    aug = AumentoGeometrico(p_volteo=1.0)
    xa = aug(x)
    # Si las fases siguen alineadas, aumentar y luego restar = restar y luego aumentar
    torch.manual_seed(0)
    realce_aug = aug(torch.stack([x[1] - x[0]] * 3))[0]
    err = (xa[1] - xa[0] - realce_aug).abs().max().item()
    print(f"   forma {tuple(xa.shape)}; error de alineación del realce = {err:.2e}")
    assert err < 1e-5

    print("4) DataLoaders y pos_weight")
    d = crear_dataloaders(fold_val=0, batch_size=8, max_pacientes=max_pacientes)
    xb, yb = next(iter(d["train"]))
    print(f"   train: {len(d['filas_train'])} cortes / {d['filas_train'].patient_id.nunique()} pacientes")
    print(f"   val:   {len(d['filas_val'])} cortes / {d['filas_val'].patient_id.nunique()} pacientes")
    print(f"   batch: x {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)}")
    print(f"   pos_weight (solo train) = {d['pos_weight']:.3f} | caché: {d['usa_cache']}")
    if d["usa_cache"]:                                  # la caché debe dar lo mismo que los PNG
        ds = d["val"].dataset
        x_png = DatasetCortes(ds.filas.iloc[:1])[0][0]
        assert torch.equal(ds[0][0], x_png), "la caché no coincide con los PNG"
        print("   caché idéntica a los PNG: OK")
    solape = set(d["filas_train"].patient_id) & set(d["filas_val"].patient_id)
    assert not solape and xb.shape[1:] == (3, TAMANO, TAMANO)
    print("\nTodo correcto.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--construir-cache", action="store_true",
                   help=f"guarda todos los cortes en {RUTA_CACHE.relative_to(REPO)} (~2,5 GB)")
    p.add_argument("--max-pacientes", type=int, default=5,
                   help="pacientes por clase en la comprobación de DataLoaders")
    args = p.parse_args()

    if not (RAIZ_DATOS / "metadata" / "samples.csv").exists():
        sys.exit(f"No encuentro los datos en {RAIZ_DATOS}")
    if args.construir_cache:
        construir_cache(uc.cargar_samples(RAIZ_DATOS))
    comprobar(args.max_pacientes)


if __name__ == "__main__":
    main()
