#!/usr/bin/env python3
"""Visor de las muestras PNG del caso BreastDCEDL.

Tres modos de uso, pensados para explicar el dataset en clase:

    # 1. Una muestra: las tres fases DCE y el mapa de realce
    python ver_muestras.py muestra ISPY1_1001_z012

    # 2. Todos los cortes de una paciente, en rejilla
    python ver_muestras.py paciente ISPY1_1001

    # 3. Comparativa entre una paciente con pCR=0 y otra con pCR=1
    python ver_muestras.py comparar

Sin --salida las figuras se abren en pantalla; con --salida se guardan en PNG.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent
FASES = ("PRE", "EARLY", "LATE")


def cargar_metadata() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "metadata" / "samples.csv")


def cargar_imagen(fila) -> np.ndarray:
    """Apila las tres fases de una fila de samples.csv en (3, 256, 256).

    Cada fase es un PNG en gris independiente y ya viene en la orientacion de
    pantalla, asi que `pintar` no tiene que reorientar nada.
    """
    rutas = (fila.path_pre, fila.path_early, fila.path_late)
    planos = []
    for ruta in rutas:
        with Image.open(RAIZ / ruta) as png:
            planos.append(np.asarray(png.convert("L"), dtype=np.float32) / 255.0)
    return np.stack(planos)


def pintar(axis, plano: np.ndarray, titulo: str, cmap: str = "gray", vmin=0.0, vmax=1.0):
    axis.imshow(plano, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(titulo, fontsize=11)
    axis.axis("off")


def compuesto(imagen: np.ndarray) -> np.ndarray:
    """Las tres fases superpuestas en color, solo para visualizar: PRE/EARLY/LATE
    como R/G/B. En el dataset viven en ficheros separados; esto no es un fichero,
    es una ayuda visual para ver donde entra el contraste."""
    return imagen.transpose(1, 2, 0)


def modo_muestra(args, samples: pd.DataFrame) -> plt.Figure:
    fila = samples[samples.sample_id == args.sample_id]
    if fila.empty:
        raise SystemExit(f"No existe la muestra {args.sample_id}")
    fila = fila.iloc[0]
    imagen = cargar_imagen(fila)
    realce = imagen[1] - imagen[0]

    figura, ejes = plt.subplots(1, 5, figsize=(18.5, 4.2), constrained_layout=True)
    for eje, canal, fase in zip(ejes, imagen, FASES):
        pintar(eje, canal, f"{fase}  (canal {FASES.index(fase)})")
    # Ayuda visual, no un fichero del dataset: las tres fases superpuestas como
    # color. El realce del contraste aparece como tinte verde-azulado.
    ejes[3].imshow(compuesto(imagen))
    ejes[3].set_title("COMPUESTO  (solo visual)", fontsize=11)
    ejes[3].axis("off")
    limite = float(np.abs(realce).max()) or 1.0
    pintar(ejes[4], realce, "REALCE  (EARLY - PRE)", cmap="magma", vmin=0.0, vmax=limite)
    figura.suptitle(
        f"{fila.sample_id}   |   paciente {fila.patient_id}   |   "
        f"corte z={fila.slice_index}   |   split {fila.split}   |   pCR={fila.pCR}",
        fontsize=13, fontweight="bold",
    )
    return figura


def modo_paciente(args, samples: pd.DataFrame) -> plt.Figure:
    cortes = samples[samples.patient_id == args.patient_id].sort_values("slice_index")
    if cortes.empty:
        raise SystemExit(f"No existe la paciente {args.patient_id}")
    columnas = 5
    filas = int(np.ceil(len(cortes) / columnas))
    figura, ejes = plt.subplots(filas, columnas, figsize=(3 * columnas, 3 * filas),
                                constrained_layout=True)
    ejes = np.atleast_1d(ejes).ravel()
    for eje, (_, fila) in zip(ejes, cortes.iterrows()):
        imagen = cargar_imagen(fila)
        canal = imagen[1] if args.fase == "EARLY" else imagen[FASES.index(args.fase)]
        pintar(eje, canal, f"z={fila.slice_index}")
    for eje in ejes[len(cortes):]:
        eje.axis("off")
    pcr = cortes.pCR.iloc[0]
    figura.suptitle(
        f"Paciente {args.patient_id}  |  {len(cortes)} cortes  |  fase {args.fase}  |  pCR={pcr}",
        fontsize=14, fontweight="bold",
    )
    return figura


def modo_comparar(args, samples: pd.DataFrame) -> plt.Figure:
    figura, ejes = plt.subplots(2, 4, figsize=(15, 8.4), constrained_layout=True)
    for indice, etiqueta in enumerate((0, 1)):
        grupo = samples[(samples.split == "train") & (samples.pCR == etiqueta)]
        paciente = sorted(grupo.patient_id.unique())[args.indice]
        cortes = grupo[grupo.patient_id == paciente].sort_values("slice_index")
        fila = cortes.iloc[len(cortes) // 2]          # corte central del tumor
        imagen = cargar_imagen(fila)
        realce = imagen[1] - imagen[0]
        for eje, canal, fase in zip(ejes[indice], imagen, FASES):
            pintar(eje, canal, fase)
        limite = float(np.abs(realce).max()) or 1.0
        pintar(ejes[indice][3], realce, "REALCE (EARLY - PRE)", cmap="magma", vmin=0.0, vmax=limite)
        ejes[indice][0].text(-0.06, 0.5, f"pCR = {etiqueta}\n{paciente}", rotation=90,
                             va="center", ha="center", transform=ejes[indice][0].transAxes,
                             fontsize=12, fontweight="bold")
    figura.suptitle("Misma anatomia, distinta respuesta al tratamiento", fontsize=14, fontweight="bold")
    return figura


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    comun = argparse.ArgumentParser(add_help=False)
    comun.add_argument("--salida", type=Path, help="guardar en PNG en lugar de mostrar")
    sub = parser.add_subparsers(dest="modo", required=True)

    p1 = sub.add_parser("muestra", parents=[comun], help="las 3 fases y el realce de un corte")
    p1.add_argument("sample_id")

    p2 = sub.add_parser("paciente", parents=[comun], help="todos los cortes de una paciente")
    p2.add_argument("patient_id")
    p2.add_argument("--fase", choices=FASES, default="EARLY")

    p3 = sub.add_parser("comparar", parents=[comun], help="un caso pCR=0 frente a uno pCR=1")
    p3.add_argument("--indice", type=int, default=0, help="que par de pacientes usar")

    args = parser.parse_args()

    samples = cargar_metadata()
    figura = {"muestra": modo_muestra, "paciente": modo_paciente,
              "comparar": modo_comparar}[args.modo](args, samples)

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(args.salida, dpi=140, bbox_inches="tight", facecolor="white")
        print(f"Guardado en {args.salida}")
    else:
        plt.show()
    plt.close(figura)


if __name__ == "__main__":
    main()
