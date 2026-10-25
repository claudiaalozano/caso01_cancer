"""Compara los LATE sospechosos con una copia del servidor original."""

import hashlib
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAIZ = REPO / "data" / "breastdcedl"

BASE = (
    "https://storage.googleapis.com/"
    "usecasesf-breastdcedl-alumnos-8264/breastdcedl"
)

SALIDA = REPO / "resultados" / "comprobacion_origen"
PACIENTE = "ISPY1_1148"


def sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def leer_png(ruta):
    with Image.open(ruta) as imagen:
        imagen.load()
        return np.array(imagen)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    for corte in range(8, 18):
        nombre = f"{PACIENTE}_z{corte:03d}_LATE.png"

        relativa = (
            Path("dataset")
            / "train"
            / PACIENTE
            / nombre
        )

        local = RAIZ / relativa
        copia = SALIDA / nombre

        if not local.exists():
            print(f"{nombre}: no encuentro el archivo local")
            continue

        try:
            respuesta = requests.get(
                f"{BASE}/{relativa.as_posix()}",
                timeout=60,
            )
            respuesta.raise_for_status()

            # La copia se guarda fuera del dataset.
            copia.write_bytes(respuesta.content)

            original = leer_png(local)
            descargada = leer_png(copia)

            mismos_pixeles = np.array_equal(
                original,
                descargada,
            )

            print(f"\n{nombre}")
            print(
                "  Archivos idénticos:",
                sha256(local) == sha256(copia),
            )
            print(
                "  Píxeles idénticos:",
                mismos_pixeles,
            )
            print(
                f"  Media local: {original.mean():.3f}"
            )
            print(
                f"  Media servidor: {descargada.mean():.3f}"
            )

        except (
            requests.RequestException,
            OSError,
            ValueError,
        ) as error:
            print(f"{nombre}: error: {error}")


if __name__ == "__main__":
    main()