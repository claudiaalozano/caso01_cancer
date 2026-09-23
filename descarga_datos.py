#!/usr/bin/env python3
"""Descarga el dataset del caso BreastDCEDL desde el bucket publico.

Este fichero es autonomo: puedes descargarlo suelto y ejecutarlo en una carpeta
vacia. Se baja el indice, la documentacion y las 38.109 imagenes.

    pip install requests
    python descargar_datos.py

Opciones utiles:

    python descargar_datos.py --destino C:\\bdcedl     ruta corta (Windows)
    python descargar_datos.py --solo-test              solo el conjunto de prueba
    python descargar_datos.py --pacientes 5            5 pacientes, para probar
    python descargar_datos.py --hilos 32               mas paralelismo

Es reanudable: si se corta, vuelve a lanzarlo y solo bajara lo que falte.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Falta la libreria 'requests'. Instalala con:  pip install requests")

BASE = "https://storage.googleapis.com/usecasesf-breastdcedl-alumnos-8264/breastdcedl"

# Ficheros sueltos que no son imagenes.
AUXILIARES = [
    "GUIA.md",
    "LICENSE",
    "utils_caso.py",
    "ver_muestras.py",
    "metadata/samples.csv",
    "metadata/patients.csv",
    "metadata/excluded_patients.csv",
    "metadata/statistics.json",
    "documentation/caso_breastdcedl.pdf",
    "documentation/figures/bucket_estructura.png",
    "documentation/figures/example_dce.png",
    "documentation/figures/estructura_dataset.png",
    "documentation/figures/pcr_two_patient_paths.png",
]


def bajar(sesion: requests.Session, ruta: str, destino: Path) -> tuple[str, bool, str]:
    """Descarga un fichero si no existe ya. Devuelve (ruta, ok, mensaje)."""
    salida = destino / ruta
    if salida.exists() and salida.stat().st_size > 0:
        return ruta, True, "ya estaba"
    try:
        r = sesion.get(f"{BASE}/{ruta}", timeout=60)
        r.raise_for_status()
        salida.parent.mkdir(parents=True, exist_ok=True)
        salida.write_bytes(r.content)
        return ruta, True, "ok"
    except Exception as e:                                  # red, 404, disco...
        return ruta, False, str(e)[:80]


def descargar_lote(rutas: list[str], destino: Path, hilos: int, etiqueta: str) -> list[str]:
    """Descarga en paralelo mostrando progreso. Devuelve las rutas que fallaron."""
    fallos: list[str] = []
    hechos = 0
    total = len(rutas)
    with requests.Session() as sesion:
        adaptador = requests.adapters.HTTPAdapter(pool_maxsize=hilos, max_retries=3)
        sesion.mount("https://", adaptador)
        with ThreadPoolExecutor(max_workers=hilos) as pool:
            futuros = [pool.submit(bajar, sesion, r, destino) for r in rutas]
            for fut in as_completed(futuros):
                ruta, ok, msg = fut.result()
                hechos += 1
                if not ok:
                    fallos.append(ruta)
                if hechos % 50 == 0 or hechos == total:
                    print(f"\r  {etiqueta}: {hechos}/{total}"
                          f"{f'  ({len(fallos)} fallos)' if fallos else ''}",
                          end="", flush=True)
    print()
    return fallos


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--destino", type=Path, default=Path("breastdcedl"))
    p.add_argument("--hilos", type=int, default=16)
    p.add_argument("--solo-test", action="store_true", help="solo el conjunto de prueba")
    p.add_argument("--pacientes", type=int, metavar="N",
                   help="limitar a N pacientes, para probar el pipeline")
    args = p.parse_args()

    destino: Path = args.destino
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Destino: {destino.resolve()}\n")

    # --- 1. auxiliares ---
    fallos = descargar_lote(AUXILIARES, destino, args.hilos, "documentacion")
    if any(f == "metadata/samples.csv" for f in fallos):
        sys.exit("No se pudo descargar samples.csv. Revisa tu conexion.")

    # --- 2. leer el indice para saber que imagenes hacen falta ---
    with open(destino / "metadata/samples.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    if args.solo_test:
        filas = [x for x in filas if x["split"] == "test"]
    if args.pacientes:
        elegidas = sorted({x["patient_id"] for x in filas})[:args.pacientes]
        filas = [x for x in filas if x["patient_id"] in elegidas]

    imagenes = [x[c] for x in filas for c in ("path_pre", "path_early", "path_late")]
    n_pac = len({x["patient_id"] for x in filas})
    print(f"\n{len(imagenes)} imagenes de {len(filas)} cortes, {n_pac} pacientes\n")

    # --- 3. imagenes ---
    fallos = descargar_lote(imagenes, destino, args.hilos, "imagenes")

    # --- 4. reintento de los fallos ---
    if fallos:
        print(f"\nReintentando {len(fallos)} que fallaron...")
        fallos = descargar_lote(fallos, destino, max(4, args.hilos // 2), "reintento")

    # --- 5. verificacion ---
    faltan = [r for r in imagenes if not (destino / r).exists()]
    print()
    if faltan:
        print(f"FALTAN {len(faltan)} imagenes. Vuelve a ejecutar para reanudar.")
        for r in faltan[:5]:
            print(f"   {r}")
        sys.exit(1)

    print(f"COMPLETO: {len(imagenes)} imagenes en {destino.resolve()}")
    print("\nSiguiente paso: lee GUIA.md")


if __name__ == "__main__":
    main()