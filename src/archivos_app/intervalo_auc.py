"""Intervalo bootstrap del AUC a partir de predicciones por paciente."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--repeticiones", type=int, default=5000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--col-prob", default="prob")
    parser.add_argument("--col-etiqueta", default="pCR")
    args = parser.parse_args()

    if args.repeticiones < 1000:
        parser.error("Utiliza al menos 1000 repeticiones.")

    ruta = Path(args.csv)
    tabla = pd.read_csv(ruta)

    necesarias = {args.col_prob, args.col_etiqueta}

    if not necesarias.issubset(tabla.columns):
        raise ValueError(
            f"Columnas disponibles: {list(tabla.columns)}. "
            "Indica las correctas con --col-prob y --col-etiqueta."
        )

    # Comprobar que hay una sola fila por paciente.
    if "patient_id" in tabla.columns:
        columna_id = "patient_id"
    elif str(tabla.columns[0]).startswith("Unnamed:"):
        columna_id = tabla.columns[0]
    else:
        raise ValueError(
            "No encuentro el identificador de paciente. "
            "Revisa la cabecera del CSV antes de continuar."
        )

    ids = tabla[columna_id].astype("string")

    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError("Hay identificadores de paciente vacíos.")

    if ids.duplicated().any():
        raise ValueError(
            "Hay pacientes repetidas: se requiere una fila por paciente."
        )

    y = pd.to_numeric(
        tabla[args.col_etiqueta], errors="raise"
    ).to_numpy(dtype=float)

    prob = pd.to_numeric(
        tabla[args.col_prob], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(y).all() or not np.isfinite(prob).all():
        raise ValueError("Hay etiquetas o puntuaciones no finitas.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Las etiquetas deben ser 0 o 1.")

    if np.any((prob < 0) | (prob > 1)):
        raise ValueError("Las puntuaciones deben estar entre 0 y 1.")

    y = y.astype(int)
    n = len(y)
    positivos = int(y.sum())
    negativos = n - positivos

    # Comprobación del test concreto que estamos analizando.
    if (n, negativos, positivos) != (176, 123, 53):
        raise ValueError(
            f"El CSV contiene {n} pacientes, {negativos} ceros "
            f"y {positivos} unos. No coincide con el test esperado."
        )

    auc_original = float(roc_auc_score(y, prob))

    if not np.isclose(auc_original, 0.518178, atol=0.00001):
        raise ValueError(
            f"AUC encontrado: {auc_original:.6f}. "
            "No coincide con el resultado guardado de esta CNN."
        )

    rng = np.random.default_rng(args.semilla)
    auc_bootstrap = []
    descartadas = 0

    # Bootstrap ordinario por paciente: muestras de tamaño n,
    # seleccionadas con reemplazo.
    for _ in range(args.repeticiones):
        indices = rng.integers(0, n, size=n)
        y_muestra = y[indices]

        if np.unique(y_muestra).size < 2:
            descartadas += 1
            continue

        auc_bootstrap.append(
            roc_auc_score(y_muestra, prob[indices])
        )

    valores = np.asarray(auc_bootstrap, dtype=float)

    if len(valores) < 1000:
        raise ValueError("No hay suficientes muestras bootstrap válidas.")

    inferior, superior = np.percentile(valores, [2.5, 97.5])

    resultado = {
        "archivo_predicciones": str(ruta.resolve()),
        "csv_sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
        "pacientes": n,
        "pCR_0": negativos,
        "pCR_1": positivos,
        "auc": auc_original,
        "ic95_inferior": float(inferior),
        "ic95_superior": float(superior),
        "metodo": "Bootstrap ordinario por paciente, intervalo percentil",
        "repeticiones_solicitadas": args.repeticiones,
        "repeticiones_validas": len(valores),
        "muestras_sin_ambas_clases": descartadas,
        "semilla": args.semilla,
        "inferencia_repetida": False,
        "limitacion": (
            "Incertidumbre estimada para el modelo fijo a partir "
            "de esta muestra. No mide variabilidad de entrenamiento "
            "ni garantiza rendimiento en otras poblaciones."
        ),
    }

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = ruta.parent / f"bootstrap_auc_{fecha}"
    salida.mkdir(exist_ok=False)

    (salida / "intervalo_auc.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame({"auc_bootstrap": valores}).to_csv(
        salida / "distribucion_auc.csv",
        index=False,
    )

    print(f"\nPacientes: {n} | pCR=0: {negativos} | pCR=1: {positivos}")
    print(f"AUC de test: {auc_original:.6f}")
    print(f"IC bootstrap del 95 %: [{inferior:.6f}, {superior:.6f}]")
    print(f"Repeticiones válidas: {len(valores)}")
    print(f"\nResultados: {salida.resolve()}")
    print("No se ha cargado la CNN ni repetido la inferencia.")


if __name__ == "__main__":
    main()