"""Analiza predicciones de test guardadas. No carga ni ejecuta la CNN."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]


def identificar_columna_id(tabla):
    for nombre in ("patient_id", "pid", "PatientID", "patient", "id"):
        if nombre in tabla.columns:
            return nombre

    primera = str(tabla.columns[0])

    if primera.startswith("Unnamed:"):
        return tabla.columns[0]

    raise ValueError(
        f"No encuentro el identificador. Columnas: {list(tabla.columns)}"
    )


def normalizar_ids(tabla, columna):
    ids = tabla[columna].astype("string").str.strip()

    if ids.isna().any() or ids.eq("").any():
        raise ValueError("Hay identificadores vacíos.")

    if ids.duplicated().any():
        raise ValueError("Hay identificadores de paciente duplicados.")

    return ids


def metricas(tabla, umbral):
    y = tabla["pCR"].to_numpy(dtype=int)
    prob = tabla["prob"].to_numpy(dtype=float)
    pred = (prob >= umbral).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    tiene_ambas = np.unique(y).size == 2

    recalls = recall_score(
        y, pred, labels=[0, 1],
        average=None, zero_division=0,
    )

    precisiones = precision_score(
        y, pred, labels=[0, 1],
        average=None, zero_division=0,
    )

    f1 = f1_score(
        y, pred, labels=[0, 1],
        average=None, zero_division=0,
    )

    return {
        "pacientes": len(tabla),
        "pCR_0": int((y == 0).sum()),
        "pCR_1": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, prob)) if tiene_ambas else np.nan,
        "brier": float(brier_score_loss(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "recall_0": float(recalls[0]) if (y == 0).any() else np.nan,
        "recall_1": float(recalls[1]) if (y == 1).any() else np.nan,
        "precision_0": float(precisiones[0]),
        "precision_1": float(precisiones[1]),
        "f1_0": float(f1[0]),
        "f1_1": float(f1[1]),
        "f1_macro": float(f1.mean()) if tiene_ambas else np.nan,
        "balanceada": float(recalls.mean()) if tiene_ambas else np.nan,
        "VN": int(vn),
        "FP": int(fp),
        "FN": int(fn),
        "VP": int(vp),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patients")
    parser.add_argument("--umbral", type=float, default=0.3)
    args = parser.parse_args()

    if not 0 <= args.umbral <= 1:
        parser.error("El umbral debe estar entre 0 y 1.")

    ruta = Path(args.csv)
    test = pd.read_csv(ruta)

    if not {"prob", "pCR"}.issubset(test.columns):
        raise ValueError(
            f"Se necesitan prob y pCR. Columnas: {list(test.columns)}"
        )

    id_test = identificar_columna_id(test)
    test["patient_id_limpio"] = normalizar_ids(test, id_test)

    for columna in ("prob", "pCR"):
        test[columna] = pd.to_numeric(test[columna], errors="raise")

    if not np.isfinite(test[["prob", "pCR"]].to_numpy()).all():
        raise ValueError("Hay puntuaciones o etiquetas no finitas.")

    if not test["pCR"].isin([0, 1]).all():
        raise ValueError("pCR debe contener solo 0 y 1.")

    if not test["prob"].between(0, 1).all():
        raise ValueError("Las puntuaciones deben estar entre 0 y 1.")

    if len(test) != 176 or int(test["pCR"].sum()) != 53:
        raise ValueError("El CSV no coincide con el test esperado.")

    # Localizar los metadatos, sin utilizar sus etiquetas para predecir.
    if args.patients:
        ruta_patients = Path(args.patients)
    else:
        candidatos = sorted((REPO / "data").rglob("patients.csv"))

        if len(candidatos) != 1:
            raise ValueError(
                "Indica --patients con la ruta correcta. Encontrados:\n"
                + "\n".join(str(p) for p in candidatos)
            )

        ruta_patients = candidatos[0]

    pacientes = pd.read_csv(ruta_patients)
    id_patients = identificar_columna_id(pacientes)

    if "dataset" not in pacientes.columns:
        raise ValueError(
            "No existe la columna dataset en patients.csv. "
            f"Columnas: {list(pacientes.columns)}"
        )

    pacientes["patient_id_limpio"] = normalizar_ids(
        pacientes, id_patients
    )

    # Solo se incorpora la procedencia, no otras variables clínicas.
    test = test.merge(
        pacientes[["patient_id_limpio", "dataset"]],
        on="patient_id_limpio",
        how="left",
        validate="one_to_one",
    )

    if test["dataset"].isna().any():
        faltan = test.loc[
            test["dataset"].isna(), "patient_id_limpio"
        ].tolist()
        raise ValueError(f"Falta la cohorte de estas pacientes: {faltan}")

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = ruta.parent / f"analisis_test_{fecha}"
    salida.mkdir(exist_ok=False)

    filas = [{"cohorte": "GLOBAL", **metricas(test, args.umbral)}]

    for cohorte, grupo in test.groupby("dataset", sort=True):
        filas.append({
            "cohorte": str(cohorte),
            **metricas(grupo, args.umbral),
        })

    resumen = pd.DataFrame(filas)
    resumen.to_csv(salida / "metricas_por_cohorte.csv", index=False)

    # Calibración descriptiva: cinco intervalos fijos.
    bordes = np.linspace(0, 1, 6)
    indices = np.digitize(
        test["prob"].to_numpy(), bordes[1:-1]
    )

    grupos_calibracion = []

    for i in range(5):
        grupo = test.loc[indices == i]

        if grupo.empty:
            continue

        grupos_calibracion.append({
            "limite_inferior": bordes[i],
            "limite_superior": bordes[i + 1],
            "pacientes": len(grupo),
            "puntuacion_media": float(grupo["prob"].mean()),
            "fraccion_pCR_1": float(grupo["pCR"].mean()),
        })

    calibracion = pd.DataFrame(grupos_calibracion)
    calibracion.to_csv(salida / "calibracion.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray",
        label="Calibración ideal",
    )

    ax.plot(
        calibracion["puntuacion_media"],
        calibracion["fraccion_pCR_1"],
        "o-", color="#126d86",
        label="CNN en test",
    )

    for _, fila in calibracion.iterrows():
        ax.annotate(
            f'n={int(fila["pacientes"])}',
            (fila["puntuacion_media"], fila["fraccion_pCR_1"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Puntuación media de pCR=1",
        ylabel="Proporción real de pCR=1",
        title="Calibración descriptiva · Test · 176 pacientes",
    )

    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "calibracion.png", dpi=180)
    plt.close(fig)

    brier = brier_score_loss(test["pCR"], test["prob"])

    print("\nRESULTADOS POR COHORTE")
    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nBrier score global: {brier:.6f}")
    print("Menor Brier indica menor error de las puntuaciones.")
    print("No es una medida exclusiva de calibración.")
    print("\nPacientes por intervalo de calibración:")
    print(calibracion.to_string(index=False))
    print(f"\nResultados: {salida.resolve()}")
    print("No se ha entrenado, calibrado ni ejecutado la CNN.")
    print("No se ha modificado el umbral.")


if __name__ == "__main__":
    main()