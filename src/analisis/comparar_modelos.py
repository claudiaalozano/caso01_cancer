import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[1]


def cargar_predicciones(ruta):
    tabla = pd.read_csv(ruta)

    requeridas = {"patient_id", "prob", "pCR"}
    if not requeridas.issubset(tabla.columns):
        raise ValueError("Faltan patient_id, prob o pCR")

    if tabla.empty or tabla.patient_id.isna().any():
        raise ValueError("Pacientes vacías")

    if tabla.patient_id.duplicated().any():
        raise ValueError("Hay pacientes duplicadas")

    if not tabla.pCR.isin([0, 1]).all():
        raise ValueError("Etiquetas inválidas")

    prob = tabla.prob.to_numpy(dtype=float)

    if not np.isfinite(prob).all():
        raise ValueError("Probabilidades no finitas")

    if ((prob < 0) | (prob > 1)).any():
        raise ValueError("Probabilidades fuera de [0,1]")

    if tabla.pCR.nunique() != 2:
        raise ValueError("Se necesitan ambas clases")

    return tabla


def metricas(tabla, umbral):
    y = tabla.pCR.to_numpy(dtype=int)
    prob = tabla.prob.to_numpy(dtype=float)
    pred = prob >= umbral

    vp = int((pred & (y == 1)).sum())
    vn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())

    def dividir(a, b):
        return a / b if b else np.nan

    recall_0 = dividir(vn, vn + fp)
    recall_1 = dividir(vp, vp + fn)

    p = np.clip(prob, 1e-7, 1 - 1e-7)
    bce = -np.mean(
        y * np.log(p) + (1 - y) * np.log(1 - p)
    )

    return {
        "recall_0": recall_0,
        "recall_1": recall_1,
        "precision_0": dividir(vn, vn + fn),
        "precision_1": dividir(vp, vp + fp),
        "balanceada": (recall_0 + recall_1) / 2,
        "menor_recall": min(recall_0, recall_1),
        "auc_val": roc_auc_score(y, prob),
        "bce_val_paciente": bce,
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn,
    }


def elegir_umbral(tabla):
    """Youden: maximiza la exactitud balanceada en validación."""
    candidatos = np.r_[
        np.nextafter(tabla.prob.max(), np.inf),
        np.unique(tabla.prob),
    ]

    y = tabla.pCR.to_numpy(dtype=int)
    prob = tabla.prob.to_numpy(dtype=float)

    mejor = None
    mejor_clave = None

    for umbral in candidatos:
        pred = prob >= umbral
        recall_1 = (pred & (y == 1)).sum() / (y == 1).sum()
        recall_0 = (~pred & (y == 0)).sum() / (y == 0).sum()

        # En caso de empate, favorece el menor recall más alto.
        clave = (
            (recall_0 + recall_1) / 2,
            min(recall_0, recall_1),
        )

        if mejor_clave is None or clave > mejor_clave:
            mejor_clave = clave
            mejor = float(umbral)

    return mejor


def firma_pacientes(tabla):
    """Agrupa solo conjuntos con las mismas pacientes y etiquetas."""
    ordenada = tabla.sort_values("patient_id")
    return tuple(
        zip(
            ordenada.patient_id.astype(str),
            ordenada.pCR.astype(int),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carpeta",
        type=Path,
        default=REPO / "resultados" / "entrenamiento",
    )
    parser.add_argument(
        "--umbral",
        type=float,
        default=None,
        help="Si se omite, calcula Youden para cada modelo",
    )
    args = parser.parse_args()

    if args.umbral is not None and not 0 <= args.umbral <= 1:
        parser.error("--umbral debe estar entre 0 y 1")

    grupos = {}
    resultados = []
    avisos = []

    for carpeta in sorted(args.carpeta.iterdir()):
        if not carpeta.is_dir():
            continue

        ruta_val = carpeta / "val_pacientes.csv"
        ruta_config = carpeta / "config.json"

        if not ruta_val.exists() or not ruta_config.exists():
            continue

        try:
            config = json.loads(
                ruta_config.read_text(encoding="utf-8")
            )
            entrenamiento = config.get("entrenamiento", {})

            if entrenamiento.get("modo_rapido", False):
                avisos.append(f"{carpeta.name}: omitido, modo rápido")
                continue

            tabla = cargar_predicciones(ruta_val)
            firma = firma_pacientes(tabla)

            if firma not in grupos:
                grupos[firma] = len(grupos) + 1

            umbral = (
                args.umbral
                if args.umbral is not None
                else elegir_umbral(tabla)
            )

            fila = {
                "grupo_validacion": grupos[firma],
                "modelo": carpeta.name,
                "pacientes": len(tabla),
                "fold": entrenamiento.get("fold_validacion"),
                "epoca_guardada": entrenamiento.get("mejor_epoca"),
                "umbral": umbral,
                **metricas(tabla, umbral),
                "auc_train": np.nan,
                "brecha_auc": np.nan,
                "bce_train_paciente": np.nan,
                "brecha_bce": np.nan,
            }

            ruta_train = carpeta / "train_pacientes.csv"

            if ruta_train.exists():
                tr = cargar_predicciones(ruta_train)

                if set(tr.patient_id) & set(tabla.patient_id):
                    raise ValueError("Solape entre train y validación")

                met_tr = metricas(tr, umbral)

                fila["auc_train"] = met_tr["auc_val"]
                fila["brecha_auc"] = (
                    fila["auc_train"] - fila["auc_val"]
                )
                fila["bce_train_paciente"] = (
                    met_tr["bce_val_paciente"]
                )
                fila["brecha_bce"] = (
                    fila["bce_val_paciente"]
                    - fila["bce_train_paciente"]
                )

            resultados.append(fila)

        except Exception as error:
            avisos.append(f"{carpeta.name}: {error}")

    if not resultados:
        print("No se encontraron experimentos completos.")
        for aviso in avisos:
            print(aviso)
        return

    resumen = pd.DataFrame(resultados).sort_values(
        ["grupo_validacion", "balanceada", "bce_val_paciente"],
        ascending=[True, False, True],
    )

    salida = args.carpeta / "comparacion_modelos.csv"
    resumen.to_csv(salida, index=False)

    columnas = [
        "modelo",
        "epoca_guardada",
        "umbral",
        "recall_0",
        "recall_1",
        "precision_0",
        "precision_1",
        "balanceada",
        "auc_val",
        "bce_val_paciente",
        "brecha_auc",
    ]

    for grupo, datos in resumen.groupby("grupo_validacion"):
        print(
            f"\nGRUPO {grupo}: mismas pacientes y etiquetas "
            f"({datos.iloc[0]['pacientes']} pacientes)"
        )
        print(
            datos[columnas].round(3).to_string(index=False)
        )

    if avisos:
        print("\nAVISOS:")
        for aviso in avisos:
            print("-", aviso)

    print(f"\nResumen guardado: {salida}")
    print(
        "\nLa ordenación no elige automáticamente el modelo final. "
        "Los umbrales elegidos en esta misma validación pueden "
        "dar resultados optimistas."
    )
    print(
        "Las BCE son por paciente, sin ponderar. Pueden diferir "
        "de las curvas originales, calculadas por corte."
    )
    print(
        "No se han evaluado pesos ni consultado predicciones de test."
    )


if __name__ == "__main__":
    main()