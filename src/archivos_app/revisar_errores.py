"""Revisión visual de errores con predicciones guardadas. Sin inferencia."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import preprocessing as pp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--umbral", type=float, default=0.3)
    args = parser.parse_args()

    if not 0 <= args.umbral <= 1:
        parser.error("Umbral fuera de rango.")

    ruta = Path(args.csv)
    tabla = pd.read_csv(ruta)

    if "patient_id" not in tabla.columns:
        primera = str(tabla.columns[0])
        if primera.startswith("Unnamed:"):
            tabla = tabla.rename(columns={tabla.columns[0]: "patient_id"})
        else:
            raise ValueError(
                f"No encuentro patient_id. Columnas: {list(tabla.columns)}"
            )

    if not {"prob", "pCR"}.issubset(tabla.columns):
        raise ValueError("El CSV debe contener prob y pCR.")

    tabla["patient_id"] = tabla["patient_id"].astype("string").str.strip()

    if tabla["patient_id"].isna().any() or tabla["patient_id"].eq("").any():
        raise ValueError("Hay identificadores vacíos.")

    if tabla["patient_id"].duplicated().any():
        raise ValueError("Se requiere una fila por paciente.")

    for columna in ("prob", "pCR"):
        tabla[columna] = pd.to_numeric(tabla[columna], errors="raise")

    if not np.isfinite(tabla[["prob", "pCR"]].to_numpy()).all():
        raise ValueError("Valores no finitos.")

    if not tabla["prob"].between(0, 1).all():
        raise ValueError("Puntuaciones fuera de [0, 1].")

    if not tabla["pCR"].isin([0, 1]).all():
        raise ValueError("Etiquetas distintas de 0 y 1.")

    # Leer la configuración guardada, no la configuración actual de la app.
    ruta_config = ruta.parent.parent / "config.json"
    config = json.loads(ruta_config.read_text(encoding="utf-8"))

    if config["arquitectura"]["representacion"] != "fases":
        raise ValueError("Este script está preparado para representación fases.")

    cfg = pp.ConfigEspacial(
        **config["entrenamiento"]["preprocesado_espacial"]
    )

    tabla["prediccion"] = (tabla["prob"] >= args.umbral).astype(int)

    tabla["tipo"] = np.select(
        [
            (tabla["pCR"] == 0) & (tabla["prediccion"] == 1),
            (tabla["pCR"] == 1) & (tabla["prediccion"] == 0),
        ],
        ["FP", "FN"],
        default="acierto",
    )

    fp = tabla.loc[tabla["tipo"] == "FP"].sort_values(
        ["prob", "patient_id"], ascending=[False, True]
    ).head(2)

    fn = tabla.loc[tabla["tipo"] == "FN"].sort_values(
        ["prob", "patient_id"], ascending=[True, True]
    ).head(2)

    seleccion = pd.concat([fp, fn], ignore_index=True)

    if seleccion.empty:
        raise ValueError("No hay errores con este umbral.")

    # Leer el índice de imágenes sin crear dataloaders ni cargar pesos.
    pp._requiere_utils()
    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS).copy()
    samples["patient_id"] = samples["patient_id"].astype(str)

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = REPO / "resultados" / "revision_errores" / fecha
    salida.mkdir(parents=True, exist_ok=False)

    tabla.loc[tabla["tipo"] != "acierto"].to_csv(
        salida / "todos_los_errores.csv", index=False
    )
    seleccion.to_csv(salida / "pacientes_seleccionadas.csv", index=False)

    registros = []

    for paciente in seleccion.itertuples(index=False):
        filas = samples.loc[
            samples["patient_id"] == paciente.patient_id
        ].sort_values("sample_id")

        if filas.empty:
            raise ValueError(f"No hay cortes de {paciente.patient_id}")

        if not filas["split"].eq("test").all():
            raise ValueError(
                f"{paciente.patient_id}: sus cortes no son todos de test."
            )

        etiquetas = pd.to_numeric(filas["pCR"], errors="raise")
        if not etiquetas.eq(paciente.pCR).all():
            raise ValueError("No coinciden las etiquetas del índice y del CSV.")

        # Inicio, centro y final de los cortes disponibles.
        posiciones = np.unique(
            np.linspace(0, len(filas) - 1, min(3, len(filas))).astype(int)
        )

        for posicion in posiciones:
            fila = filas.iloc[int(posicion)]

            originales = pp.cargar_corte_uint8(
                fila, pp.RAIZ_DATOS
            )

            espaciales, informe_espacial = pp.procesar_espacial(
                originales, cfg
            )

            procesadas, informe_intensidad = pp.procesar_intensidad(
                espaciales, cfg
            )

            fig, ejes = plt.subplots(2, 3, figsize=(11, 7))

            for canal, fase in enumerate(pp.FASES):
                ejes[0, canal].imshow(
                    originales[canal],
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )
                ejes[0, canal].set_title(f"{fase} · original [0, 255]")

                ejes[1, canal].imshow(
                    procesadas[canal],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                ejes[1, canal].set_title(
                    f"{fase} · entrada CNN [0, 1]"
                )

                ejes[0, canal].axis("off")
                ejes[1, canal].axis("off")

            fig.suptitle(
                f"{paciente.tipo} · {paciente.patient_id} · {fila.sample_id}\n"
                f"Real: pCR={int(paciente.pCR)} | "
                f"Predicción paciente: pCR={paciente.prediccion} | "
                f"Puntuación paciente: {paciente.prob:.4f} | "
                f"Umbral: {args.umbral:g}"
            )

            fig.tight_layout(rect=[0, 0, 1, 0.90])

            nombre = f"{paciente.tipo}_{fila.sample_id}.png"
            fig.savefig(salida / nombre, dpi=150)
            plt.close(fig)

            registros.append({
                "patient_id": paciente.patient_id,
                "sample_id": str(fila.sample_id),
                "tipo": paciente.tipo,
                "figura": nombre,
                "informe_espacial": informe_espacial,
                "informe_intensidad": informe_intensidad,
            })

            print(f"Guardado: {nombre}")

    def convertir(valor):
        if isinstance(valor, np.ndarray):
            return valor.tolist()
        if isinstance(valor, np.generic):
            return valor.item()
        if isinstance(valor, Path):
            return str(valor)
        raise TypeError(f"No serializable: {type(valor)}")

    (salida / "revision.json").write_text(
        json.dumps(
            {
                "config_preprocesado": config["entrenamiento"][
                    "preprocesado_espacial"
                ],
                "criterio_seleccion": (
                    "Dos FP con mayor puntuación y dos FN con menor puntuación"
                ),
                "advertencia": (
                    "Selección ilustrativa. La puntuación mostrada es "
                    "por paciente, no una predicción individual del corte."
                ),
                "inferencia_repetida": False,
                "cortes": registros,
            },
            ensure_ascii=False,
            indent=2,
            default=convertir,
        ),
        encoding="utf-8",
    )

    print(f"\nResultados: {salida.resolve()}")
    print("No se han cargado pesos ni repetido predicciones.")


if __name__ == "__main__":
    main()