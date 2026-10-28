"""Auditoría visual y de intensidad por procedencia. No evalúa test."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import preprocessing as pp


def identificar_cohorte(patient_id):
    nombre = str(patient_id)

    for prefijo in ("ISPY1", "ISPY2", "ACRIN", "Breast_MRI"):
        if nombre.startswith(prefijo):
            return prefijo

    return "Otra"


def estadisticas(imagen):
    valores = imagen[imagen > 0]

    return {
        "media": float(imagen.mean()),
        "desviacion": float(imagen.std()),
        "fraccion_cero": float((imagen == 0).mean()),
        "fraccion_255": float((imagen >= 1).mean()),
        "p50_no_cero": (
            float(np.percentile(valores, 50))
            if valores.size else np.nan
        ),
        "p99_no_cero": (
            float(np.percentile(valores, 99))
            if valores.size else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=1)
    parser.add_argument("--pacientes", type=int, default=20)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    if args.pacientes < 1:
        parser.error("--pacientes debe ser positivo")

    pp._requiere_utils()
    config = pp.ConfigEspacial.desde_json(args.config)

    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)
    train, _ = pp.uc.particion(samples, fold_val=args.fold)

    if train.empty or not train["split"].eq("train").all():
        raise ValueError("Partición de train inválida")

    train = train.copy()
    train["cohorte"] = train.patient_id.map(identificar_cohorte)

    # Un corte aleatorio por paciente, sin seleccionar por pCR.
    candidatos = train.groupby(
        "patient_id", group_keys=False
    ).sample(n=1, random_state=args.semilla)

    seleccion = pd.concat([
        grupo.sample(
            n=min(args.pacientes, len(grupo)),
            random_state=args.semilla,
        )
        for _, grupo in candidatos.groupby("cohorte")
    ]).reset_index(drop=True)

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = pp.REPO / "resultados" / f"auditoria_cohortes_{fecha}"
    salida.mkdir(parents=True)

    registros = []
    informes = []
    errores = []

    for numero, fila in enumerate(
        seleccion.itertuples(index=False), start=1
    ):
        try:
            raw = pp.cargar_corte_uint8(fila)
            original = raw.astype(np.float32) / 255.0

            espacial, informe_espacial = pp.procesar_espacial(
                raw, config
            )
            procesada, informe_intensidad = pp.procesar_intensidad(
                espacial, config
            )

            recorte = informe_espacial["recorte"]
            escala = recorte["escala"] if recorte else 1.0
            caja = (
                recorte["bbox_xyxy"]
                if recorte else [0, 0, 256, 256]
            )
            x0, y0, x1, y1 = caja

            # Medida auxiliar: señal descartada respecto al campo
            # posterior al registro y anterior al recorte.
            # No equivale a cantidad de tejido ni de tumor.
            referencia = raw.copy()

            for indice, fase in ((1, "EARLY"), (2, "LATE")):
                registro = informe_espacial["registro"].get(fase, {})
                if registro.get("aplicado"):
                    referencia[indice] = pp._mover(
                        raw[indice],
                        *registro["desplazamiento_yx"],
                    )

            dentro = np.zeros((256, 256), dtype=bool)
            dentro[y0:y1, x0:x1] = True

            for indice, fase in enumerate(pp.FASES):
                total = referencia[indice].astype(float).sum()
                fuera = referencia[indice][~dentro].astype(float).sum()
                fraccion_fuera = fuera / total if total else 0.0

                for estado, imagen in (
                    ("original", original[indice]),
                    ("procesada", procesada[indice]),
                ):
                    registros.append({
                        "patient_id": fila.patient_id,
                        "sample_id": fila.sample_id,
                        "cohorte": fila.cohorte,
                        "fase": fase,
                        "estado": estado,
                        "escala_recorte": escala,
                        "fraccion_senal_fuera_caja": fraccion_fuera,
                        **estadisticas(imagen),
                    })

            carpeta = salida / fila.cohorte
            carpeta.mkdir(exist_ok=True)

            fig, axes = plt.subplots(2, 3, figsize=(10, 7))

            for indice, fase in enumerate(pp.FASES):
                axes[0, indice].imshow(
                    original[indice], cmap="gray", vmin=0, vmax=1
                )
                axes[0, indice].add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="orange",
                        linewidth=1,
                    )
                )
                axes[0, indice].set_title(f"{fase} · original")

                axes[1, indice].imshow(
                    procesada[indice], cmap="gray", vmin=0, vmax=1
                )
                axes[1, indice].set_title(f"{fase} · procesada")

            for ax in axes.flat:
                ax.axis("off")

            fig.suptitle(
                f"{fila.cohorte} · {fila.sample_id}\n"
                f"Caja naranja: recorte | escala {escala:.3f}"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            fig.savefig(
                carpeta / f"corte_{numero:03d}.png", dpi=120
            )
            plt.close(fig)

            informes.append({
                "sample_id": fila.sample_id,
                "cohorte": fila.cohorte,
                "espacial": informe_espacial,
                "intensidad": informe_intensidad,
            })

            print(
                f"{numero}/{len(seleccion)}: {fila.sample_id}",
                flush=True,
            )

        except Exception as error:
            plt.close("all")
            errores.append({
                "sample_id": fila.sample_id,
                "error": str(error),
            })
            print(f"ERROR en {fila.sample_id}: {error}", flush=True)

    if registros:
        detalle = pd.DataFrame(registros)
        detalle.to_csv(salida / "detalle.csv", index=False)

        columnas = [
            "media",
            "desviacion",
            "fraccion_cero",
            "fraccion_255",
            "p50_no_cero",
            "p99_no_cero",
            "escala_recorte",
            "fraccion_senal_fuera_caja",
        ]

        resumen = detalle.groupby(
            ["cohorte", "fase", "estado"]
        )[columnas].median()

        resumen.to_csv(salida / "resumen_cohortes.csv")

    seleccion[
        ["patient_id", "sample_id", "cohorte"]
    ].to_csv(salida / "seleccion.csv", index=False)

    (salida / "informe.json").write_text(
        json.dumps(
            {
                "fold": args.fold,
                "test_consultado": False,
                "aumentos_aplicados": False,
                "config": config.to_dict(),
                "cortes_completados": len(informes),
                "errores": errores,
                "cortes": informes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nResultados: {salida}")
    print(f"Cortes completados: {len(informes)}")
    print(f"Errores: {len(errores)}")


if __name__ == "__main__":
    main()