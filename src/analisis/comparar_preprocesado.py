"""Compara preprocesados usando solo train del fold elegido."""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import preprocessing as pp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), default=1)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument(
        "--salida",
        type=Path,
        default=pp.REPO / "resultados" / "revision_otsu",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n debe ser positivo")

    pp._requiere_utils()
    config = pp.ConfigEspacial.desde_json(args.config)

    if config.registrar:
        parser.error(
            "Usa registrar=false para comparar únicamente "
            "el recorte y la intensidad."
        )

    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)
    filas_train, _ = pp.uc.particion(
        samples,
        fold_val=args.fold,
    )

    if filas_train.empty:
        raise ValueError("La partición de entrenamiento está vacía")

    if not filas_train["split"].eq("train").all():
        raise ValueError(
            "La partición incluye filas fuera del conjunto train"
        )

    # Selecciona un corte por paciente, sin utilizar su etiqueta.
    seleccion = filas_train.groupby(
        "patient_id",
        group_keys=False,
    ).sample(n=1, random_state=args.semilla)

    seleccion = seleccion.sample(
        n=min(args.n, len(seleccion)),
        random_state=args.semilla,
    )

    variantes = [
        ("Original /255", pp.ConfigEspacial()),
        (
            "Recorte fijo",
            replace(
                config,
                metodo_roi="fijo",
                normalizacion="escala_comun",
                clahe=False,
            ),
        ),
        (
            "Otsu",
            replace(
                config,
                metodo_roi="otsu",
                normalizacion="escala_comun",
                clahe=False,
            ),
        ),
        (
            "Otsu + truncamiento",
            replace(
                config,
                metodo_roi="otsu",
                normalizacion="truncamiento_comun",
                clahe=False,
            ),
        ),
        (
            "Otsu + CLAHE 1",
            replace(
                config,
                metodo_roi="otsu",
                normalizacion="escala_comun",
                clahe=True,
                clahe_clip=1.0,
            ),
        ),
        (
            "Otsu + CLAHE 2",
            replace(
                config,
                metodo_roi="otsu",
                normalizacion="escala_comun",
                clahe=True,
                clahe_clip=2.0,
            ),
        ),
    ]

    salida = args.salida

    if salida.exists():
        fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        salida = salida.with_name(f"{salida.name}_{fecha}")

    salida.mkdir(parents=True)

    informes = []

    for numero, fila in enumerate(
        seleccion.itertuples(index=False)
    ):
        originales = pp.cargar_corte_uint8(fila)
        resultados = []

        for nombre, configuracion in variantes:
            espacial, informe_espacial = pp.procesar_espacial(
                originales,
                configuracion,
            )
            imagenes, informe_intensidad = pp.procesar_intensidad(
                espacial,
                configuracion,
            )

            resultados.append({
                "nombre": nombre,
                "imagenes": imagenes,
                "espacial": informe_espacial,
                "intensidad": informe_intensidad,
            })

        fig, axes = plt.subplots(
            3,
            len(variantes),
            figsize=(19, 10),
        )

        for columna, resultado in enumerate(resultados):
            recorte = resultado["espacial"]["recorte"]

            detalle = (
                f"escala {recorte['escala']:.2f}"
                if recorte
                else "sin recorte"
            )

            for fase, nombre_fase in enumerate(pp.FASES):
                ax = axes[fase, columna]

                ax.imshow(
                    resultado["imagenes"][fase],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                ax.axis("off")

                if fase == 0:
                    ax.set_title(
                        f"{resultado['nombre']}\n{detalle}",
                        fontsize=9,
                    )

                if columna == 0:
                    ax.text(
                        -0.05,
                        0.5,
                        nombre_fase,
                        transform=ax.transAxes,
                        rotation=90,
                        ha="right",
                        va="center",
                        fontsize=11,
                    )

                    # Muestra ambas cajas sobre la imagen original.
                    for indice, color in ((1, "cyan"), (2, "orange")):
                        caja = resultados[indice]["espacial"]["recorte"]

                        coordenadas = (
                            caja["bbox_xyxy"]
                            if caja
                            else [0, 0, 256, 256]
                        )
                        x0, y0, x1, y1 = coordenadas

                        ax.add_patch(
                            Rectangle(
                                (x0, y0),
                                x1 - x0,
                                y1 - y0,
                                fill=False,
                                edgecolor=color,
                                linewidth=1,
                            )
                        )

        fig.suptitle(
            f"{fila.sample_id}\n"
            "Original: caja cyan = fijo; naranja = Otsu. "
            "Revisar si se pierde tejido.",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        ruta_figura = salida / f"corte_{numero:03d}.png"
        fig.savefig(ruta_figura, dpi=120)
        plt.close(fig)

        informes.append({
            "sample_id": str(fila.sample_id),
            "patient_id": str(fila.patient_id),
            "figura": ruta_figura.name,
            "variantes": {
                resultado["nombre"]: {
                    "espacial": resultado["espacial"],
                    "intensidad": resultado["intensidad"],
                }
                for resultado in resultados
            },
        })

        print(
            f"{numero + 1}/{len(seleccion)}: {fila.sample_id}",
            flush=True,
        )

    resumen = {
        "fold": args.fold,
        "cortes_revisados": len(informes),
        "seleccion": "Un corte por paciente del train del fold",
        "test_consultado": False,
        "cortes": informes,
    }

    (salida / "informe.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nComparación completada. Resultados: {salida}")


if __name__ == "__main__":
    main()