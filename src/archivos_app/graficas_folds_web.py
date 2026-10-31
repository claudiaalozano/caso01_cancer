"""Genera gráficas de los cinco folds para la web. No entrena modelos."""

from pathlib import Path
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "resultados" / "entrenamiento"
SALIDA = REPO / "app" / "figuras"

COLORES = [
    "#66eeee",
    "#b69aff",
    "#ffb86b",
    "#ff79ad",
    "#95e38c",
]

COLUMNAS = [
    "epoca",
    "perdida_train_eval",
    "perdida_val",
    "auc_train",
    "auc_val",
]


def cargar_historial(fold):
    ruta = BASE / f"cnn5_relu_spatial_cv_f{fold}" / "historial.csv"

    if not ruta.exists():
        raise FileNotFoundError(f"Falta el historial: {ruta}")

    with ruta.open(encoding="utf-8-sig", newline="") as archivo:
        lector = csv.DictReader(archivo)
        faltan = set(COLUMNAS) - set(lector.fieldnames or [])

        if faltan:
            raise ValueError(f"{ruta}: faltan columnas {sorted(faltan)}")

        filas = []

        for fila in lector:
            valores = {nombre: float(fila[nombre]) for nombre in COLUMNAS}

            if not all(math.isfinite(valor) for valor in valores.values()):
                raise ValueError(f"{ruta}: contiene valores no finitos")

            filas.append(valores)

    if not filas:
        raise ValueError(f"Historial vacío: {ruta}")

    filas.sort(key=lambda fila: fila["epoca"])
    epocas = [fila["epoca"] for fila in filas]

    if len(set(epocas)) != len(epocas):
        raise ValueError(f"{ruta}: contiene épocas duplicadas")

    return {
        nombre: [fila[nombre] for fila in filas]
        for nombre in COLUMNAS
    }


def dibujar(historiales, columna_train, columna_val, titulo, eje_y, nombre):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    fig.patch.set_facecolor("#09111e")
    ax.set_facecolor("#0e192a")

    for fold, historial in enumerate(historiales):
        color = COLORES[fold]

        ax.plot(
            historial["epoca"],
            historial[columna_train],
            color=color,
            linewidth=1.8,
            linestyle="-",
            alpha=0.85,
        )

        ax.plot(
            historial["epoca"],
            historial[columna_val],
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.95,
        )

    ax.set_title(titulo, color="#edf3ff", fontsize=16, pad=18)
    ax.set_xlabel("Época", color="#bcc9df")
    ax.set_ylabel(eje_y, color="#bcc9df")
    ax.tick_params(colors="#bcc9df")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, color="#ffffff", alpha=0.08)
    ax.set_axisbelow(True)

    for borde in ax.spines.values():
        borde.set_color("#34435a")

    if columna_val == "auc_val":
        ax.set_ylim(0, 1)
        ax.axhline(
            0.5,
            color="#bcc9df",
            linewidth=1,
            linestyle=":",
            alpha=0.6,
        )

    leyenda = [
        Line2D([0], [0], color=color, linewidth=2, label=f"Fold {fold}")
        for fold, color in enumerate(COLORES)
    ]

    leyenda.extend([
        Line2D(
            [0], [0],
            color="#edf3ff",
            linewidth=2,
            linestyle="-",
            label="Train · evaluación",
        ),
        Line2D(
            [0], [0],
            color="#edf3ff",
            linewidth=2,
            linestyle="--",
            label="Validación",
        ),
    ])

    ax.legend(
        handles=leyenda,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        facecolor="#0e192a",
        edgecolor="#34435a",
        labelcolor="#edf3ff",
        fontsize=9,
    )

    fig.text(
        0.5,
        0.02,
        "Cada curva termina en su última época registrada. "
        "No se prolongan ni rellenan los folds.",
        ha="center",
        color="#bcc9df",
        fontsize=9,
    )

    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.09, right=0.97)

    for extension in ("svg", "png"):
        destino = SALIDA / f"{nombre}.{extension}"
        fig.savefig(destino, dpi=180, facecolor=fig.get_facecolor())
        print(f"Guardado: {destino}")

    plt.close(fig)


def main():
    # Cargar todos antes de crear las gráficas.
    historiales = [cargar_historial(fold) for fold in range(5)]
    SALIDA.mkdir(parents=True, exist_ok=True)

    dibujar(
        historiales,
        "perdida_train_eval",
        "perdida_val",
        "Pérdida · cinco folds superpuestos",
        "Pérdida registrada",
        "folds_perdida",
    )

    dibujar(
        historiales,
        "auc_train",
        "auc_val",
        "AUC por paciente · cinco folds superpuestos",
        "AUC",
        "folds_auc",
    )

    print("\nCompletado. No se han cargado modelos ni consultado el test.")


if __name__ == "__main__":
    main()