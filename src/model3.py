#!/usr/bin/env python3
"""CNN propia de cinco etapas, entrenada desde cero.

Entrada: PRE, EARLY, LATE, o representación de realce.
Evaluación por paciente: media de probabilidades de sus cortes.
Selección de época según AUC de validación.
Test únicamente con --test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import preprocessing as pp
from distribucion_probabilidades import dibujar_distribucion


REPO = pp.REPO
CANALES = (16, 32, 64, 128, 128)
ENTRADA = (3, 256, 256)
ATENCION_ETAPA = 4


# ---------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def calcular_mascara(self, x):
        media = x.mean(dim=1, keepdim=True)
        maximo = x.amax(dim=1, keepdim=True)
        resumen = torch.cat((media, maximo), dim=1)
        return self.sigmoid(self.conv(resumen))

    def forward(self, x):
        return x * self.calcular_mascara(x)


def etapa(c_in, c_out, atencion=False):
    capas = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if atencion:
        capas.append(SpatialAttention(7))
    capas.append(nn.MaxPool2d(2))
    return nn.Sequential(*capas)


def representar_entrada(x, representacion):
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            "Se esperaba [N, 3, H, W]: PRE, EARLY, LATE"
        )

    if representacion == "fases":
        return x

    pre = x[:, 0]
    early = x[:, 1]
    late = x[:, 2]

    if representacion == "realce":
        return torch.stack(
            (pre, early - pre, late - early),
            dim=1,
        )

    if representacion == "realce4":
        return torch.stack(
            (
                pre,
                early - pre,
                late - early,
                late - pre,
            ),
            dim=1,
        )

    raise ValueError("Representación desconocida")


class CNNpCR(nn.Module):
    def __init__(
        self,
        canales=CANALES,
        dropout=0.3,
        representacion="fases",
        atencion="spatial",
    ):
        super().__init__()

        if len(canales) != 5 or any(c < 1 for c in canales):
            raise ValueError("Se necesitan cinco canales positivos")
        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en [0,1)")
        if representacion not in ("fases", "realce", "realce4"):
            raise ValueError("Representación desconocida")
        if atencion not in ("ninguna", "spatial"):
            raise ValueError("Atención desconocida")

        self.canales = tuple(canales)
        self.dropout = dropout
        self.representacion = representacion
        self.atencion = atencion

        bloques = []
        c_in = 4 if representacion == "realce4" else 3

        for numero, c_out in enumerate(canales, start=1):
            bloques.append(
                etapa(
                    c_in,
                    c_out,
                    atencion=(
                        atencion == "spatial"
                        and numero == ATENCION_ETAPA
                    ),
                )
            )
            c_in = c_out

        self.extractor = nn.Sequential(*bloques)
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c_in, 1),
        )

        self._inicializar()

    def _inicializar(self):
        for modulo in self.modules():
            if isinstance(modulo, nn.Conv2d):
                nn.init.kaiming_normal_(
                    modulo.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(modulo, nn.BatchNorm2d):
                nn.init.ones_(modulo.weight)
                nn.init.zeros_(modulo.bias)
            elif isinstance(modulo, nn.Linear):
                nn.init.normal_(modulo.weight, std=0.01)
                nn.init.zeros_(modulo.bias)

        for modulo in self.modules():
            if isinstance(modulo, SpatialAttention):
                nn.init.xavier_uniform_(modulo.conv.weight)

    def forward(self, x):
        x = representar_entrada(x, self.representacion)
        return self.cabeza(self.extractor(x))

    def config(self):
        return {
            "version": "cinco_etapas_cinco_poolings_v1",
            "canales": list(self.canales),
            "dropout": self.dropout,
            "entrada": list(ENTRADA),
            "parametros": contar_parametros(self),
            "etapas": 5,
            "max_poolings": 5,
            "pooling_por_etapa": [True] * 5,
            "resolucion_final_hw": [8, 8],
            "representacion": self.representacion,
            "canales_derivados": {
                "fases": ["PRE", "EARLY", "LATE"],
                "realce": ["PRE", "EARLY-PRE", "LATE-EARLY"],
                "realce4": [
                    "PRE",
                    "EARLY-PRE",
                    "LATE-EARLY",
                    "LATE-PRE",
                ],
            }[self.representacion],
            "atencion": self.atencion,
            "atencion_etapa": (
                4 if self.atencion == "spatial" else None
            ),
            "atencion_resolucion_hw": (
                [32, 32] if self.atencion == "spatial" else None
            ),
            "atencion_kernel": (
                7 if self.atencion == "spatial" else None
            ),
        }


def contar_parametros(modelo):
    return sum(
        p.numel()
        for p in modelo.parameters()
        if p.requires_grad
    )


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def fijar_semilla(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as fichero:
        for bloque in iter(
            lambda: fichero.read(1 << 20), b""
        ):
            h.update(bloque)
    return h.hexdigest()


def limpiar_json(valor):
    if isinstance(valor, dict):
        return {
            clave: limpiar_json(dato)
            for clave, dato in valor.items()
        }
    if isinstance(valor, (list, tuple)):
        return [limpiar_json(dato) for dato in valor]
    if isinstance(valor, (float, np.floating)):
        return (
            round(float(valor), 6)
            if np.isfinite(valor)
            else None
        )
    if isinstance(valor, np.integer):
        return int(valor)
    return valor


def guardar_json(ruta, datos):
    ruta.write_text(
        json.dumps(
            limpiar_json(datos),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Métricas por paciente
# ---------------------------------------------------------------------

def metricas_paciente(prob_cortes, filas, umbral=0.3):
    prob = np.asarray(prob_cortes).reshape(-1)

    if len(prob) != len(filas):
        raise ValueError(
            "No coincide el número de probabilidades y cortes"
        )
    if not np.isfinite(prob).all():
        raise ValueError("Probabilidades no finitas")
    if ((prob < 0) | (prob > 1)).any():
        raise ValueError("Probabilidades fuera de [0,1]")

    datos = pd.DataFrame({
        "patient_id": filas.patient_id.to_numpy(),
        "prob": prob,
        "pCR": filas.pCR.to_numpy(),
    })

    if datos.empty:
        raise ValueError("No hay pacientes para evaluar")
    if datos.patient_id.isna().any():
        raise ValueError("Identificadores de paciente vacíos")
    if not datos.pCR.isin([0, 1]).all():
        raise ValueError("Las etiquetas deben ser 0 o 1")
    if (datos.groupby("patient_id").pCR.nunique() > 1).any():
        raise ValueError(
            "Una paciente tiene etiquetas contradictorias"
        )

    tabla = datos.groupby("patient_id").agg(
        prob=("prob", "mean"),
        pCR=("pCR", "first"),
    )

    y = tabla.pCR.to_numpy()
    p = tabla.prob.to_numpy()
    pred = p >= umbral

    vp = int((pred & (y == 1)).sum())
    vn = int((~pred & (y == 0)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())

    def dividir(a, b):
        return a / b if b else float("nan")

    recall = dividir(vp, vp + fn)

    met = {
        "auc": (
            float(roc_auc_score(y, p))
            if len(np.unique(y)) == 2
            else float("nan")
        ),
        "sensibilidad": recall,
        "recall": recall,
        "precision": dividir(vp, vp + fp),
        "especificidad": dividir(vn, vn + fp),
        "accuracy": dividir(vp + vn, len(y)),
        "f1": dividir(2 * vp, 2 * vp + fp + fn),
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn,
    }

    return met, tabla


def umbral_youden(tabla):
    """Calculado exclusivamente con validación."""
    y = tabla.pCR.to_numpy()
    p = tabla.prob.to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError("Youden necesita ambas clases")

    candidatos = np.r_[
        np.nextafter(p.max(), np.inf),
        np.unique(p),
    ]

    mejor_j = -np.inf
    mejor_umbral = 0.5

    for umbral in candidatos:
        pred = p >= umbral
        sens = (pred & (y == 1)).sum() / (y == 1).sum()
        esp = (~pred & (y == 0)).sum() / (y == 0).sum()
        j = sens + esp - 1

        if j > mejor_j:
            mejor_j = j
            mejor_umbral = float(umbral)

    return mejor_umbral


def metricas_tabla(tabla, umbral):
    filas = tabla.reset_index()
    return metricas_paciente(
        filas.prob.to_numpy(),
        filas,
        umbral,
    )[0]


# ---------------------------------------------------------------------
# Entrenamiento y predicción
# ---------------------------------------------------------------------

def entrenar_epoca(
    modelo, loader, criterio, optimizador, dispositivo
):
    modelo.train()
    total = 0.0
    n = 0

    for x, y in loader:
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo).unsqueeze(1)

        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(x), y)

        if not torch.isfinite(perdida):
            raise RuntimeError("Pérdida de train no finita")

        perdida.backward()
        optimizador.step()

        total += perdida.item() * len(y)
        n += len(y)

    if n == 0:
        raise ValueError("DataLoader de train vacío")

    return total / n


@torch.no_grad()
def predecir(modelo, loader, criterio, dispositivo):
    modelo.eval()
    total = 0.0
    n = 0
    probabilidades = []

    for x, y in loader:
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo).unsqueeze(1)

        logits = modelo(x)
        perdida = criterio(logits, y)

        if not torch.isfinite(perdida):
            raise RuntimeError("Pérdida de evaluación no finita")

        total += perdida.item() * len(y)
        n += len(y)

        probabilidades.append(
            torch.sigmoid(logits)
            .squeeze(1)
            .cpu()
            .numpy()
        )

    if n == 0:
        raise ValueError("DataLoader de evaluación vacío")

    return total / n, np.concatenate(probabilidades)


# ---------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------

def dibujar_curvas(hist, mejor, ruta, titulo, umbral):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    comparaciones = [
        (
            axes[0, 0],
            "perdida_train_eval",
            "perdida_val",
            "BCE sin ponderar · modo evaluación",
        ),
        (
            axes[0, 1],
            "auc_train",
            "auc_val",
            "AUC por paciente",
        ),
        (
            axes[1, 0],
            "recall_train",
            "recall_val",
            "Recall por paciente · pCR=1",
        ),
        (
            axes[1, 1],
            "precision_train",
            "precision_val",
            "Precisión por paciente · pCR=1",
        ),
    ]

    for ax, columna_tr, columna_va, etiqueta in comparaciones:
        ax.plot(
            hist.epoca,
            hist[columna_tr],
            "o-",
            markersize=3,
            label="Train sin aumentos",
        )
        ax.plot(
            hist.epoca,
            hist[columna_va],
            "o-",
            markersize=3,
            label="Validación",
        )
        ax.axvline(
            mejor,
            color="gray",
            linestyle=":",
        )
        ax.set_title(etiqueta)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if columna_tr != "perdida_train_eval":
            ax.set_ylim(-0.03, 1.03)

    fig.suptitle(
        f"{titulo}\n"
        f"Umbral {umbral:g} · mejor época por AUC val: {mejor}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def dibujar_matriz(met, ruta, titulo):
    matriz = np.array([
        [met["VN"], met["FP"]],
        [met["FN"], met["VP"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(matriz, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=15,
                color=(
                    "white"
                    if matriz[i, j] > matriz.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks([0, 1], ["Pred. pCR=0", "Pred. pCR=1"])
    ax.set_yticks([0, 1], ["Real pCR=0", "Real pCR=1"])
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# Verificación pequeña
# ---------------------------------------------------------------------

def verificar(args, config_espacial):
    fijar_semilla(args.semilla)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    modelo = CNNpCR(
        canales=tuple(args.canales),
        dropout=0.0,
        representacion=args.representacion,
        atencion=args.atencion,
    ).to(dispositivo)

    modelo.eval()
    with torch.no_grad():
        x = torch.zeros(2, *ENTRADA, device=dispositivo)
        z = representar_entrada(x, args.representacion)

        print("Entrada:", tuple(x.shape))

        for numero, bloque in enumerate(
            modelo.extractor, start=1
        ):
            z = bloque(z)
            print(f"Etapa {numero}:", tuple(z.shape))

        assert z.shape == (2, args.canales[-1], 8, 8)
        assert modelo.cabeza(z).shape == (2, 1)

    del x, z

    pp._requiere_utils()
    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)

    # Solo train del fold: nunca validación ni test.
    train, _ = pp.uc.particion(samples, fold_val=args.fold)
    filas = train.groupby("pCR").head(5)

    if len(filas) != 10 or filas.pCR.nunique() != 2:
        raise ValueError("Se necesitan cinco cortes de cada clase")

    x = torch.stack([
        pp.a_tensor(
            pp.cargar_corte_uint8(fila),
            config_espacial,
        )
        for fila in filas.itertuples(index=False)
    ]).to(dispositivo)

    y = torch.tensor(
        filas.pCR.to_numpy(),
        dtype=torch.float32,
        device=dispositivo,
    ).unsqueeze(1)

    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.BCEWithLogitsLoss()

    print(f"Memorizar 10 cortes en {dispositivo}:")

    for paso in range(1, 151):
        modelo.train()
        optimizador.zero_grad(set_to_none=True)
        perdida = criterio(modelo(x), y)

        if not torch.isfinite(perdida):
            raise RuntimeError("Pérdida no finita")

        perdida.backward()
        optimizador.step()

        if paso in (1, 10, 25, 50, 100, 150):
            print(
                f"Paso {paso}: pérdida {perdida.item():.4f}",
                flush=True,
            )

    modelo.eval()

    with torch.no_grad():
        prob = torch.sigmoid(modelo(x))
        aciertos = int(
            ((prob >= 0.5).float() == y).sum().item()
        )
        perdida_eval = criterio(modelo(x), y).item()

    print(f"Aciertos en modo eval: {aciertos}/10")
    print(f"BCE en modo eval: {perdida_eval:.4f}")
    print("Esta prueba no mide generalización ni guarda pesos.")


# ---------------------------------------------------------------------
# Test opcional
# ---------------------------------------------------------------------

def evaluar_test(
    modelo,
    ruta_pesos,
    umbral,
    args,
    dispositivo,
    carpeta,
    config_espacial,
):
    modelo.load_state_dict(
        torch.load(
            ruta_pesos,
            map_location=dispositivo,
            weights_only=True,
        )
    )

    loader, filas = pp.crear_dataloader_test(
        batch_size=args.batch,
        num_workers=args.workers,
        config_espacial=config_espacial,
    )

    _, prob = predecir(
        modelo,
        loader,
        nn.BCEWithLogitsLoss(),
        dispositivo,
    )

    met, tabla = metricas_paciente(prob, filas, umbral)
    tabla.to_csv(carpeta / "test_pacientes.csv")

    dibujar_matriz(
        met,
        carpeta / "matriz_confusion_test.png",
        f"Test · umbral {umbral:.3f}",
    )
    dibujar_distribucion(
        tabla,
        carpeta / "distribucion_probabilidades_test.png",
        umbral=umbral,
        titulo="Test: distribución por paciente",
    )

    return {
        "pacientes": len(tabla),
        "umbral_fijo": umbral,
        "metricas_umbral_fijo": met,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=10)
    parser.add_argument(
        "--perdida",
        choices=["ponderada", "normal"],
        default="ponderada",
    )
    parser.add_argument(
        "--representacion",
        choices=["fases", "realce", "realce4"],
        default="fases",
    )
    parser.add_argument(
        "--atencion",
        choices=["ninguna", "spatial"],
        default="spatial",
    )
    parser.add_argument("--fold", type=int, choices=range(5), default=1)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--canales", type=int, nargs=5, default=list(CANALES)
    )
    parser.add_argument("--sin-aumento", action="store_true")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--nombre", default=None)
    parser.add_argument("--umbral", type=float, default=0.3)
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--umbral-test",
        choices=["fijo", "youden"],
        default="fijo",
        help="Criterio de umbral elegido antes de consultar test",
    )
    parser.add_argument(
        "--preprocesado-config",
        type=pp.Path,
        default=None,
    )

    args = parser.parse_args()

    if args.epocas < 1 or args.paciencia < 1:
        parser.error("epocas y paciencia deben ser positivas")
    if args.batch < 1 or args.workers < 0:
        parser.error("batch debe ser positivo y workers >= 0")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr debe ser positivo y finito")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay debe ser finito y >= 0")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        parser.error("dropout debe estar en [0,1)")
    if not np.isfinite(args.umbral) or not 0 <= args.umbral <= 1:
        parser.error("umbral debe estar entre 0 y 1")
    if any(c < 1 for c in args.canales):
        parser.error("Los canales deben ser positivos")
    if args.rapido and args.test:
        parser.error("No se permite --test con --rapido")

    config_espacial = (
        pp.ConfigEspacial.desde_json(args.preprocesado_config)
        if args.preprocesado_config
        else pp.ConfigEspacial()
    )

    if args.verificar:
        verificar(args, config_espacial)
        return

    fijar_semilla(args.semilla)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    nombre = args.nombre or (
        f"cnn5_5pool_{args.perdida}_f{args.fold}"
        f"_{args.representacion}_{args.atencion}"
    )

    if not args.nombre and args.rapido:
        nombre += "_rapido"

    if pp.Path(nombre).name != nombre or nombre in (".", ".."):
        parser.error("--nombre debe ser un nombre, no una ruta")

    carpeta = REPO / "resultados" / "entrenamiento" / nombre
    ruta_pesos = REPO / "models" / f"{nombre}.pt"

    if ruta_pesos.exists() or (
        carpeta.exists() and any(carpeta.iterdir())
    ):
        parser.error(
            f"El experimento '{nombre}' ya existe. "
            "Utiliza otro --nombre."
        )

    datos = pp.crear_dataloaders(
        fold_val=args.fold,
        batch_size=args.batch,
        aumentar=not args.sin_aumento,
        num_workers=args.workers,
        semilla=args.semilla,
        max_pacientes=10 if args.rapido else None,
        config_espacial=config_espacial,
    )

    for clave in ("filas_train", "filas_val"):
        if datos[clave].pCR.nunique() != 2:
            raise ValueError(
                f"{clave} necesita ambas clases"
            )

    modelo = CNNpCR(
        canales=tuple(args.canales),
        dropout=args.dropout,
        representacion=args.representacion,
        atencion=args.atencion,
    ).to(dispositivo)

    pos_weight = float(datos["pos_weight"])

    peso_positivo = (
        torch.tensor([pos_weight], device=dispositivo)
        if args.perdida == "ponderada"
        else None
    )

    criterio_train = nn.BCEWithLogitsLoss(
        pos_weight=peso_positivo
    )
    criterio_eval = nn.BCEWithLogitsLoss()

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos.parent.mkdir(parents=True, exist_ok=True)

    print(f"Experimento: {nombre}")
    print("Arquitectura: 5 etapas, 5 MaxPooling, GAP final")
    print(f"Canales: {tuple(args.canales)}")
    print(f"Parámetros: {contar_parametros(modelo):,}")
    print(f"Representación: {args.representacion}")
    print(f"Atención: {args.atencion}")
    print(f"Dispositivo: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    for etiqueta, clave in (
        ("Train", "filas_train"),
        ("Val", "filas_val"),
    ):
        filas = datos[clave]
        print(
            f"{etiqueta}: {len(filas)} cortes / "
            f"{filas.patient_id.nunique()} pacientes"
        )

    print(
        f"Pérdida: {args.perdida} | "
        f"pos_weight calculado: {pos_weight:.3f}"
    )
    print("Preprocesado:", config_espacial.to_dict())
    print(f"Aumento: {datos['aumento']}")
    print(
        f"Épocas: {args.epocas} | paciencia: {args.paciencia} | "
        f"batch: {args.batch} | lr: {args.lr}"
    )
    print(
        "BCE train_eval y val: sin ponderar, "
        "modo eval, sin aumentos."
    )
    print(
        f"Recall y precisión: pCR=1, umbral {args.umbral:g}."
    )
    print(
        "\népoca  BCE_opt  BCE_tr_eval  BCE_val  "
        "AUC_tr  AUC_val  recall_val  prec_val  esp_val"
    )

    historial = []
    mejor_auc = -np.inf
    mejor_epoca = 0
    sin_mejora = 0
    inicio_total = time.time()

    for epoca in range(1, args.epocas + 1):
        inicio = time.time()

        perdida_opt = entrenar_epoca(
            modelo,
            datos["train"],
            criterio_train,
            optimizador,
            dispositivo,
        )

        # Diagnóstico: no realiza actualizaciones de pesos.
        perdida_tr_eval, prob_tr = predecir(
            modelo,
            datos["train_eval"],
            criterio_eval,
            dispositivo,
        )
        met_tr, tabla_tr = metricas_paciente(
            prob_tr,
            datos["filas_train"],
            args.umbral,
        )

        perdida_val, prob_va = predecir(
            modelo,
            datos["val"],
            criterio_eval,
            dispositivo,
        )
        met_va, tabla_va = metricas_paciente(
            prob_va,
            datos["filas_val"],
            args.umbral,
        )

        if not np.isfinite(met_va["auc"]):
            raise RuntimeError("AUC de validación no finita")

        scheduler.step(met_va["auc"])

        historial.append({
            "epoca": epoca,
            "perdida_train_optimizacion": perdida_opt,
            "perdida_train_eval": perdida_tr_eval,
            "perdida_val": perdida_val,
            "auc_train": met_tr["auc"],
            "auc_val": met_va["auc"],
            "recall_train": met_tr["recall"],
            "recall_val": met_va["recall"],
            "precision_train": met_tr["precision"],
            "precision_val": met_va["precision"],
            "esp_train": met_tr["especificidad"],
            "esp_val": met_va["especificidad"],
            "segundos": time.time() - inicio,
        })

        marca = ""

        if met_va["auc"] > mejor_auc:
            mejor_auc = met_va["auc"]
            mejor_epoca = epoca
            sin_mejora = 0

            torch.save(modelo.state_dict(), ruta_pesos)
            tabla_va.to_csv(carpeta / "val_pacientes.csv")
            tabla_tr.to_csv(carpeta / "train_pacientes.csv")
            marca = " * guardado"
        else:
            sin_mejora += 1

        hist = pd.DataFrame(historial)
        hist.to_csv(carpeta / "historial.csv", index=False)

        dibujar_curvas(
            hist,
            mejor_epoca,
            carpeta / "curvas.png",
            nombre,
            args.umbral,
        )

        print(
            f"{epoca:5d} "
            f"{perdida_opt:8.4f} "
            f"{perdida_tr_eval:11.4f} "
            f"{perdida_val:8.4f} "
            f"{met_tr['auc']:7.3f} "
            f"{met_va['auc']:8.3f} "
            f"{met_va['recall']:10.2f} "
            f"{met_va['precision']:9.2f} "
            f"{met_va['especificidad']:8.2f}"
            f"{marca}",
            flush=True,
        )

        if sin_mejora >= args.paciencia:
            print(
                f"\nEarly stopping: {args.paciencia} "
                "épocas sin mejorar AUC val."
            )
            break

    segundos_total = time.time() - inicio_total

    tabla_va = pd.read_csv(
        carpeta / "val_pacientes.csv",
        index_col=0,
    )
    tabla_tr = pd.read_csv(
        carpeta / "train_pacientes.csv",
        index_col=0,
    )

    met_fijo = metricas_tabla(tabla_va, args.umbral)
    met_train = metricas_tabla(tabla_tr, args.umbral)
    youden = umbral_youden(tabla_va)
    met_youden = metricas_tabla(tabla_va, youden)

    for tabla, conjunto in (
        (tabla_tr, "train"),
        (tabla_va, "val"),
    ):
        dibujar_distribucion(
            tabla,
            carpeta / f"distribucion_probabilidades_{conjunto}.png",
            umbral=args.umbral,
            titulo=f"{conjunto}: distribución por paciente",
        )

    dibujar_matriz(
        met_fijo,
        carpeta / "matriz_confusion_val_umbral_fijo.png",
        f"Validación · umbral {args.umbral:g}",
    )
    dibujar_matriz(
        met_youden,
        carpeta / "matriz_confusion_val.png",
        f"Validación · Youden {youden:.3f}",
    )

    config = {
        "experimento": nombre,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": modelo.config(),
        "entrenamiento": {
            "epocas_max": args.epocas,
            "epocas_realizadas": len(historial),
            "mejor_epoca": mejor_epoca,
            "paciencia": args.paciencia,
            "criterio_parada": "AUC por paciente en validación",
            "perdida": args.perdida,
            "pos_weight": (
                pos_weight
                if args.perdida == "ponderada"
                else None
            ),
            "optimizador": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch": args.batch,
            "semilla": args.semilla,
            "fold_validacion": args.fold,
            "aumento": datos["aumento"],
            "preprocesado_espacial": config_espacial.to_dict(),
            "preprocesado_config": (
                str(args.preprocesado_config)
                if args.preprocesado_config
                else None
            ),
            "modo_rapido": args.rapido,
            "agregacion_paciente": "media",
            "umbral_curvas_recall_precision": args.umbral,
            "clase_positiva": "pCR=1",
            "perdida_val_ponderada": False,
            "evaluacion_train_por_epoca": True,
        },
        "train_mejor_epoca": met_train,
        "validacion_mejor_epoca": {
            "umbral_fijo": args.umbral,
            "metricas_umbral_fijo": met_fijo,
            "umbral_youden": youden,
            "con_umbral_youden": met_youden,
        },
        "entorno": {
            "dispositivo": str(dispositivo),
            "gpu": (
                torch.cuda.get_device_name(0)
                if dispositivo.type == "cuda"
                else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sistema": platform.platform(),
            "segundos_entrenamiento": segundos_total,
        },
        "pesos": {
            "ruta": str(ruta_pesos.relative_to(REPO)),
            "sha256": sha256(ruta_pesos),
        },
        "test": None,
    }

    guardar_json(carpeta / "config.json", config)

    if args.test:
        umbral_final = (
            youden
            if args.umbral_test == "youden"
            else args.umbral
        )

        # El criterio se eligió mediante CLI antes de consultar test.
        guardar_json(
            carpeta / "evaluacion_test_iniciada.json",
            {
                "criterio": args.umbral_test,
                "umbral": umbral_final,
                "pesos_sha256": config["pesos"]["sha256"],
            },
        )

        config["test"] = evaluar_test(
            modelo,
            ruta_pesos,
            umbral_final,
            args,
            dispositivo,
            carpeta,
            config_espacial,
        )
        guardar_json(carpeta / "config.json", config)

    print(
        f"\nMejor época: {mejor_epoca} | "
        f"AUC train: {met_train['auc']:.3f} | "
        f"AUC val: {met_fijo['auc']:.3f}"
    )

    for etiqueta, umbral, met in (
        ("Fijo", args.umbral, met_fijo),
        ("Youden", youden, met_youden),
    ):
        print(
            f"{etiqueta} {umbral:.3f} -> "
            f"recall {met['recall']:.2f} | "
            f"precisión {met['precision']:.2f} | "
            f"especificidad {met['especificidad']:.2f}"
        )
        print(
            f"VP {met['VP']} | VN {met['VN']} | "
            f"FP {met['FP']} | FN {met['FN']}"
        )

    print(f"Tiempo: {segundos_total / 60:.1f} min")
    print(f"Pesos: {ruta_pesos}")
    print(f"Resultados: {carpeta}")

    if args.rapido:
        print("Modo rápido: no interpretar como resultado final.")


if __name__ == "__main__":
    main()