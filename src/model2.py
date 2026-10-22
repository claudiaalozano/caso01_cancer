#!/usr/bin/env python3
"""CNN de cuatro etapas para predecir pCR.

Entrada:
    PRE, EARLY y LATE, o representación de realce.

Arquitectura:
    Etapa 1: Conv-BN-ReLU                 -> 16 x 256 x 256
    Etapa 2: Conv-BN-ReLU-MaxPool         -> 32 x 128 x 128
    Etapa 3: Conv-BN-ReLU-MaxPool         -> 64 x 64 x 64
    Etapa 4: Conv-BN-ReLU-Spatial-MaxPool -> 128 x 32 x 32
    GAP-Flatten-Dropout-Linear           -> 1 logit

La atención espacial actúa antes del último pooling, en mapas 64x64.
No garantiza localizar el tumor.

Validación por paciente: media de probabilidades de sus cortes.
Mejor época y early stopping según AUC de validación.
Test solo cuando se especifica --test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocessing as pp

REPO = pp.REPO

CANALES = (16, 32, 64, 128)
DROPOUT = 0.3
ENTRADA = (3, 256, 256)


# =====================================================================
# ATENCIÓN ESPACIAL
# =====================================================================

class SpatialAttention(nn.Module):
    """Aprende una máscara espacial compartida por los canales."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size debe ser 3 o 7")

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def calcular_mascara(self, x: torch.Tensor) -> torch.Tensor:
        media = torch.mean(x, dim=1, keepdim=True)
        maximo = torch.amax(x, dim=1, keepdim=True)
        resumen = torch.cat((media, maximo), dim=1)

        return self.sigmoid(self.conv(resumen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.calcular_mascara(x)


# =====================================================================
# ARQUITECTURA
# =====================================================================

def etapa(
    c_in: int,
    c_out: int,
    pooling: bool,
    atencion: bool = False,
) -> nn.Sequential:
    capas = [
        nn.Conv2d(
            c_in,
            c_out,
            kernel_size=3,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]

    if atencion:
        capas.append(SpatialAttention(kernel_size=7))

    if pooling:
        capas.append(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    return nn.Sequential(*capas)


def representar_entrada(
    x: torch.Tensor,
    representacion: str = "fases",
) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            "Se esperaba [N, 3, H, W] en orden PRE, EARLY, LATE"
        )

    if representacion == "fases":
        return x

    if representacion == "realce":
        pre = x[:, 0]
        early = x[:, 1]
        late = x[:, 2]

        return torch.stack(
            (pre, early - pre, late - early),
            dim=1,
        )

    raise ValueError(
        f"Representación desconocida: {representacion}"
    )


class CNNpCR(nn.Module):
    def __init__(
        self,
        canales: tuple[int, ...] = CANALES,
        dropout: float = DROPOUT,
        representacion: str = "fases",
        atencion: str = "spatial",
    ):
        super().__init__()

        if len(canales) != 4 or any(c < 1 for c in canales):
            raise ValueError(
                "Se necesitan cuatro cantidades de canales positivas"
            )

        if representacion not in ("fases", "realce"):
            raise ValueError(
                "representacion debe ser fases o realce"
            )

        if atencion not in ("ninguna", "spatial"):
            raise ValueError(
                "atencion debe ser ninguna o spatial"
            )

        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar en [0, 1)")

        self.canales = tuple(canales)
        self.dropout = dropout
        self.representacion = representacion
        self.atencion = atencion

        capas = []
        c_in = ENTRADA[0]

        for numero, c_out in enumerate(self.canales, start=1):
            capas.append(
                etapa(
                    c_in,
                    c_out,
                    pooling=numero > 1,
                    atencion=(
                        atencion == "spatial" and numero == 4
                    ),
                )
            )
            c_in = c_out

        self.extractor = nn.Sequential(*capas)

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
                nn.init.normal_(
                    modulo.weight,
                    mean=0.0,
                    std=0.01,
                )
                nn.init.zeros_(modulo.bias)

        for modulo in self.modules():
            if isinstance(modulo, SpatialAttention):
                nn.init.xavier_uniform_(modulo.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = representar_entrada(x, self.representacion)
        return self.cabeza(self.extractor(x))

    def config(self) -> dict:
        return {
            "version": "cuatro_etapas_tres_poolings_v1",
            "canales": list(self.canales),
            "dropout": self.dropout,
            "entrada": list(ENTRADA),
            "parametros": contar_parametros(self),
            "etapas": 4,
            "max_poolings": 3,
            "pooling_por_etapa": [False, True, True, True],
            "resolucion_final_hw": [32, 32],
            "representacion": self.representacion,
            "canales_derivados": (
                ["PRE", "EARLY-PRE", "LATE-EARLY"]
                if self.representacion == "realce"
                else ["PRE", "EARLY", "LATE"]
            ),
            "atencion": self.atencion,
            "atencion_etapa": (
                4 if self.atencion == "spatial" else None
            ),
            "atencion_resolucion_hw": (
                [64, 64] if self.atencion == "spatial" else None
            ),
            "atencion_kernel": (
                7 if self.atencion == "spatial" else None
            ),
        }


def contar_parametros(modelo: nn.Module) -> int:
    return sum(
        parametro.numel()
        for parametro in modelo.parameters()
        if parametro.requires_grad
    )


def resumen(modelo: CNNpCR) -> None:
    modelo.eval()
    x = torch.zeros(1, *ENTRADA)

    print(f"{'Capa':<34}{'Salida':<22}{'Parámetros':>12}")
    print("-" * 68)
    print(
        f"{'Entrada':<34}"
        f"{' x '.join(map(str, ENTRADA)):<22}"
    )

    with torch.no_grad():
        for numero, bloque in enumerate(modelo.extractor, start=1):
            for modulo in bloque:
                if isinstance(modulo, SpatialAttention):
                    print(
                        "  Atención espacial sobre mapas "
                        f"{x.shape[-2]}x{x.shape[-1]}"
                    )
                x = modulo(x)

            forma = " x ".join(map(str, x.shape[1:]))
            print(
                f"{f'Etapa {numero}':<34}"
                f"{forma:<22}"
                f"{contar_parametros(bloque):>12,}"
            )

    print(
        f"{'Cabeza GAP-Dropout-Linear':<34}"
        f"{'1 logit':<22}"
        f"{contar_parametros(modelo.cabeza):>12,}"
    )
    print("-" * 68)
    print(f"Total: {contar_parametros(modelo):,} parámetros")
    print("MaxPooling: 3")
    print(f"Atención: {modelo.atencion}")


# =====================================================================
# VERIFICACIÓN
# =====================================================================

def verificar_formas(modelo: CNNpCR) -> None:
    modelo.eval()
    formas_atencion = []

    def registrar_forma(modulo, entradas):
        formas_atencion.append(tuple(entradas[0].shape))

    manejadores = [
        modulo.register_forward_pre_hook(registrar_forma)
        for modulo in modelo.modules()
        if isinstance(modulo, SpatialAttention)
    ]

    try:
        with torch.no_grad():
            x = torch.randn(2, *ENTRADA)
            representada = representar_entrada(
                x,
                modelo.representacion,
            )
            caracteristicas = modelo.extractor(representada)
            salida = modelo.cabeza(caracteristicas)

        assert caracteristicas.shape == (
            2, modelo.canales[-1], 32, 32
        )
        assert salida.shape == (2, 1)
        assert torch.isfinite(salida).all()

        if modelo.atencion == "spatial":
            assert formas_atencion == [
                (2, modelo.canales[-1], 64, 64)
            ]

        print("A) Salida: (2, 1). OK")
        print("   Mapas finales: 32x32. OK")

        if modelo.atencion == "spatial":
            print("   Atención en mapas de 64x64. OK")

    finally:
        for manejador in manejadores:
            manejador.remove()


def verificar_aprendizaje(
    semilla: int = 42,
    pasos: int = 150,
    representacion: str = "fases",
    canales: tuple[int, ...] = CANALES,
    config_espacial=None,
    atencion: str = "spatial",
) -> None:
    """Memoriza 10 cortes; no mide generalización."""
    fijar_semilla(semilla)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)
    filas = (
        samples[samples.split == "train"]
        .groupby("pCR")
        .head(5)
    )

    if len(filas) != 10:
        raise ValueError(
            "Se necesitan cinco cortes de cada clase"
        )

    x = torch.stack([
        pp.a_tensor(
            pp.cargar_corte_uint8(fila),
            config_espacial=config_espacial,
        )
        for fila in filas.itertuples()
    ]).to(dispositivo)

    y = torch.tensor(
        filas.pCR.values,
        dtype=torch.float32,
    ).unsqueeze(1).to(dispositivo)

    modelo = CNNpCR(
        canales=canales,
        dropout=0.0,
        representacion=representacion,
        atencion=atencion,
    ).to(dispositivo)

    optimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=1e-3,
    )
    criterio = nn.BCEWithLogitsLoss()

    print(f"B) Memorizar 10 cortes ({dispositivo}):")

    for paso in range(1, pasos + 1):
        modelo.train()
        optimizador.zero_grad()
        perdida = criterio(modelo(x), y)

        if not torch.isfinite(perdida):
            raise RuntimeError("Pérdida no finita")

        perdida.backward()
        optimizador.step()

        if paso in (1, 10, 25, 50, 100) or paso == pasos:
            print(
                f"   Paso {paso:>3}: pérdida {perdida.item():.4f}"
            )

    modelo.eval()

    with torch.no_grad():
        prob = torch.sigmoid(modelo(x))

    aciertos = int(
        ((prob >= 0.5).float() == y).sum().item()
    )

    print(f"   Aciertos en modo eval: {aciertos}/10")

    if perdida.item() < 0.1 and aciertos >= 9:
        print("   OK: puede memorizar estos cortes.")
    else:
        print("   No ha memorizado suficientemente.")


# =====================================================================
# UTILIDADES Y MÉTRICAS
# =====================================================================

def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()

    with ruta.open("rb") as fichero:
        for bloque in iter(
            lambda: fichero.read(1 << 20),
            b"",
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
            if np.isfinite(valor) else None
        )

    if isinstance(valor, np.integer):
        return int(valor)

    return valor


def metricas_paciente(
    prob_cortes: np.ndarray,
    filas: pd.DataFrame,
    umbral: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    prob_cortes = np.asarray(prob_cortes).reshape(-1)

    if len(prob_cortes) != len(filas):
        raise ValueError(
            "No coincide el número de probabilidades y cortes"
        )

    datos = pd.DataFrame({
        "patient_id": filas.patient_id.values,
        "prob": prob_cortes,
        "pCR": filas.pCR.values,
    })

    if datos.empty:
        raise ValueError("No hay pacientes para evaluar")

    if not np.isfinite(prob_cortes).all():
        raise ValueError("Probabilidades no finitas")

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

    y = tabla.pCR.values
    p = tabla.prob.values
    pred = (p >= umbral).astype(int)

    vp = int(((pred == 1) & (y == 1)).sum())
    vn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def dividir(a, b):
        return a / b if b else float("nan")

    recall = dividir(vp, vp + fn)

    return {
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
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn,
    }, tabla


def umbral_youden(tabla: pd.DataFrame) -> float:
    """Selecciona el umbral exclusivamente en validación."""
    y = tabla.pCR.values
    p = tabla.prob.values

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Youden necesita ambas clases en validación"
        )

    candidatos = np.r_[
        np.nextafter(p.max(), np.inf),
        np.unique(p),
    ]

    mejor_j = -np.inf
    mejor_umbral = 0.5

    for umbral in candidatos:
        pred = p >= umbral

        sens = (
            (pred & (y == 1)).sum() / (y == 1).sum()
        )
        esp = (
            (~pred & (y == 0)).sum() / (y == 0).sum()
        )

        j = sens + esp - 1

        if j > mejor_j:
            mejor_j = j
            mejor_umbral = float(umbral)

    return mejor_umbral


# =====================================================================
# ENTRENAMIENTO Y PREDICCIÓN
# =====================================================================

def entrenar_epoca(
    modelo,
    dl,
    criterio,
    optimizador,
    dispositivo,
) -> float:
    modelo.train()
    total, n = 0.0, 0

    for x, y in dl:
        x = x.to(dispositivo, non_blocking=True)
        y = y.to(dispositivo).unsqueeze(1)

        optimizador.zero_grad()
        perdida = criterio(modelo(x), y)

        if not torch.isfinite(perdida):
            raise RuntimeError(
                "Pérdida de entrenamiento no finita"
            )

        perdida.backward()
        optimizador.step()

        total += perdida.item() * len(y)
        n += len(y)

    if not n:
        raise ValueError("DataLoader de train vacío")

    return total / n


@torch.no_grad()
def predecir(
    modelo,
    dl,
    criterio,
    dispositivo,
) -> tuple[float, np.ndarray]:
    modelo.eval()
    total, n, probabilidades = 0.0, 0, []

    for x, y in dl:
        x = x.to(dispositivo)
        y = y.to(dispositivo).unsqueeze(1)

        logits = modelo(x)
        perdida = criterio(logits, y)

        if not torch.isfinite(perdida):
            raise RuntimeError(
                "Pérdida de evaluación no finita"
            )

        total += perdida.item() * len(y)
        n += len(y)

        probabilidades.append(
            torch.sigmoid(logits)
            .squeeze(1)
            .cpu()
            .numpy()
        )

    if not n:
        raise ValueError("DataLoader de evaluación vacío")

    return total / n, np.concatenate(probabilidades)


# =====================================================================
# GRÁFICAS
# =====================================================================

def dibujar_curvas(
    hist: pd.DataFrame,
    mejor: int,
    ruta: Path,
    titulo: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(
        hist.epoca,
        hist.perdida_train,
        "o-",
        markersize=4,
        color="#2a78d6",
        label="Train",
    )
    axes[0].plot(
        hist.epoca,
        hist.perdida_val,
        "o-",
        markersize=4,
        color="#eb6834",
        label="Validación",
    )
    axes[0].set_title("Pérdida BCE")
    axes[0].set_ylabel("Pérdida")

    for ax, columna, etiqueta in (
        (axes[1], "recall_val", "Recall"),
        (axes[2], "precision_val", "Precisión"),
    ):
        ax.plot(
            hist.epoca,
            hist[columna],
            "o-",
            markersize=4,
            color="#eb6834",
            label="Validación",
        )
        ax.set_title(
            f"{etiqueta} por paciente · pCR = 1"
        )
        ax.set_ylabel(etiqueta)
        ax.set_ylim(-0.03, 1.03)

    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        ax.axvline(
            mejor,
            color="gray",
            linestyle=":",
            linewidth=1,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"{titulo}\n"
        f"Recall y precisión: umbral 0,5 · "
        f"mejor época por AUC: {mejor}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def dibujar_matriz(
    met: dict,
    ruta: Path,
    titulo: str,
) -> None:
    matriz = np.array([
        [met["VN"], met["FP"]],
        [met["FN"], met["VP"]],
    ])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matriz, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                color=(
                    "white"
                    if matriz[i, j] > matriz.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks(
        [0, 1],
        ["Pred. no pCR", "Pred. pCR"],
    )
    ax.set_yticks(
        [0, 1],
        ["Real no pCR", "Real pCR"],
    )
    ax.set_title(titulo, fontsize=10)

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# =====================================================================
# TEST FINAL OPCIONAL
# =====================================================================

def evaluar_test(
    modelo,
    ruta_pesos,
    umbral,
    args,
    dispositivo,
    carpeta,
    config_espacial,
) -> dict:
    modelo.load_state_dict(
        torch.load(
            ruta_pesos,
            map_location=dispositivo,
            weights_only=True,
        )
    )

    dl_test, filas_test = pp.crear_dataloader_test(
        batch_size=args.batch * 2,
        num_workers=args.workers,
        config_espacial=config_espacial,
    )

    _, prob = predecir(
        modelo,
        dl_test,
        nn.BCEWithLogitsLoss(),
        dispositivo,
    )

    met_05, tabla = metricas_paciente(
        prob,
        filas_test,
        umbral=0.5,
    )
    met_umbral, _ = metricas_paciente(
        prob,
        filas_test,
        umbral=umbral,
    )

    tabla.to_csv(carpeta / "test_pacientes.csv")

    dibujar_matriz(
        met_umbral,
        carpeta / "matriz_confusion_test.png",
        f"Test · umbral de validación {umbral:.3f}",
    )

    return {
        "pacientes": len(tabla),
        "umbral_0.5": met_05,
        "umbral_validacion": umbral,
        "metricas_umbral_validacion": met_umbral,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--paciencia", type=int, default=5)
    parser.add_argument(
        "--perdida",
        choices=["ponderada", "normal"],
        default="ponderada",
    )
    parser.add_argument(
        "--representacion",
        choices=["fases", "realce"],
        default="fases",
    )
    parser.add_argument(
        "--atencion",
        choices=["ninguna", "spatial"],
        default="spatial",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROPOUT,
    )
    parser.add_argument(
        "--canales",
        type=int,
        nargs=4,
        default=list(CANALES),
    )
    parser.add_argument("--sin-aumento", action="store_true")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--nombre", default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--verificar", action="store_true")
    parser.add_argument(
        "--preprocesado-config",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.epocas < 1 or args.paciencia < 1:
        parser.error("epocas y paciencia deben ser positivas")

    if args.batch < 1 or args.workers < 0:
        parser.error("batch debe ser positivo y workers >= 0")

    if not 0 <= args.fold <= 4:
        parser.error("fold debe estar entre 0 y 4")

    if not 0 <= args.dropout < 1:
        parser.error("dropout debe estar en [0, 1)")

    if any(c < 1 for c in args.canales):
        parser.error("Todos los canales deben ser positivos")

    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr debe ser positivo y finito")

    if (
        not np.isfinite(args.weight_decay)
        or args.weight_decay < 0
    ):
        parser.error("weight-decay debe ser finito y >= 0")

    if args.rapido and args.test:
        parser.error(
            "Reserva --test para la evaluación final, "
            "sin modo rápido"
        )

    config_espacial = (
        pp.ConfigEspacial.desde_json(args.preprocesado_config)
        if args.preprocesado_config
        else pp.ConfigEspacial()
    )

    if args.verificar:
        modelo = CNNpCR(
            canales=tuple(args.canales),
            dropout=args.dropout,
            representacion=args.representacion,
            atencion=args.atencion,
        )
        resumen(modelo)
        verificar_formas(modelo)
        verificar_aprendizaje(
            semilla=args.semilla,
            representacion=args.representacion,
            canales=tuple(args.canales),
            config_espacial=config_espacial,
            atencion=args.atencion,
        )
        return

    fijar_semilla(args.semilla)

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    nombre = args.nombre

    if nombre is None:
        nombre = (
            f"cnn4_3pool_{args.perdida}_f{args.fold}"
            f"_{args.representacion}_{args.atencion}"
        )
        if args.preprocesado_config:
            nombre += f"_{args.preprocesado_config.stem}"
        if args.rapido:
            nombre += "_rapido"

    carpeta = (
        REPO / "resultados" / "entrenamiento" / nombre
    )
    ruta_pesos = REPO / "models" / f"{nombre}.pt"

    if (
        ruta_pesos.exists()
        or (carpeta.exists() and any(carpeta.iterdir()))
    ):
        parser.error(
            f"El experimento '{nombre}' ya tiene resultados. "
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

    if datos["filas_val"].pCR.nunique() != 2:
        raise ValueError(
            "Validación necesita ambas clases para calcular AUC"
        )

    pos_weight = float(datos["pos_weight"])

    modelo = CNNpCR(
        canales=tuple(args.canales),
        dropout=args.dropout,
        representacion=args.representacion,
        atencion=args.atencion,
    ).to(dispositivo)

    peso_positivo = (
        torch.tensor([pos_weight], device=dispositivo)
        if args.perdida == "ponderada"
        else None
    )

    criterio = nn.BCEWithLogitsLoss(
        pos_weight=peso_positivo
    )

    # Mantiene la validación sin ponderar.
    criterio_val = nn.BCEWithLogitsLoss()

    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos.parent.mkdir(parents=True, exist_ok=True)

    print(f"Experimento: {nombre}")
    print("Arquitectura: 4 etapas, 3 MaxPooling y GAP final")
    print(f"Canales: {tuple(args.canales)}")
    print(f"Representación: {args.representacion}")
    print(f"Atención: {args.atencion}")

    if args.atencion == "spatial":
        print(
            "Atención en etapa 4, antes del último pooling, "
            "sobre mapas de 64x64"
        )

    print(f"Dispositivo: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(
        f"Train: {len(datos['filas_train'])} cortes / "
        f"{datos['filas_train'].patient_id.nunique()} pacientes | "
        f"Val: {len(datos['filas_val'])} cortes / "
        f"{datos['filas_val'].patient_id.nunique()} pacientes"
    )
    print(
        f"Pérdida: {args.perdida} | pos_weight: {pos_weight:.3f} | "
        f"parámetros: {contar_parametros(modelo):,} | "
        f"caché: {datos['usa_cache']}"
    )
    print("Preprocesado:", config_espacial.to_dict())
    print(
        f"Épocas máx.: {args.epocas} | "
        f"paciencia: {args.paciencia} | "
        f"batch: {args.batch} | lr: {args.lr}"
    )
    print(
        "Recall y precisión: por paciente, pCR=1, "
        "umbral fijo 0.5.\n"
    )
    print(
        f"{'época':>5} {'pérd_train':>10} "
        f"{'pérd_val':>9} {'AUC_val':>8} "
        f"{'recall':>7} {'prec.':>7} "
        f"{'esp.':>7} {'tiempo':>7}"
    )

    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    historial = []
    mejor_auc = -np.inf
    mejor_epoca = 0
    mejores_metricas = None
    sin_mejora = 0
    inicio_total = time.time()

    for epoca in range(1, args.epocas + 1):
        inicio_epoca = time.time()

        perdida_train = entrenar_epoca(
            modelo,
            datos["train"],
            criterio,
            optimizador,
            dispositivo,
        )

        perdida_val, prob = predecir(
            modelo,
            datos["val"],
            criterio_val,
            dispositivo,
        )

        met, tabla = metricas_paciente(
            prob,
            datos["filas_val"],
            umbral=0.5,
        )

        if not np.isfinite(met["auc"]):
            raise RuntimeError("AUC de validación no finita")

        segundos = time.time() - inicio_epoca

        historial.append({
            "epoca": epoca,
            "perdida_train": perdida_train,
            "perdida_val": perdida_val,
            "auc_val": met["auc"],
            "sens_val": met["sensibilidad"],
            "recall_val": met["recall"],
            "precision_val": met["precision"],
            "esp_val": met["especificidad"],
            "segundos": segundos,
        })

        marca = ""

        if met["auc"] > mejor_auc:
            mejor_auc = met["auc"]
            mejor_epoca = epoca
            mejores_metricas = met.copy()
            sin_mejora = 0

            torch.save(modelo.state_dict(), ruta_pesos)
            tabla.to_csv(carpeta / "val_pacientes.csv")
            marca = "  * guardado"

        else:
            sin_mejora += 1

        hist = pd.DataFrame(historial)
        hist.to_csv(carpeta / "historial.csv", index=False)

        dibujar_curvas(
            hist,
            mejor_epoca,
            carpeta / "curvas.png",
            (
                f"{nombre} · pérdida {args.perdida} "
                f"· atención {args.atencion}"
            ),
        )

        print(
            f"{epoca:>5} "
            f"{perdida_train:>10.4f} "
            f"{perdida_val:>9.4f} "
            f"{met['auc']:>8.3f} "
            f"{met['recall']:>7.2f} "
            f"{met['precision']:>7.2f} "
            f"{met['especificidad']:>7.2f} "
            f"{segundos:>6.0f}s{marca}",
            flush=True,
        )

        if sin_mejora >= args.paciencia:
            print(
                f"\nEarly stopping: {args.paciencia} épocas "
                "sin mejorar el AUC de validación."
            )
            break

    segundos_entrenamiento = time.time() - inicio_total
    hist = pd.DataFrame(historial)

    tabla_mejor = pd.read_csv(
        carpeta / "val_pacientes.csv",
        index_col=0,
    )

    umbral = umbral_youden(tabla_mejor)

    filas_mejor = pd.DataFrame({
        "patient_id": tabla_mejor.index.to_numpy(),
        "pCR": tabla_mejor.pCR.to_numpy(),
    })

    met_umbral, _ = metricas_paciente(
        tabla_mejor.prob.to_numpy(),
        filas_mejor,
        umbral=umbral,
    )

    dibujar_matriz(
        met_umbral,
        carpeta / "matriz_confusion_val.png",
        f"Validación · umbral Youden {umbral:.3f}",
    )

    config = {
        "experimento": nombre,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": modelo.config(),
        "entrenamiento": {
            "epocas_max": args.epocas,
            "epocas_realizadas": len(hist),
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
            "batch": args.batch,
            "semilla": args.semilla,
            "fold_validacion": args.fold,
            "aumento": datos["aumento"],
            "preprocesado_espacial": config_espacial.to_dict(),
            "preprocesado_config": (
                str(args.preprocesado_config)
                if args.preprocesado_config else None
            ),
            "modo_rapido": args.rapido,
            "agregacion_paciente": "media",
            "umbral_curvas_recall_precision": 0.5,
            "clase_positiva": "pCR=1",
            "perdida_val_ponderada": False,
        },
        "validacion_mejor_epoca": {
            "umbral_0.5": mejores_metricas,
            "umbral_youden": umbral,
            "con_umbral_youden": met_umbral,
        },
        "entorno": {
            "dispositivo": str(dispositivo),
            "gpu": (
                torch.cuda.get_device_name(0)
                if dispositivo.type == "cuda" else None
            ),
            "memoria_max_MB": (
                torch.cuda.max_memory_allocated() / 2**20
                if dispositivo.type == "cuda" else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sistema": platform.platform(),
            "segundos_por_epoca_medio": float(hist.segundos.mean()),
            "segundos_entrenamiento": segundos_entrenamiento,
        },
        "pesos": {
            "ruta": str(ruta_pesos.relative_to(REPO)),
            "sha256": sha256(ruta_pesos),
        },
        "test": None,
    }

    def guardar_config():
        (carpeta / "config.json").write_text(
            json.dumps(
                limpiar_json(config),
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    guardar_config()

    if args.test:
        print(
            "\nEvaluando TEST con pesos y umbral "
            "elegidos en validación..."
        )

        config["test"] = evaluar_test(
            modelo,
            ruta_pesos,
            umbral,
            args,
            dispositivo,
            carpeta,
            config_espacial,
        )
        guardar_config()

    m = mejores_metricas

    print(
        f"\nMejor época: {mejor_epoca} | "
        f"AUC val por paciente: {mejor_auc:.3f}"
    )
    print(
        f"Umbral 0.5 -> recall {m['recall']:.2f} | "
        f"precisión {m['precision']:.2f} | "
        f"especificidad {m['especificidad']:.2f}"
    )
    print(
        f"VP {m['VP']} | VN {m['VN']} | "
        f"FP {m['FP']} | FN {m['FN']}"
    )
    print(
        f"Umbral Youden {umbral:.3f} -> "
        f"recall {met_umbral['recall']:.2f} | "
        f"precisión {met_umbral['precision']:.2f} | "
        f"especificidad {met_umbral['especificidad']:.2f}"
    )

    if config["test"] is not None:
        mt = config["test"]["metricas_umbral_validacion"]

        print(
            f"TEST -> AUC {mt['auc']:.3f} | "
            f"recall {mt['recall']:.2f} | "
            f"precisión {mt['precision']:.2f} | "
            f"especificidad {mt['especificidad']:.2f}"
        )

    print(
        f"Tiempo de entrenamiento: "
        f"{segundos_entrenamiento / 60:.1f} min"
    )
    print(f"Pesos: {ruta_pesos.relative_to(REPO)}")
    print(f"Resultados: {carpeta.relative_to(REPO)}")
    print("Gráficas: curvas.png")

    if args.rapido:
        print(
            "\nModo rápido: las métricas solo comprueban "
            "el funcionamiento."
        )


if __name__ == "__main__":
    main()