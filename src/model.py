#!/usr/bin/env python3
"""CNN desde cero para predecir pCR con DCE-MRI (caso BreastDCEDL): modelo, entrenamiento y test.

Este archivo contiene TODO el modelo:
  1. ARQUITECTURA    la red CNNpCR (6 etapas Conv-BN-ReLU-MaxPool + GAP + Dropout + Linear)
  2. VERIFICACIÓN    paso 6 del método de clase: A) formas, B) sobreajustar 10 cortes
  3. ENTRENAMIENTO   épocas con validación por paciente, early stopping, mejor época
  4. TEST            evaluación final UNA sola vez (con --test), con la mejor época y el
                     umbral elegidos en validación; test nunca decide nada

El preprocesado (carga de PNG, /255, aumento, DataLoaders) está en preprocessing.py,
porque también lo usa la aplicación web y tiene que ser idéntico.

Diseño (método de 6 pasos de clase):
  entrada 3x256x256 (PRE, EARLY, LATE) -> 6 etapas (256 -> 4) con canales
  16-32-64-128-128-128 -> Global Average Pooling -> Dropout 0,3 -> Linear(128, 1) = 1 logit.
  ~393 K parámetros. Sin pesos preentrenados.

Al entrenar guarda models/<nombre>.pt y en resultados/entrenamiento/<nombre>/:
  config.json (hiperparámetros, entorno, tiempos, métricas, SHA-256 de los pesos),
  historial.csv, curvas.png, val_pacientes.csv y, con --test,
  matriz_confusion_test.png y test_pacientes.csv.

Uso, desde la raíz del repositorio:

    python src/model.py --verificar           # resumen de la red + verificaciones A y B
    python src/model.py --rapido              # prueba en CPU: pocos pacientes, 10 épocas
    python src/model.py                       # entrenamiento real (10 épocas)
    python src/model.py --perdida normal      # comparar con la pérdida sin ponderar
    python src/model.py --test                # entrena y, al final, evalúa en test
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
import preprocessing as pp                       # noqa: E402

REPO = pp.REPO

# Valores por defecto de la arquitectura
CANALES = (16, 32, 64, 128, 128, 128)     # filtros de cada una de las 6 etapas
DROPOUT = 0.3
ENTRADA = (3, 256, 256)                   # PRE, EARLY, LATE x 256 x 256


# =========================================================================== #
# 1. ARQUITECTURA
# =========================================================================== #

def etapa(c_in: int, c_out: int) -> nn.Sequential:
    """Una etapa: entra c_in x H x W, sale c_out x H/2 x W/2.

    - Conv 3x3 con padding 1: conserva el tamaño; sin sesgo porque BatchNorm
      lo sustituye con su parámetro beta.
    - BatchNorm: estabiliza el entrenamiento (media 0, varianza 1 por canal).
    - ReLU: no linealidad; sin ella, apilar capas equivaldría a una sola.
    - MaxPool 2x2: divide alto y ancho entre 2 y tolera pequeños desplazamientos.
    El orden Conv -> BN -> ReLU -> Pool es el canónico visto en clase.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class CNNpCR(nn.Module):
    """CNN 2D desde cero. Devuelve un logit por imagen: [N, 3, 256, 256] -> [N, 1].

    La sigmoide NO está dentro de la red: BCEWithLogitsLoss la aplica al entrenar
    (más estable numéricamente) y en inferencia se hace torch.sigmoid(logit).
    """

    def __init__(self, canales: tuple[int, ...] = CANALES, dropout: float = DROPOUT):
        super().__init__()
        self.canales = tuple(canales)
        self.dropout = dropout

        # Extractor de características: 3x256x256 -> 128x4x4
        capas, c_in = [], ENTRADA[0]          # 3 canales = 3 fases DCE (no son colores)
        for c_out in canales:
            capas.append(etapa(c_in, c_out))
            c_in = c_out
        self.extractor = nn.Sequential(*capas)

        # Cabeza: 128x4x4 -> 128 -> 1 logit
        self.cabeza = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # Global Average Pooling: media de cada mapa
            nn.Flatten(),
            nn.Dropout(dropout),              # solo actúa en model.train()
            nn.Linear(c_in, 1),
        )
        self._inicializar()

    def _inicializar(self):
        """Kaiming (He) para las convoluciones: pensada para ReLU, mantiene estable
        la varianza de las activaciones a lo largo de las 6 etapas."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)       # gamma = 1
                nn.init.zeros_(m.bias)        # beta = 0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cabeza(self.extractor(x))

    def config(self) -> dict:
        """Hiperparámetros de arquitectura, para guardarlos junto a los pesos."""
        return {"canales": list(self.canales), "dropout": self.dropout,
                "entrada": list(ENTRADA), "parametros": contar_parametros(self)}


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Resumen capa a capa (tabla para la diapositiva 3)
# --------------------------------------------------------------------------- #

def resumen(modelo: CNNpCR) -> None:
    x = torch.zeros(1, *ENTRADA)
    modelo.eval()
    print(f"{'Capa':<28}{'Salida':<20}{'Parámetros':>12}   Cuenta")
    print("-" * 86)
    print(f"{'Entrada':<28}{' x '.join(map(str, ENTRADA)):<20}{'—':>12}")
    c_in = ENTRADA[0]
    with torch.no_grad():
        for i, bloque in enumerate(modelo.extractor, 1):
            x = bloque(x)
            c = x.shape[1]
            n = contar_parametros(bloque)
            cuenta = f"3·3·{c_in}·{c} + 2·{c}"
            print(f"{f'Etapa {i} (Conv-BN-ReLU-Pool)':<28}{' x '.join(map(str, x.shape[1:])):<20}{n:>12,}   {cuenta}")
            c_in = c
        gap, _, drop, lineal = modelo.cabeza
        print(f"{'Global Average Pooling':<28}{str(c_in):<20}{0:>12}")
        print(f"{f'Dropout (p={drop.p})':<28}{str(c_in):<20}{0:>12}")
        n = contar_parametros(lineal)
        print(f"{'Linear':<28}{'1 (logit)':<20}{n:>12,}   {c_in}·1 + 1")
    print("-" * 86)
    print(f"{'TOTAL':<48}{contar_parametros(modelo):>12,}")
    print(f"Campo receptivo final: {campo_receptivo(len(modelo.extractor))} x "
          f"{campo_receptivo(len(modelo.extractor))} píxeles de la imagen original")


def campo_receptivo(n_etapas: int) -> int:
    """Cada conv 3x3 suma 2·salto; cada MaxPool 2x2 suma 1·salto y duplica el salto."""
    rf, salto = 1, 1
    for _ in range(n_etapas):
        rf += 2 * salto          # conv 3x3
        rf += 1 * salto          # maxpool 2x2
        salto *= 2
    return rf


# =========================================================================== #
# 2. VERIFICACIÓN (paso 6 del método de clase)
# =========================================================================== #

def verificar_formas(modelo: CNNpCR) -> None:
    """A · ¿Las formas cuadran? Si falla aquí, es por dimensiones, no por aprendizaje."""
    modelo.eval()
    with torch.no_grad():
        y = modelo(torch.randn(2, *ENTRADA))
    assert y.shape == (2, 1), f"salida {tuple(y.shape)}, se esperaba (2, 1)"
    print(f"A) Formas: entrada (2, 3, 256, 256) -> salida {tuple(y.shape)}   OK")


def verificar_aprendizaje(semilla: int = 42, pasos: int = 150) -> None:
    """B · ¿Es capaz de aprender? Debe MEMORIZAR 10 cortes reales (5 de cada clase).

    No mide si el modelo es bueno: solo que modelo + pérdida + optimizador funcionan.
    Si la pérdida se queda en ~0,69 (= ln 2, el azar en binario), hay un error.
    """
    torch.manual_seed(semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)
    tr = samples[samples.split == "train"]
    filas = tr.groupby("pCR").head(5)                      # 5 cortes de cada clase
    x = torch.stack([pp.a_tensor(pp.cargar_corte_uint8(f)) for f in filas.itertuples()])
    y = torch.tensor(filas.pCR.values, dtype=torch.float32).unsqueeze(1)
    x, y = x.to(dispositivo), y.to(dispositivo)

    modelo = CNNpCR(dropout=0.0).to(dispositivo)           # sin dropout: queremos memorizar
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = nn.BCEWithLogitsLoss()

    print(f"B) Sobreajustar 10 cortes reales ({dispositivo}):")
    t0 = time.time()
    for paso in range(1, pasos + 1):
        modelo.train()
        optimizador.zero_grad()                            # PyTorch acumula gradientes
        perdida = criterio(modelo(x), y)                   # forward + pérdida
        perdida.backward()                                 # retropropagación
        optimizador.step()                                 # actualizar pesos
        if paso in (1, 10, 25, 50, 100) or paso == pasos:
            print(f"   paso {paso:>3}: pérdida = {perdida.item():.4f}")

    modelo.eval()
    with torch.no_grad():
        prob = torch.sigmoid(modelo(x))
    aciertos = int(((prob >= 0.5).float() == y).sum())
    print(f"   aciertos en esos 10 cortes (modo eval): {aciertos}/10   [{time.time() - t0:.0f} s]")
    if perdida.item() < 0.1 and aciertos >= 9:
        print("   OK: la red es capaz de aprender (memoriza los 10 cortes).")
    else:
        print("   AVISO: no ha memorizado. Revisa modelo, pérdida u optimizador.")



# =========================================================================== #
# 3. ENTRENAMIENTO
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Utilidades: semilla, checksum y métricas por paciente
# --------------------------------------------------------------------------- #

def fijar_semilla(semilla: int) -> None:
    """Misma semilla en random, numpy y torch: resultados reproducibles."""
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def metricas_paciente(prob_cortes: np.ndarray, filas: pd.DataFrame, umbral: float = 0.5) -> tuple[dict, pd.DataFrame]:
    """Agrega por paciente (media de sus cortes) y calcula las métricas.

    pCR es una propiedad de la paciente, no del corte: la unidad de evaluación es la paciente.
    """
    tabla = (pd.DataFrame({"patient_id": filas.patient_id.values, "prob": prob_cortes,
                           "pCR": filas.pCR.values})
             .groupby("patient_id").agg(prob=("prob", "mean"), pCR=("pCR", "first")))
    y, p = tabla.pCR.values, tabla.prob.values
    pred = (p >= umbral).astype(int)
    vp, vn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return {"auc": auc, "sensibilidad": div(vp, vp + fn), "especificidad": div(vn, vn + fp),
            "accuracy": div(vp + vn, len(y)), "VP": vp, "VN": vn, "FP": fp, "FN": fn}, tabla


def umbral_youden(tabla: pd.DataFrame) -> float:
    """Umbral que maximiza sensibilidad + especificidad - 1 en VALIDACIÓN (nunca en test)."""
    y, p = tabla.pCR.values, tabla.prob.values
    mejor, u_mejor = -1.0, 0.5
    for u in np.unique(p):
        pred = p >= u
        sens = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        esp = (~pred & (y == 0)).sum() / max(1, (y == 0).sum())
        if sens + esp - 1 > mejor:
            mejor, u_mejor = sens + esp - 1, float(u)
    return u_mejor


# --------------------------------------------------------------------------- #
# Una época de entrenamiento y la predicción
# --------------------------------------------------------------------------- #

def entrenar_epoca(modelo, dl, criterio, optimizador, dispositivo) -> float:
    modelo.train()                                   # activa dropout y BatchNorm en modo lote
    total, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo).unsqueeze(1)
        optimizador.zero_grad()                      # PyTorch acumula gradientes: hay que borrarlos
        perdida = criterio(modelo(x), y)             # forward + pérdida
        perdida.backward()                           # retropropagación
        optimizador.step()                           # actualizar pesos
        total += perdida.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()                                     # sin gradientes: más rápido y menos memoria
def predecir(modelo, dl, criterio, dispositivo) -> tuple[float, np.ndarray]:
    modelo.eval()                                    # desactiva dropout; BatchNorm usa medias móviles
    total, n, probs = 0.0, 0, []
    for x, y in dl:
        x, y = x.to(dispositivo), y.to(dispositivo).unsqueeze(1)
        logits = modelo(x)
        total += criterio(logits, y).item() * len(y)
        n += len(y)
        probs.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    return total / n, np.concatenate(probs)


# =========================================================================== #
# 4. GRÁFICAS Y TEST
# =========================================================================== #

def dibujar_curvas(hist: pd.DataFrame, mejor: int, ruta: Path, titulo: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.epoca, hist.perdida_train, marker="o", color="#2a78d6", label="train")
    axes[0].plot(hist.epoca, hist.perdida_val, marker="o", color="#eb6834", label="validación")
    axes[0].set_title("Pérdida (BCE)"); axes[0].set_xlabel("época"); axes[0].legend(frameon=False)
    axes[1].plot(hist.epoca, hist.auc_val, marker="o", color="#eb6834", label="AUC validación (paciente)")
    axes[1].axhline(0.5, color="gray", ls="--", lw=1)
    axes[1].text(hist.epoca.min(), 0.505, " azar", color="gray", fontsize=8)
    axes[1].set_ylim(0.3, 1.0); axes[1].set_title("AUC por paciente"); axes[1].set_xlabel("época")
    for ax in axes:
        ax.axvline(mejor, color="gray", ls=":", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].text(mejor, 0.32, f" mejor época = {mejor}", fontsize=8, color="gray")
    axes[1].legend(frameon=False, loc="upper left")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def dibujar_matriz(met: dict, ruta: Path, titulo: str) -> None:
    m = np.array([[met["VN"], met["FP"]], [met["FN"], met["VP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=14,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred. no pCR", "pred. pCR"])
    ax.set_yticks([0, 1], ["real no pCR", "real pCR"])
    ax.set_title(titulo, fontsize=10)
    fig.tight_layout(); fig.savefig(ruta, dpi=120); plt.close(fig)


def evaluar_test(modelo, ruta_pesos: Path, umbral: float, args, dispositivo, carpeta: Path) -> dict:
    """Evaluación FINAL en test: una sola pasada, sin tocar nada del modelo.

    Usa los pesos de la mejor época (elegida en validación) y el umbral elegido en
    validación. Si después de mirar estos resultados se cambia el modelo, dejan de
    ser válidos.
    """
    modelo.load_state_dict(torch.load(ruta_pesos, map_location=dispositivo))
    dl_te, filas_te = pp.crear_dataloader_test(batch_size=args.batch * 2, num_workers=args.workers)
    if args.rapido:                                   # modo prueba: pocos pacientes también en test
        pac = filas_te.drop_duplicates("patient_id").groupby("pCR").head(5).patient_id
        filas_te = filas_te[filas_te.patient_id.isin(pac)].reset_index(drop=True)
        dl_te = torch.utils.data.DataLoader(pp.DatasetCortes(filas_te), batch_size=args.batch * 2,
                                            shuffle=False)
    _, prob = predecir(modelo, dl_te, nn.BCEWithLogitsLoss(), dispositivo)
    met_05, tabla = metricas_paciente(prob, filas_te, 0.5)
    met_u, _ = metricas_paciente(prob, filas_te, umbral)
    tabla.to_csv(carpeta / "test_pacientes.csv")
    dibujar_matriz(met_u, carpeta / "matriz_confusion_test.png",
                   f"Test ({len(tabla)} pacientes) · umbral {umbral:.3f}")
    redondear = lambda d: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
    return {"pacientes": len(tabla), "umbral_0.5": redondear(met_05),
            f"umbral_validacion_{umbral:.4f}": redondear(met_u)}


# =========================================================================== #
# MAIN: verificar, entrenar y (opcional) evaluar en test
# =========================================================================== #

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epocas", type=int, default=10, help="número máximo de épocas (defecto: 10)")
    p.add_argument("--paciencia", type=int, default=5, help="early stopping: épocas sin mejorar el AUC")
    p.add_argument("--perdida", choices=["ponderada", "normal"], default="ponderada",
                   help="ponderada = BCEWithLogitsLoss con pos_weight = N0/N1")
    p.add_argument("--fold", type=int, default=0, help="fold de validación (0-4)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--canales", type=int, nargs=6, default=[16, 32, 64, 128, 128, 128])
    p.add_argument("--sin-aumento", action="store_true", help="desactiva el aumento de datos")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--workers", type=int, default=0, help="procesos para cargar datos (0 en Windows/CPU)")
    p.add_argument("--rapido", action="store_true",
                   help="modo prueba: 10 pacientes por clase en train y 5 en validación")
    p.add_argument("--nombre", default=None, help="nombre del experimento (carpeta y pesos)")
    p.add_argument("--test", action="store_true",
                   help="al final, evalúa UNA vez en test con la mejor época y el umbral de validación")
    p.add_argument("--verificar", action="store_true",
                   help="solo muestra el resumen de la red y hace las verificaciones A y B (no entrena)")
    args = p.parse_args()

    if args.verificar:
        modelo = CNNpCR(canales=tuple(args.canales), dropout=args.dropout)
        resumen(modelo)
        print()
        verificar_formas(modelo)
        verificar_aprendizaje(semilla=args.semilla)
        return

    fijar_semilla(args.semilla)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = args.nombre or f"{args.perdida}_f{args.fold}" + ("_rapido" if args.rapido else "")
    carpeta = REPO / "resultados" / "entrenamiento" / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_pesos = REPO / "models" / f"{nombre}.pt"
    ruta_pesos.parent.mkdir(exist_ok=True)

    # ---- Datos ----
    d = pp.crear_dataloaders(fold_val=args.fold, batch_size=args.batch, aumentar=not args.sin_aumento,
                             num_workers=args.workers, semilla=args.semilla,
                             max_pacientes=10 if args.rapido else None)
    pos_weight = d["pos_weight"]

    # ---- Modelo, pérdida y optimizador ----
    modelo = CNNpCR(canales=tuple(args.canales), dropout=args.dropout).to(dispositivo)
    pw = torch.tensor([pos_weight], device=dispositivo) if args.perdida == "ponderada" else None
    criterio = nn.BCEWithLogitsLoss(pos_weight=pw)
    criterio_val = nn.BCEWithLogitsLoss()           # pérdida de validación sin ponderar: comparable
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Experimento: {nombre}")
    print(f"Dispositivo: {dispositivo}" + (f" ({torch.cuda.get_device_name(0)})" if dispositivo.type == "cuda" else ""))
    print(f"Train: {len(d['filas_train'])} cortes / {d['filas_train'].patient_id.nunique()} pacientes | "
          f"Val: {len(d['filas_val'])} cortes / {d['filas_val'].patient_id.nunique()} pacientes")
    print(f"Pérdida: {args.perdida} (pos_weight = {pos_weight:.3f}) | parámetros: {contar_parametros(modelo):,} | "
          f"caché: {d['usa_cache']}")
    print(f"Épocas máx.: {args.epocas} | paciencia: {args.paciencia} | batch: {args.batch} | lr: {args.lr}\n")
    print(f"{'época':>5} {'pérd_train':>10} {'pérd_val':>9} {'AUC_val':>8} {'sens':>6} {'esp':>6} {'tiempo':>7}")

    # ---- Bucle de épocas ----
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    historial, mejor_auc, mejor_epoca, sin_mejora = [], -1.0, 0, 0
    t_total = time.time()
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        perd_tr = entrenar_epoca(modelo, d["train"], criterio, optimizador, dispositivo)
        perd_va, prob = predecir(modelo, d["val"], criterio_val, dispositivo)
        met, tabla = metricas_paciente(prob, d["filas_val"])
        t_epoca = time.time() - t0
        historial.append({"epoca": epoca, "perdida_train": perd_tr, "perdida_val": perd_va,
                          "auc_val": met["auc"], "sens_val": met["sensibilidad"],
                          "esp_val": met["especificidad"], "segundos": t_epoca})

        marca = ""
        if met["auc"] > mejor_auc:                   # guardar la MEJOR época, no la última
            mejor_auc, mejor_epoca, sin_mejora = met["auc"], epoca, 0
            torch.save(modelo.state_dict(), ruta_pesos)
            tabla.to_csv(carpeta / "val_pacientes.csv")
            mejores_metricas = met
            marca = "  * guardado"
        else:
            sin_mejora += 1
        print(f"{epoca:>5} {perd_tr:>10.4f} {perd_va:>9.4f} {met['auc']:>8.3f} "
              f"{met['sensibilidad']:>6.2f} {met['especificidad']:>6.2f} {t_epoca:>6.0f}s{marca}")

        if sin_mejora >= args.paciencia:
            print(f"\nEarly stopping: {args.paciencia} épocas sin mejorar el AUC de validación.")
            break

    # ---- Resultados ----
    t_total = time.time() - t_total
    hist = pd.DataFrame(historial)
    hist.to_csv(carpeta / "historial.csv", index=False)
    dibujar_curvas(hist, mejor_epoca, carpeta / "curvas.png", f"{nombre} · pérdida {args.perdida}")

    tabla_mejor = pd.read_csv(carpeta / "val_pacientes.csv", index_col=0)
    umbral = umbral_youden(tabla_mejor)
    met_umbral, _ = metricas_paciente(
        tabla_mejor.prob.values, pd.DataFrame({"patient_id": tabla_mejor.index, "pCR": tabla_mejor.pCR}), umbral)

    resultados_test = None
    if args.test:
        print("\nEvaluando en TEST (una sola vez, con la mejor época y el umbral de validación)...")
        resultados_test = evaluar_test(modelo, ruta_pesos, umbral, args, dispositivo, carpeta)

    config = {
        "experimento": nombre,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "arquitectura": modelo.config(),
        "entrenamiento": {
            "epocas_max": args.epocas, "epocas_realizadas": len(hist), "mejor_epoca": mejor_epoca,
            "paciencia": args.paciencia, "criterio_parada": "AUC por paciente en validación",
            "perdida": args.perdida, "pos_weight": round(pos_weight, 4) if args.perdida == "ponderada" else None,
            "optimizador": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "batch": args.batch, "semilla": args.semilla, "fold_validacion": args.fold,
            "aumento": d["aumento"], "normalizacion": "PNG / 255, apilado PRE, EARLY, LATE",
            "modo_rapido": args.rapido, "agregacion_paciente": "media",
        },
        "validacion_mejor_epoca": {
            "umbral_0.5": {k: round(v, 4) if isinstance(v, float) else v for k, v in mejores_metricas.items()},
            "umbral_youden": round(umbral, 4),
            "con_umbral_youden": {k: round(v, 4) if isinstance(v, float) else v for k, v in met_umbral.items()},
        },
        "entorno": {
            "dispositivo": str(dispositivo),
            "gpu": torch.cuda.get_device_name(0) if dispositivo.type == "cuda" else None,
            "memoria_max_MB": round(torch.cuda.max_memory_allocated() / 2**20) if dispositivo.type == "cuda" else None,
            "python": platform.python_version(), "torch": torch.__version__, "sistema": platform.platform(),
            "segundos_por_epoca_medio": round(float(hist.segundos.mean()), 1),
            "segundos_total": round(t_total, 1),
        },
        "pesos": {"ruta": str(ruta_pesos.relative_to(REPO)), "sha256": sha256(ruta_pesos)},
        "test": resultados_test,
    }
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    m = mejores_metricas
    print(f"\nMejor época: {mejor_epoca} | AUC val (paciente) = {mejor_auc:.3f}")
    print(f"Umbral 0,5    -> sens {m['sensibilidad']:.2f} | esp {m['especificidad']:.2f} | "
          f"VP {m['VP']} VN {m['VN']} FP {m['FP']} FN {m['FN']}")
    print(f"Umbral Youden {umbral:.3f} -> sens {met_umbral['sensibilidad']:.2f} | "
          f"esp {met_umbral['especificidad']:.2f}")
    if resultados_test:
        mt = resultados_test[f"umbral_validacion_{umbral:.4f}"]
        print(f"\nTEST ({resultados_test['pacientes']} pacientes, umbral {umbral:.3f}): "
              f"AUC {mt['auc']:.3f} | sens {mt['sensibilidad']:.2f} | esp {mt['especificidad']:.2f} | "
              f"VP {mt['VP']} VN {mt['VN']} FP {mt['FP']} FN {mt['FN']}")
    print(f"Tiempo total: {t_total / 60:.1f} min ({hist.segundos.mean():.0f} s/época)")
    print(f"Pesos:      {ruta_pesos.relative_to(REPO)}")
    print(f"Resultados: {carpeta.relative_to(REPO)}")
    if args.rapido:
        print("\n(Modo rápido: las métricas NO son representativas; solo comprueba que todo funciona.)")


if __name__ == "__main__":
    main()
