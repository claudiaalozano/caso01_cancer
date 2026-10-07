#!/usr/bin/env python3
"""Entrenamiento de la CNN para predecir pCR (caso BreastDCEDL).

Qué hace en cada época:
  1. Entrena con los folds de entrenamiento (con aumento de datos).
  2. Evalúa en el fold de validación A NIVEL DE PACIENTE (media de sus cortes):
     AUC, sensibilidad, especificidad y matriz de confusión.
  3. Guarda los pesos si el AUC de validación mejora (se queda con la MEJOR época).
  4. Early stopping: para si el AUC no mejora en `--paciencia` épocas seguidas.

Al terminar guarda en resultados/entrenamiento/<nombre>/:
  config.json        todos los hiperparámetros, dispositivo, tiempos y métricas
  historial.csv      pérdida y métricas por época
  curvas.png         curvas de pérdida y AUC (para detectar sobreajuste)
  val_pacientes.csv  probabilidad por paciente de validación (para elegir umbral)
y los pesos en models/<nombre>.pt (con su checksum SHA-256 en config.json).

El conjunto de TEST no se usa aquí: se evalúa una sola vez al final, en otro script.

Uso, desde la raíz del repositorio:

    python src/train.py --rapido                    # prueba en CPU: pocos pacientes, 10 épocas
    python src/train.py                             # entrenamiento real (10 épocas)
    python src/train.py --epocas 30 --perdida normal --nombre normal_f0
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

sys.path.insert(0, os.path.dirname(__file__))
import preprocessing as pp                       # noqa: E402
from model import CNNpCR, contar_parametros      # noqa: E402

REPO = pp.REPO



# Utilidades

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



# Una época

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



# Gráficas

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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

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
    args = p.parse_args()

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
    }
    (carpeta / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    m = mejores_metricas
    print(f"\nMejor época: {mejor_epoca} | AUC val (paciente) = {mejor_auc:.3f}")
    print(f"Umbral 0,5    -> sens {m['sensibilidad']:.2f} | esp {m['especificidad']:.2f} | "
          f"VP {m['VP']} VN {m['VN']} FP {m['FP']} FN {m['FN']}")
    print(f"Umbral Youden {umbral:.3f} -> sens {met_umbral['sensibilidad']:.2f} | "
          f"esp {met_umbral['especificidad']:.2f}")
    print(f"Tiempo total: {t_total / 60:.1f} min ({hist.segundos.mean():.0f} s/época)")
    print(f"Pesos:      {ruta_pesos.relative_to(REPO)}")
    print(f"Resultados: {carpeta.relative_to(REPO)}")
    if args.rapido:
        print("\n(Modo rápido: las métricas NO son representativas; solo comprueba que todo funciona.)")


if __name__ == "__main__":
    main()
