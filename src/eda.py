#!/usr/bin/env python3
"""Análisis exploratorio (EDA) del dataset BreastDCEDL — paso 1 del plan.

Antes de entrenar ninguna red hay que responder dos preguntas:

  A. AUDITORÍA  ¿Los datos son lo que creemos que son?
     (recuentos, separación por paciente, folds, imágenes, orden de fases)
  B. EXPLORACIÓN  ¿Qué nos dicen los datos?
     (desbalance, cohortes, variables clínicas, realce por clase, ejemplos)

Regla metodológica: todo lo que COMPARA clases (pCR=0 frente a pCR=1) se hace
solo con TRAIN. De test solo se cuentan pacientes y clases, para no "mirar" el
conjunto de evaluación final antes de tiempo.

Uso, desde la raíz del repositorio:

    python src/eda.py                    # dataset completo
    python src/eda.py --max-pacientes 5  # prueba rápida (5 pacientes por split y clase)

Salidas en resultados/eda/: resumen.json y figuras PNG.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# Rutas 

REPO = Path(__file__).resolve().parents[1]          
RAIZ_DATOS = REPO / "data" / "breastdcedl"
SALIDA = REPO / "resultados" / "eda"

sys.path.insert(0, str(RAIZ_DATOS))
try:
    import utils_caso as uc                         
except ModuleNotFoundError:
    sys.exit(f"No encuentro utils_caso.py en {RAIZ_DATOS}. "
             "Comprueba que está al mismo nivel que dataset/ y metadata/.")

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

UMBRAL_TEJIDO = 0.1                 # PRE > 0.1 = tejido (descarta el aire negro del fondo)
SEMILLA = 42
COLOR = {0: "#2a78d6", 1: "#eb6834"}               # pCR=0 azul, pCR=1 naranja
ETIQUETA = {0: "no pCR (0)", 1: "pCR (1)"}
GRIS = "#52514e"

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


# --------------------------------------------------------------------------- #
# Utilidades de impresión
# --------------------------------------------------------------------------- #

class Checks:
    """Acumula comprobaciones [OK] / [AVISO] / [FALLO] para el resumen final."""

    def __init__(self):
        self.lista: list[dict] = []

    def __call__(self, nombre: str, ok: bool, detalle: str = "", solo_aviso: bool = False):
        estado = "OK" if ok else ("AVISO" if solo_aviso else "FALLO")
        self.lista.append({"check": nombre, "estado": estado, "detalle": detalle})
        print(f"  [{estado:5}] {nombre}" + (f"  -> {detalle}" if detalle else ""))

    @property
    def fallos(self) -> int:
        return sum(c["estado"] == "FALLO" for c in self.lista)


def titulo(texto: str):
    print(f"\n=== {texto} " + "=" * max(0, 70 - len(texto)))


def tabla(df: pd.DataFrame | pd.Series):
    """Imprime una tabla de pandas con sangría."""
    print("  " + df.to_string().replace("\n", "\n  "))


# =========================================================================== #
# PARTE A. AUDITORÍA
# =========================================================================== #

def a1_recuentos(samples: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """A1. Cuántos datos tenemos.

    La unidad estadística es la PACIENTE, no el corte: ~10 cortes de la misma
    paciente son muy parecidos entre sí (observaciones correlacionadas).
    """
    titulo("A1. Recuentos por split y clase")
    por_pac = samples.groupby("patient_id").agg(
        split=("split", "first"), pCR=("pCR", "first"), fold=("fold", "first"),
        n_cortes=("sample_id", "size"))

    print("  Pacientes:")
    tabla(por_pac.groupby(["split", "pCR"]).size().unstack(fill_value=0))
    print("\n  Cortes:")
    tabla(samples.groupby(["split", "pCR"]).size().unstack(fill_value=0))

    resumen = {}
    for split in ("train", "test"):
        p = por_pac[por_pac.split == split]
        s = samples[samples.split == split]
        resumen[split] = {
            "pacientes": int(len(p)),
            "pacientes_pCR0": int((p.pCR == 0).sum()),
            "pacientes_pCR1": int((p.pCR == 1).sum()),
            "prop_pCR_pacientes": round(float(p.pCR.mean()), 4),
            "cortes": int(len(s)),
            "cortes_pCR0": int((s.pCR == 0).sum()),
            "cortes_pCR1": int((s.pCR == 1).sum()),
            "cortes_por_paciente": {"min": int(p.n_cortes.min()),
                                    "mediana": float(p.n_cortes.median()),
                                    "max": int(p.n_cortes.max())},
        }
        print(f"\n  {split}: {len(p)} pacientes, {len(s)} cortes; cortes/paciente "
              f"min={p.n_cortes.min()} mediana={p.n_cortes.median():.0f} max={p.n_cortes.max()}")
    return resumen, por_pac


def a2_separacion(samples: pd.DataFrame, check: Checks):
    """A2. Separación por paciente: la regla que invalida el trabajo si se rompe.

    Si cortes de una misma paciente caen en entrenamiento y en validación/test,
    la red reconoce anatomía ya vista y las métricas salen infladas (data leakage).
    """
    titulo("A2. Separación por paciente (fuga de datos)")
    n_split = samples.groupby("patient_id").split.nunique()
    check("ninguna paciente en dos splits", (n_split == 1).all(),
          f"{(n_split > 1).sum()} pacientes repetidas")

    train = samples[samples.split == "train"]
    test = samples[samples.split == "test"]
    n_fold = train.groupby("patient_id").fold.nunique()
    check("cada paciente de train en un único fold", (n_fold == 1).all(),
          f"{(n_fold > 1).sum()} pacientes en varios folds")
    check("folds de train entre 0 y 4", set(train.fold.unique()) <= set(range(5)))
    check("test tiene fold = -1 (no participa en la validación)", (test.fold == -1).all())

    n_etq = samples.groupby("patient_id").pCR.nunique()
    check("pCR constante dentro de cada paciente", (n_etq == 1).all(),
          f"{(n_etq > 1).sum()} pacientes con etiquetas mezcladas")
    check("pCR sin nulos y binaria", samples.pCR.notna().all() and set(samples.pCR) <= {0, 1})

    # La función del profesor también comprueba el solape; la llamamos como prueba
    for f in range(5):
        uc.particion(samples, fold_val=f)
    check("uc.particion() sin solape para los 5 folds", True)


def a3_folds(samples: pd.DataFrame, check: Checks) -> list[dict]:
    """A3. ¿Están equilibrados los folds?

    Si un fold tiene muchas más pCR que otro, las métricas de validación
    variarán entre folds por esa razón, no por el modelo.
    """
    titulo("A3. Folds de validación interna")
    train = samples[samples.split == "train"]
    pac = train.groupby("patient_id").agg(fold=("fold", "first"), pCR=("pCR", "first"))
    folds = pd.DataFrame({
        "pacientes": pac.groupby("fold").size(),
        "cortes": train.groupby("fold").size(),
        "prop_pCR_pacientes": pac.groupby("fold").pCR.mean().round(4),
    })
    tabla(folds)
    rango = folds.prop_pCR_pacientes.max() - folds.prop_pCR_pacientes.min()
    check("proporción de pCR similar entre folds (diferencia < 2 puntos)", rango < 0.02,
          f"diferencia = {rango:.1%}; tenlo en cuenta al comparar folds", solo_aviso=True)
    return folds.reset_index().to_dict(orient="records")


def _leer_png(ruta: Path) -> tuple[np.ndarray | None, str]:
    """Lee un PNG y valida formato, modo y tamaño. Devuelve (array, '') o (None, error)."""
    try:
        with Image.open(ruta) as png:
            if png.format != "PNG":
                return None, f"formato {png.format}"
            if png.mode != "L":
                return None, f"modo {png.mode} (se esperaba gris 'L')"
            if png.size != (256, 256):
                return None, f"tamaño {png.size}"
            return np.asarray(png, dtype=np.uint8), ""
    except FileNotFoundError:
        return None, "no existe"
    except Exception as e:                       # PNG corrupto, permisos...
        return None, f"ilegible: {str(e)[:60]}"


def _medir_corte(fila) -> dict:
    """Lee las 3 fases de un corte y mide la intensidad media en el tejido.

    Se valida cada PNG por separado y después se normaliza igual que
    uc.cargar_imagen (dividir entre 255), para medir exactamente lo que verá la red.
    """
    res = {"sample_id": fila.sample_id, "error": ""}
    fases = []
    for col in ("path_pre", "path_early", "path_late"):
        arr, err = _leer_png(RAIZ_DATOS / getattr(fila, col))
        if err:
            res["error"] = f"{getattr(fila, col)}: {err}"
            return res
        fases.append(arr)

    x = np.stack(fases).astype(np.float32) / 255.0          # (3, 256, 256) en [0, 1]
    tejido = x[0] > UMBRAL_TEJIDO
    if not tejido.any():
        res["error"] = "sin tejido (PRE <= 0.1 en todo el corte)"
        return res
    pre, early, late = (float(c[tejido].mean()) for c in x)
    res.update(min=float(x.min()), max=float(x.max()), frac_tejido=float(tejido.mean()),
               pre=pre, early=early, late=late, realce=early - pre)
    return res


def a4_imagenes(samples: pd.DataFrame, check: Checks, hilos: int) -> pd.DataFrame:
    """A4. Imágenes: existen, son PNG gris 256x256, [0,1], y el orden de fases es correcto.

    Comprobación fisiológica: al inyectar el contraste el tejido se ilumina,
    así que la intensidad media de EARLY debe superar a la de PRE. Si no, las
    fases estarían mal ordenadas o mal normalizadas.
    """
    titulo("A4. Imágenes y orden de las fases")
    filas = list(samples.itertuples(index=False))
    t0 = time.time()
    resultados = []
    with ThreadPoolExecutor(max_workers=hilos) as pool:
        for i, r in enumerate(pool.map(_medir_corte, filas), 1):
            resultados.append(r)
            if i % 500 == 0 or i == len(filas):
                print(f"\r  leídos {i}/{len(filas)} cortes ({3 * i} PNG)", end="", flush=True)
    print(f"   [{time.time() - t0:.0f} s]")

    medidas = pd.DataFrame(resultados)
    errores = medidas[medidas.error != ""]
    check(f"las {3 * len(samples)} imágenes existen y son PNG gris 8 bits 256x256",
          errores.empty, "" if errores.empty else
          f"{len(errores)} cortes con problemas, p. ej. {errores.error.iloc[0]}")
    if not errores.empty:
        SALIDA.mkdir(parents=True, exist_ok=True)
        errores.to_csv(SALIDA / "cortes_con_error.csv", index=False)

    ok = medidas[medidas.error == ""].drop(columns="error")
    check("valores en [0, 1] tras dividir entre 255",
          bool(((ok["min"] >= 0) & (ok["max"] <= 1)).all()))

    ok = ok.merge(samples[["sample_id", "patient_id", "split", "pCR"]], on="sample_id")
    pac = ok.groupby("patient_id")[["pre", "early", "late"]].mean()
    frac_pre_lt_early = float((pac.pre < pac.early).mean())
    check("PRE < EARLY en el tejido (orden de fases correcto)", frac_pre_lt_early == 1.0,
          f"{frac_pre_lt_early:.1%} de pacientes", solo_aviso=frac_pre_lt_early > 0.99)
    washout = float((pac.late < pac.early).mean())
    print(f"  Lavado (LATE < EARLY): {washout:.1%} de pacientes "
          "(la guía indica ~17 %; es un patrón real, no un error)")
    return ok


# =========================================================================== #
# PARTE B. EXPLORACIÓN
# =========================================================================== #

def b5_desbalance(recuentos: dict) -> dict:
    """B5. Desbalance de clases: la referencia mínima que el modelo debe superar."""
    titulo("B5. Desbalance de clases")
    n0, n1 = recuentos["train"]["cortes_pCR0"], recuentos["train"]["cortes_pCR1"]
    pw = n0 / n1
    base = 1 - recuentos["test"]["prop_pCR_pacientes"]
    print(f"  pos_weight = N0/N1 (cortes de train) = {n0}/{n1} = {pw:.3f}")
    print(f"  Un modelo que prediga SIEMPRE 0 tendría accuracy = {base:.1%} en test.")
    print("  -> La accuracy sola no sirve: hay que mirar sensibilidad, especificidad y AUC.")
    return {"pos_weight": round(pw, 4), "accuracy_siempre_0_test": round(base, 4)}


def b6_cohortes(por_pac: pd.DataFrame, medidas: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """B6. Cohortes (Duke, I-SPY1, I-SPY2): posible fuente de sesgo.

    Si las cohortes tienen tasas de pCR distintas Y sus imágenes se distinguen
    (escáner, protocolo, selección de cortes), la red podría aprender a
    reconocer la cohorte en lugar de la biología del tumor.
    """
    titulo("B6. Cohortes (solo train para comparar)")
    d = por_pac.join(patients.set_index("pid")[["dataset"]])
    print("  Pacientes por cohorte y split:")
    tabla(d.groupby(["dataset", "split"]).size().unstack(fill_value=0))

    tr = d[d.split == "train"]
    img = (medidas[medidas.split == "train"]
           .groupby("patient_id")[["pre", "early", "realce", "frac_tejido"]].mean())
    coh = tr.join(img).groupby("dataset").agg(
        pacientes=("pCR", "size"), tasa_pCR=("pCR", "mean"),
        intensidad_PRE=("pre", "mean"), intensidad_EARLY=("early", "mean"),
        realce_medio=("realce", "mean"), frac_tejido=("frac_tejido", "mean"))
    print("\n  Train por cohorte:")
    tabla(coh.round(3))
    return coh.round(4).to_dict(orient="index")


def b7_clinicas(por_pac: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """B7. Variables clínicas (solo para interpretar, NO son entradas de la red).

    El subtipo tumoral influye mucho en la respuesta: en la literatura, los
    tumores triple negativo y HER2+ alcanzan pCR con más frecuencia que HR+/HER2-.
    Las variables de raza no se analizan: tratarlas como causales es un error
    metodológico (lo advierte la guía).
    """
    titulo("B7. Variables clínicas (solo train)")
    cols = ["HR_HER2_STATUS", "age", "tum_vol", "menopause", "HR", "HER2"]
    tr = por_pac[por_pac.split == "train"].join(patients.set_index("pid")[cols])

    faltan = (patients[cols].isna().mean() * 100).round(1)
    print("  % de valores faltantes (todas las pacientes):")
    tabla(faltan)

    subtipo = tr.groupby("HR_HER2_STATUS").pCR.agg(pacientes="size", tasa_pCR="mean")
    print("\n  Tasa de pCR por subtipo:")
    tabla(subtipo.round(3))

    num = tr.groupby("pCR")[["age", "tum_vol"]].median()
    print("\n  Mediana de edad y volumen tumoral por clase:")
    tabla(num.round(2))
    return {"faltantes_pct": faltan.to_dict(),
            "pCR_por_subtipo": subtipo.round(4).to_dict(orient="index"),
            "medianas_por_clase": num.round(3).to_dict(orient="index")}


def b8_realce_por_clase(medidas: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """B8. ¿Se ve alguna diferencia de realce entre pCR=0 y pCR=1?

    Si ya hay diferencias en medias simples, la CNN tiene señal que aprender.
    Si no se ven, es una pista de lo difícil que es el problema.
    """
    titulo("B8. Realce y lavado por clase (solo train)")
    tr = medidas[medidas.split == "train"]
    pac = tr.groupby("patient_id").agg(pre=("pre", "mean"), early=("early", "mean"),
                                       late=("late", "mean"), realce=("realce", "mean"),
                                       pCR=("pCR", "first"))
    pac["lavado"] = pac.late < pac.early
    res = pac.groupby("pCR").agg(pacientes=("realce", "size"), realce_medio=("realce", "mean"),
                                 realce_mediana=("realce", "median"), prop_lavado=("lavado", "mean"))
    tabla(res.round(4))
    return res.round(4).to_dict(orient="index"), pac


# =========================================================================== #
# FIGURAS
# =========================================================================== #

def _guardar(fig, nombre: str):
    fig.tight_layout()
    fig.savefig(SALIDA / nombre, bbox_inches="tight")
    plt.close(fig)


def fig_balance(por_pac: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    t = por_pac.groupby(["split", "pCR"]).size().unstack(fill_value=0).reindex(["train", "test"])
    x = np.arange(len(t))
    for k, c in enumerate((0, 1)):
        barras = ax.bar(x + (k - 0.5) * 0.38, t[c], 0.36, color=COLOR[c], label=ETIQUETA[c])
        ax.bar_label(barras, padding=2, fontsize=9)
    ax.set_xticks(x, t.index)
    ax.set_ylabel("pacientes")
    ax.set_title("Pacientes por split y clase")
    ax.legend(frameon=False)
    _guardar(fig, "balance_clases.png")


def _barras_tasa(serie_tasa: pd.Series, serie_n: pd.Series, global_: float, titulo_: str, nombre: str):
    orden = serie_tasa.sort_values().index
    fig, ax = plt.subplots(figsize=(6.5, 0.6 * len(orden) + 1.6))
    barras = ax.barh(orden, serie_tasa[orden], color=COLOR[1], height=0.55)
    ax.bar_label(barras, labels=[f"{serie_tasa[i]:.0%}  (n={serie_n[i]})" for i in orden],
                 padding=3, fontsize=9)
    ax.axvline(global_, color=GRIS, ls="--", lw=1)
    ax.text(global_, -0.6, f" global {global_:.0%}", fontsize=8, color=GRIS, va="bottom")
    ax.set_xlim(0, max(0.6, serie_tasa.max() * 1.35))
    ax.set_xlabel("proporción de pacientes con pCR (train)")
    ax.set_title(titulo_)
    ax.grid(axis="y", visible=False)
    _guardar(fig, nombre)


def fig_pcr_por_cohorte(por_pac: pd.DataFrame, patients: pd.DataFrame):
    tr = por_pac[por_pac.split == "train"].join(patients.set_index("pid")[["dataset"]])
    g = tr.groupby("dataset").pCR
    _barras_tasa(g.mean(), g.size(), tr.pCR.mean(), "Tasa de pCR por cohorte", "pcr_por_cohorte.png")


def fig_pcr_por_subtipo(por_pac: pd.DataFrame, patients: pd.DataFrame):
    tr = (por_pac[por_pac.split == "train"]
          .join(patients.set_index("pid")[["HR_HER2_STATUS"]]).dropna(subset=["HR_HER2_STATUS"]))
    g = tr.groupby("HR_HER2_STATUS").pCR
    _barras_tasa(g.mean(), g.size(), tr.pCR.mean(), "Tasa de pCR por subtipo tumoral",
                 "pcr_por_subtipo.png")


def fig_cortes_por_paciente(por_pac: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cuentas = por_pac.n_cortes.value_counts().sort_index()
    ax.bar(cuentas.index, cuentas.values, color=COLOR[0], width=0.7)
    ax.set_xticks(cuentas.index)
    ax.set_xlabel("cortes por paciente")
    ax.set_ylabel("pacientes")
    ax.set_title("Cortes por paciente (observaciones correlacionadas)")
    _guardar(fig, "cortes_por_paciente.png")


def fig_pre_vs_early(pac: pd.DataFrame):
    """Cada punto es una paciente: si todas quedan sobre la diagonal, PRE < EARLY."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c in (0, 1):
        s = pac[pac.pCR == c]
        ax.scatter(s.pre, s.early, s=10, alpha=0.6, color=COLOR[c], label=ETIQUETA[c],
                   edgecolors="none")
    lo = min(pac.pre.min(), pac.early.min()) * 0.95
    hi = max(pac.pre.max(), pac.early.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=GRIS, lw=1, ls="--")
    ax.text(hi, hi, "EARLY = PRE ", ha="right", va="bottom", fontsize=8, color=GRIS)
    ax.set_xlabel("intensidad media PRE en tejido")
    ax.set_ylabel("intensidad media EARLY en tejido")
    ax.set_title("Orden de fases: todas las pacientes sobre la diagonal")
    ax.legend(frameon=False, loc="lower right")
    _guardar(fig, "pre_vs_early.png")


def fig_realce_por_clase_y_cohorte(pac: pd.DataFrame, patients: pd.DataFrame):
    d = pac.join(patients.set_index("pid")[["dataset"]])
    cohortes = sorted(d.dataset.dropna().unique())
    pos = np.arange(len(cohortes))
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, c in enumerate((0, 1)):
        datos = [d[(d.dataset == coh) & (d.pCR == c)].realce.values for coh in cohortes]
        datos = [v if len(v) else np.array([np.nan]) for v in datos]
        bp = ax.boxplot(datos, positions=pos + (k - 0.5) * 0.36, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops={"color": "white", "lw": 1.5})
        for caja in bp["boxes"]:
            caja.set(facecolor=COLOR[c], edgecolor=COLOR[c])
        for elem in ("whiskers", "caps"):
            for linea in bp[elem]:
                linea.set(color=COLOR[c])
    ax.set_xticks(pos, cohortes)
    ax.set_ylabel("realce medio EARLY - PRE (por paciente)")
    ax.set_title("Realce por cohorte y clase (train)")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in (0, 1)],
              [ETIQUETA[c] for c in (0, 1)], frameon=False)
    _guardar(fig, "realce_por_clase_y_cohorte.png")


def fig_ejemplos(samples: pd.DataFrame, validos: set[str]):
    """2 pacientes de cada clase (train), corte central: PRE, EARLY, LATE y realce."""
    rng = np.random.default_rng(SEMILLA)
    tr = samples[(samples.split == "train") & samples.sample_id.isin(validos)]
    elegidos = []
    for c in (0, 0, 1, 1):
        candidatos = sorted(set(tr[tr.pCR == c].patient_id) - {p for p, _ in elegidos})
        if not candidatos:
            print(f"  (no hay pacientes de train con pCR={c} para la figura de ejemplos)")
            return
        elegidos.append((rng.choice(candidatos), c))

    fig, axes = plt.subplots(4, 4, figsize=(10, 10.5))
    for fila_ax, (pid, c) in zip(axes, elegidos):
        cortes = tr[tr.patient_id == pid].sort_values("slice_index")
        fila = cortes.iloc[len(cortes) // 2]
        x = uc.cargar_imagen(fila, RAIZ_DATOS)          # misma carga que usará la red
        for ax, img, nombre in zip(fila_ax[:3], x, ("PRE", "EARLY", "LATE")):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(nombre, fontsize=9)
        im = fila_ax[3].imshow(np.clip(x[1] - x[0], 0, None), cmap="magma", vmin=0, vmax=0.4)
        fila_ax[3].set_title("realce EARLY - PRE", fontsize=9)
        fila_ax[0].set_ylabel(f"{fila.sample_id}\n{ETIQUETA[c]}", fontsize=8, color=COLOR[c])
        for ax in fila_ax:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.colorbar(im, ax=axes[:, 3], shrink=0.5, label="realce")
    fig.suptitle("Ejemplos de train (corte central de cada paciente)", fontsize=11)
    fig.savefig(SALIDA / "ejemplos_pcr0_vs_pcr1.png", bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
# MAIN
# =========================================================================== #

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-pacientes", type=int, metavar="N",
                   help="leer solo las imágenes de N pacientes por split y clase (prueba rápida)")
    p.add_argument("--hilos", type=int, default=8, help="hilos para leer imágenes")
    args = p.parse_args()

    if not (RAIZ_DATOS / "metadata" / "samples.csv").exists():
        sys.exit(f"No encuentro {RAIZ_DATOS / 'metadata' / 'samples.csv'}.")
    SALIDA.mkdir(parents=True, exist_ok=True)

    samples = uc.cargar_samples(RAIZ_DATOS)
    patients = uc.cargar_patients(RAIZ_DATOS)
    print(f"Datos:  {RAIZ_DATOS}\nSalida: {SALIDA}")

    check = Checks()

    # ---- Parte A: auditoría ----
    recuentos, por_pac = a1_recuentos(samples)
    a2_separacion(samples, check)
    folds = a3_folds(samples, check)

    muestras_img = samples
    if args.max_pacientes:
        elegidas = (samples.drop_duplicates("patient_id")
                    .groupby(["split", "pCR"]).head(args.max_pacientes).patient_id)
        muestras_img = samples[samples.patient_id.isin(elegidas)]
        print(f"\n  (MODO PRUEBA: imágenes de {muestras_img.patient_id.nunique()} pacientes)")
    medidas = a4_imagenes(muestras_img, check, args.hilos)

    # ---- Parte B: exploración ----
    desbalance = b5_desbalance(recuentos)
    cohortes = b6_cohortes(por_pac, medidas, patients)
    clinicas = b7_clinicas(por_pac, patients)
    realce, pac_img = b8_realce_por_clase(medidas)

    # ---- Figuras ----
    titulo("Figuras")
    fig_balance(por_pac)
    fig_pcr_por_cohorte(por_pac, patients)
    fig_pcr_por_subtipo(por_pac, patients)
    fig_cortes_por_paciente(por_pac)
    fig_pre_vs_early(pac_img)
    fig_realce_por_clase_y_cohorte(pac_img, patients)
    fig_ejemplos(samples, set(medidas.sample_id))
    for f in sorted(SALIDA.glob("*.png")):
        print(f"  {f.name}")

    # ---- Resumen ----
    resumen = {
        "modo_prueba": bool(args.max_pacientes),
        "recuentos": recuentos,
        "folds": folds,
        "desbalance": desbalance,
        "cohortes_train": cohortes,
        "clinicas_train": clinicas,
        "realce_por_clase_train": realce,
        "checks": check.lista,
    }
    (SALIDA / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    titulo("Resultado")
    print(f"  {len(check.lista)} comprobaciones, {check.fallos} fallos. "
          f"Detalle en {SALIDA / 'resumen.json'}")
    sys.exit(1 if check.fallos else 0)


if __name__ == "__main__":
    main()
