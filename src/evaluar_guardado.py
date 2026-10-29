"""Evalúa la CNN de cinco etapas con ReLU sin reentrenar."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import model3 as m
import preprocessing as pp
from distribucion_probabilidades import dibujar_distribucion


VERSION = "cinco_etapas_cinco_poolings_v1"


def guardar_json(ruta, datos):
    ruta.write_text(
        json.dumps(
            m.limpiar_json(datos),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def comprobar_arquitectura(modelo, arq):
    if arq.get("version") != VERSION:
        raise ValueError(
            "Este evaluador requiere la arquitectura "
            "cinco_etapas_cinco_poolings_v1."
        )

    if arq.get("representacion") != "fases":
        raise ValueError(
            "Este evaluador está preparado para la entrada fases."
        )

    if len(arq["canales"]) != 5:
        raise ValueError("Se esperaban cinco etapas.")

    if any(
        isinstance(capa, nn.LeakyReLU)
        for capa in modelo.modules()
    ):
        raise ValueError(
            "model3_relu.py todavía contiene LeakyReLU. "
            "Restaura ReLU antes de evaluar estos pesos."
        )

    relus = sum(
        isinstance(capa, nn.ReLU)
        for capa in modelo.modules()
    )
    pools = sum(
        isinstance(capa, nn.MaxPool2d)
        for capa in modelo.modules()
    )

    if relus != 5 or pools != 5:
        raise ValueError(
            f"Arquitectura inesperada: {relus} ReLU y "
            f"{pools} MaxPooling."
        )

    if m.contar_parametros(modelo) != arq["parametros"]:
        raise ValueError(
            "El número de parámetros no coincide con config.json."
        )


def cargar_loader(conjunto, entrenamiento, cfg, batch, workers):
    """No usa el filtro de exclusiones añadido posteriormente."""
    pp._requiere_utils()
    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)

    if conjunto == "test":
        filas = pp.uc.conjunto_test(samples)
    else:
        train, val = pp.uc.particion(
            samples,
            fold_val=entrenamiento["fold_validacion"],
        )
        filas = train if conjunto == "train" else val

    filas = filas.reset_index(drop=True)

    if filas.empty:
        raise ValueError("El conjunto está vacío.")

    dataset = pp.DatasetCortes(
        filas,
        transform=None,
        cache=None,
        indice_cache=None,
        config_espacial=cfg,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, filas


def comprobar_validacion_original(carpeta, tabla):
    ruta = carpeta / "val_pacientes.csv"

    if not ruta.exists():
        raise FileNotFoundError(
            "Falta val_pacientes.csv para comprobar "
            "la reproducción de la validación original."
        )

    original = pd.read_csv(ruta)
    original["patient_id"] = original["patient_id"].astype(str)
    nueva = tabla.reset_index()
    nueva["patient_id"] = nueva["patient_id"].astype(str)

    if original.patient_id.duplicated().any():
        raise ValueError("Pacientes duplicadas en el CSV original.")

    if set(original.patient_id) != set(nueva.patient_id):
        raise ValueError(
            "Las pacientes no coinciden con la validación original."
        )

    comparacion = original.merge(
        nueva,
        on="patient_id",
        validate="one_to_one",
        suffixes=("_original", "_nueva"),
    )

    if not np.array_equal(
        comparacion.pCR_original,
        comparacion.pCR_nueva,
    ):
        raise ValueError("Las etiquetas no coinciden.")

    diferencia = np.max(
        np.abs(
            comparacion.prob_original
            - comparacion.prob_nueva
        )
    )

    print(
        "Diferencia máxima respecto a probabilidades originales: "
        f"{diferencia:.6g}"
    )

    # Permite pequeñas diferencias numéricas entre dispositivos.
    if not np.allclose(
        comparacion.prob_original,
        comparacion.prob_nueva,
        atol=1e-3,
        rtol=1e-3,
    ):
        raise RuntimeError(
            "No se reproduce la validación original. "
            "Revisa arquitectura, activación y preprocesado "
            "antes de consultar el test."
        )

    return float(diferencia)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", required=True)
    parser.add_argument(
        "--conjunto",
        choices=["train", "val", "test"],
        default="val",
    )
    parser.add_argument("--umbral", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if (
        not np.isfinite(args.umbral)
        or not 0 <= args.umbral <= 1
        or args.batch < 1
        or args.workers < 0
    ):
        parser.error("Argumentos fuera de rango.")

    carpeta = (
        pp.REPO / "resultados" / "entrenamiento"
        / args.experimento
    )

    ruta_config = carpeta / "config.json"
    config = json.loads(
        ruta_config.read_text(encoding="utf-8")
    )

    arq = config["arquitectura"]
    entrenamiento = config["entrenamiento"]

    if entrenamiento.get("modo_rapido"):
        raise ValueError("No se evalúan pesos de modo rápido.")

    if "preprocesado_espacial" not in entrenamiento:
        raise ValueError("Falta el preprocesado original.")

    cfg = pp.ConfigEspacial(
        **entrenamiento["preprocesado_espacial"]
    )

    # Admite rutas relativas guardadas en Windows o Linux.
    ruta_relativa = config["pesos"]["ruta"].replace("\\", "/")
    pesos = pp.REPO / Path(ruta_relativa)

    if not pesos.exists():
        raise FileNotFoundError(f"Faltan los pesos: {pesos}")

    if m.sha256(pesos) != config["pesos"]["sha256"]:
        raise ValueError("No coincide el SHA-256 de los pesos.")

    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    modelo = m.CNNpCR(
        canales=tuple(arq["canales"]),
        dropout=arq["dropout"],
        representacion=arq["representacion"],
        atencion=arq.get("atencion", "ninguna"),
    )

    comprobar_arquitectura(modelo, arq)

    modelo.load_state_dict(
        torch.load(
            pesos,
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    modelo.to(dispositivo)
    modelo.eval()

    verificacion = carpeta / "validacion_relu_verificada.json"
    marca_test = carpeta / "evaluacion_test_iniciada.json"

    if args.conjunto == "test":
        ya_evaluado = (
            config.get("test") is not None
            or marca_test.exists()
            or any(carpeta.rglob("test_pacientes.csv"))
            or any(carpeta.rglob("metricas_test.json"))
        )

        if ya_evaluado:
            raise RuntimeError(
                "El test ya se inició o tiene resultados. "
                "Utiliza los archivos guardados."
            )

        if not verificacion.exists():
            raise RuntimeError(
                "Ejecuta primero --conjunto val para verificar "
                "que se reproducen las predicciones originales."
            )

        comprobacion = json.loads(
            verificacion.read_text(encoding="utf-8")
        )

        if (
            comprobacion["pesos_sha256"]
            != config["pesos"]["sha256"]
            or comprobacion["config_sha256"]
            != m.sha256(ruta_config)
        ):
            raise RuntimeError(
                "La verificación corresponde a otros pesos "
                "o a otra configuración."
            )

    fecha = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    salida = carpeta / (
        f"evaluacion_{args.conjunto}_u{args.umbral:g}_{fecha}"
    )
    salida.mkdir(parents=True)

    print(f"Experimento: {args.experimento}")
    print(f"Conjunto: {args.conjunto}")
    print(f"Dispositivo: {dispositivo}")
    print(f"Umbral: {args.umbral}")
    print("Arquitectura: cinco etapas con ReLU")
    print("Se utiliza el preprocesado de config.json.")

    if args.conjunto == "test":
        # Registra las decisiones antes de cargar imágenes de test.
        with marca_test.open("x", encoding="utf-8") as archivo:
            json.dump(
                {
                    "umbral": args.umbral,
                    "pesos_sha256": config["pesos"]["sha256"],
                    "fecha": datetime.now().isoformat(),
                    "salida": str(salida.relative_to(pp.REPO)),
                },
                archivo,
                indent=2,
            )

    loader, filas = cargar_loader(
        args.conjunto,
        entrenamiento,
        cfg,
        args.batch,
        args.workers,
    )

    print(
        f"Cortes: {len(filas)} | "
        f"Pacientes: {filas.patient_id.nunique()}",
        flush=True,
    )
    print("Calculando predicciones...", flush=True)

    perdida, probabilidades = m.predecir(
        modelo,
        loader,
        nn.BCEWithLogitsLoss(),
        dispositivo,
    )

    met, tabla = m.metricas_paciente(
        probabilidades,
        filas,
        args.umbral,
    )

    tabla.to_csv(
        salida / f"{args.conjunto}_pacientes.csv"
    )

    # Métricas de pCR=0, además de las de pCR=1.
    denominador = met["VN"] + met["FN"]
    met["recall_0"] = met["especificidad"]
    met["precision_0"] = (
        met["VN"] / denominador
        if denominador else float("nan")
    )
    met["recall_1"] = met["recall"]
    met["precision_1"] = met["precision"]
    met["exactitud_balanceada"] = (
        met["recall_0"] + met["recall_1"]
    ) / 2

    m.dibujar_matriz(
        met,
        salida / "matriz_confusion.png",
        f"{args.conjunto} · umbral {args.umbral:g}",
    )

    dibujar_distribucion(
        tabla,
        salida / "distribucion_probabilidades.png",
        umbral=args.umbral,
        titulo=f"{args.conjunto}: distribución por paciente",
    )

    resultado = {
        "conjunto": args.conjunto,
        "pacientes": len(tabla),
        "perdida_BCE_por_corte_sin_ponderar": perdida,
        "umbral": args.umbral,
        "metricas": met,
        "modo": "eval sin aumentos",
        "pesos_sha256": config["pesos"]["sha256"],
    }

    guardar_json(
        salida / f"metricas_{args.conjunto}.json",
        resultado,
    )

    if args.conjunto == "val":
        diferencia = comprobar_validacion_original(
            carpeta, tabla
        )
        guardar_json(
            verificacion,
            {
                "pesos_sha256": config["pesos"]["sha256"],
                "config_sha256": m.sha256(ruta_config),
                "diferencia_maxima": diferencia,
                "fecha": datetime.now().isoformat(),
            },
        )
        print("Validación original reproducida correctamente.")

    print(
        json.dumps(
            m.limpiar_json(resultado),
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Resultados: {salida}")


if __name__ == "__main__":
    main()