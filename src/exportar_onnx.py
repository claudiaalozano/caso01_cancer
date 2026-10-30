"""Exporta la CNN elegida y compara PyTorch con ONNX.
No entrena ni consulta datos de test.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

import model3


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTO = "cnn5_5pool_otsu_spatial_f1"


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


class ModeloWeb(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, imagenes):
        return torch.sigmoid(self.cnn(imagenes))


def main():
    carpeta = REPO / "resultados" / "entrenamiento" / EXPERIMENTO
    config = json.loads(
        (carpeta / "config.json").read_text(encoding="utf-8")
    )

    arquitectura = config["arquitectura"]
    pesos = REPO / config["pesos"]["ruta"].replace("\\", "/")

    if not pesos.is_file():
        raise FileNotFoundError(f"No se encuentran los pesos: {pesos}")

    if sha256(pesos) != config["pesos"]["sha256"]:
        raise ValueError("Los pesos no coinciden con el SHA-256 guardado.")

    cnn = model3.CNNpCR(
        canales=tuple(arquitectura["canales"]),
        dropout=arquitectura["dropout"],
        representacion=arquitectura["representacion"],
        atencion=arquitectura["atencion"],
    ).cpu()

    if any(isinstance(capa, nn.LeakyReLU) for capa in cnn.modules()):
        raise ValueError("Este experimento requiere ReLU, no LeakyReLU.")

    parametros = sum(p.numel() for p in cnn.parameters())
    if parametros != arquitectura["parametros"]:
        raise ValueError(
            f"Parámetros incompatibles: {parametros} "
            f"frente a {arquitectura['parametros']}"
        )

    cnn.load_state_dict(
        torch.load(pesos, map_location="cpu", weights_only=True),
        strict=True,
    )

    modelo = ModeloWeb(cnn).eval()

    destino = REPO / "app" / "models"
    destino.mkdir(parents=True, exist_ok=True)

    ruta_onnx = destino / "cnn_pcr.onnx"
    ruta_temporal = destino / "cnn_pcr_temporal.onnx"

    torch.manual_seed(42)
    ejemplo = torch.rand(1, 3, 256, 256)

    print("Exportando en CPU...", flush=True)

    torch.onnx.export(
        modelo,
        ejemplo,
        str(ruta_temporal),
        input_names=["imagenes"],
        output_names=["probabilidad_pcr1"],
        opset_version=17,
        dynamo=False,
        external_data=False,
    )

    onnx.checker.check_model(str(ruta_temporal))

    sesion = ort.InferenceSession(
        str(ruta_temporal),
        providers=["CPUExecutionProvider"],
    )

    pruebas = [
        torch.zeros_like(ejemplo),
        torch.ones_like(ejemplo),
        ejemplo,
        torch.rand_like(ejemplo),
        torch.rand_like(ejemplo),
    ]

    diferencias = []

    with torch.inference_mode():
        for numero, entrada in enumerate(pruebas, start=1):
            referencia = modelo(entrada).numpy()

            resultado = sesion.run(
                ["probabilidad_pcr1"],
                {"imagenes": entrada.numpy()},
            )[0]

            np.testing.assert_allclose(
                resultado,
                referencia,
                rtol=1e-4,
                atol=1e-5,
            )

            diferencia = float(
                np.max(np.abs(resultado - referencia))
            )
            diferencias.append(diferencia)

            print(
                f"Prueba {numero}: diferencia máxima "
                f"{diferencia:.8g}",
                flush=True,
            )

    # Publicamos el archivo final solo si las comprobaciones pasan.
    ruta_temporal.replace(ruta_onnx)

    manifiesto = {
        "experimento": EXPERIMENTO,
        "archivo": ruta_onnx.name,
        "onnx_sha256": sha256(ruta_onnx),
        "pesos_sha256": sha256(pesos),
        "activacion": "ReLU",
        "entrada": {
            "nombre": "imagenes",
            "forma": [1, 3, 256, 256],
            "tipo": "float32",
            "orden_canales": ["PRE", "EARLY", "LATE"],
            "requiere_preprocesado": True,
        },
        "salida": {
            "nombre": "probabilidad_pcr1",
            "sigmoide_incluida": True,
        },
        "umbral": config["validacion_mejor_epoca"]["umbral_fijo"],
        "agregacion_paciente": "media de puntuaciones de sus cortes",
        "preprocesado": config["entrenamiento"]["preprocesado_espacial"],
        "verificacion": {
            "pruebas_sinteticas": len(pruebas),
            "diferencia_maxima": max(diferencias),
            "preprocesado_navegador_verificado": False,
            "inferencia_navegador_verificada": False,
        },
    }

    (destino / "modelo_config.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExportación y comprobación sintética correctas.")
    print(f"Modelo: {ruta_onnx}")
    print(f"Tamaño: {ruta_onnx.stat().st_size / 1024**2:.2f} MB")
    print("Pendiente: comprobar imágenes reales y conectar la web.")


if __name__ == "__main__":
    main()