"""Compara PyTorch y ONNX con los ejemplos incluidos en la app.
No consulta test ni entrena.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import model3
import preprocessing as pp


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTO = "cnn5_5pool_otsu_spatial_f1"


def sha256(ruta):
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def main():
    carpeta = REPO / "resultados" / "entrenamiento" / EXPERIMENTO
    config = json.loads(
        (carpeta / "config.json").read_text(encoding="utf-8")
    )

    pesos = REPO / config["pesos"]["ruta"].replace("\\", "/")
    archivo_onnx = REPO / "app" / "models" / "cnn_pcr.onnx"
    carpeta_ejemplos = REPO / "app" / "examples"

    if sha256(pesos) != config["pesos"]["sha256"]:
        raise ValueError("La huella de los pesos no coincide.")

    manifiesto = json.loads(
        (archivo_onnx.parent / "modelo_config.json").read_text(
            encoding="utf-8"
        )
    )

    if sha256(archivo_onnx) != manifiesto["onnx_sha256"]:
        raise ValueError("La huella del ONNX no coincide.")

    arq = config["arquitectura"]

    modelo = model3.CNNpCR(
        canales=tuple(arq["canales"]),
        dropout=arq["dropout"],
        representacion=arq["representacion"],
        atencion=arq["atencion"],
    ).cpu()

    modelo.load_state_dict(
        torch.load(pesos, map_location="cpu", weights_only=True),
        strict=True,
    )
    modelo.eval()

    cfg = pp.ConfigEspacial(
        **config["entrenamiento"]["preprocesado_espacial"]
    )

    sesion = ort.InferenceSession(
        str(archivo_onnx),
        providers=["CPUExecutionProvider"],
    )

    ejemplos = sorted(carpeta_ejemplos.glob("*_PRE.png"))

    if not ejemplos:
        raise FileNotFoundError(
            f"No hay ejemplos *_PRE.png en {carpeta_ejemplos}"
        )

    # Referencias para comprobar después el navegador.
    destino = REPO / "app" / "verificacion"
    destino.mkdir(parents=True, exist_ok=True)

    resultados = []

    for pre in ejemplos:
        nombre = pre.name.removesuffix("_PRE.png")
        early = pre.with_name(nombre + "_EARLY.png")
        late = pre.with_name(nombre + "_LATE.png")

        if not early.is_file() or not late.is_file():
            raise FileNotFoundError(
                f"Faltan fases para el ejemplo {nombre}"
            )

        # Usa directamente el preprocesado original de Python.
        entrada = pp.cargar_corte(
            pre,
            early,
            late,
            config_espacial=cfg,
        ).unsqueeze(0).cpu().float()

        with torch.inference_mode():
            referencia = torch.sigmoid(modelo(entrada)).numpy()

        prediccion = sesion.run(
            ["probabilidad_pcr1"],
            {"imagenes": entrada.numpy()},
        )[0]

        np.testing.assert_allclose(
            prediccion,
            referencia,
            rtol=1e-4,
            atol=1e-5,
        )

        diferencia = float(
            np.max(np.abs(prediccion - referencia))
        )

        # Tensor de referencia: float32 little-endian, orden NCHW.
        archivo_tensor = destino / f"{nombre}.bin"
        tensor = entrada.numpy().astype("<f4")
        archivo_tensor.write_bytes(tensor.tobytes(order="C"))

        registro = {
            "ejemplo": nombre,
            "fases": {
                "PRE": f"examples/{pre.name}",
                "EARLY": f"examples/{early.name}",
                "LATE": f"examples/{late.name}",
            },
            "tensor": f"verificacion/{archivo_tensor.name}",
            "tensor_sha256": sha256(archivo_tensor),
            "forma": list(tensor.shape),
            "dtype": "float32",
            "orden": "NCHW",
            "pytorch": float(referencia.item()),
            "onnx": float(prediccion.item()),
            "diferencia": diferencia,
        }
        resultados.append(registro)

        print(
            f"{nombre}\n"
            f"  PyTorch:   {registro['pytorch']:.8f}\n"
            f"  ONNX:      {registro['onnx']:.8f}\n"
            f"  Diferencia: {diferencia:.8g}",
            flush=True,
        )

    informe = {
        "experimento": EXPERIMENTO,
        "onnx_sha256": sha256(archivo_onnx),
        "preprocesado": cfg.to_dict(),
        "ejemplos": resultados,
        "pytorch_onnx_python_verificados": True,
        "navegador_verificado": False,
        "nota": (
            "Comprobación técnica de equivalencia, "
            "no evaluación del rendimiento."
        ),
    }

    ruta_informe = destino / "referencias.json"
    ruta_informe.write_text(
        json.dumps(informe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nCorrecto: {len(resultados)} ejemplos comprobados.")
    print(f"Referencias: {ruta_informe}")
    print("Pendiente: reproducir y verificar el pipeline en navegador.")


if __name__ == "__main__":
    main()