import io

import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Predicción de respuesta pCR",
    layout="wide",
)

FASES = ("PRE", "EARLY", "LATE")


def leer_imagen(archivo, fase):
    """Valida el formato y devuelve imagen e identificador del corte."""
    nombre = archivo.name
    sufijo = f"_{fase}.PNG"

    if not nombre.upper().endswith(sufijo):
        raise ValueError(
            f"{fase}: el nombre debe terminar en _{fase}.png"
        )

    identificador = nombre[:-len(sufijo)]

    with Image.open(io.BytesIO(archivo.getvalue())) as imagen:
        if imagen.format != "PNG":
            raise ValueError(f"{fase}: el archivo debe ser PNG.")

        if imagen.mode != "L":
            raise ValueError(
                f"{fase}: debe ser una imagen gris de 8 bits."
            )

        if imagen.size != (256, 256):
            raise ValueError(
                f"{fase}: tamaño recibido {imagen.size}; "
                "se esperaba 256 × 256."
            )

        imagen.load()
        array = np.array(imagen, dtype=np.uint8)

    return array, identificador


st.title("Predicción de respuesta al tratamiento")
st.write(
    "Análisis de las fases PRE, EARLY y LATE de resonancia "
    "magnética para estudiar la respuesta patológica completa."
)

with st.sidebar:
    st.header("Estado de la aplicación")
    st.info("Modelo pendiente de conectar.")
    st.caption(
        "Prototipo académico. No destinado a decisiones clínicas."
    )

    st.divider()
    st.subheader("Etiquetas")
    st.write("**pCR=1:** respuesta patológica completa.")
    st.write("**pCR=0:** no se alcanza respuesta patológica completa.")


st.subheader("1. Cargar un corte")
st.write(
    "Selecciona las tres fases del mismo corte. "
    "Los archivos deben compartir el identificador y terminar "
    "en `_PRE.png`, `_EARLY.png` y `_LATE.png`."
)

columnas = st.columns(3)
archivos = {}

for columna, fase in zip(columnas, FASES):
    with columna:
        archivos[fase] = st.file_uploader(
            f"Fase {fase}",
            type=["png"],
            key=f"archivo_{fase}",
        )

if not all(archivo is not None for archivo in archivos.values()):
    st.info("Carga las tres fases para continuar.")
    st.stop()


imagenes = {}
identificadores = []

try:
    for fase in FASES:
        imagen, identificador = leer_imagen(
            archivos[fase], fase
        )
        imagenes[fase] = imagen
        identificadores.append(identificador)

    if len(set(identificadores)) != 1:
        raise ValueError(
            "Los nombres indican cortes o pacientes diferentes. "
            "Selecciona tres fases del mismo corte."
        )

except Exception as error:
    st.error(f"No se pueden utilizar estos archivos: {error}")
    st.stop()


st.success(
    f"Formato y nombres compatibles: {identificadores[0]}"
)
st.caption(
    "Esta comprobación no certifica la alineación anatómica "
    "entre las fases."
)

st.subheader("2. Revisar imágenes")
columnas = st.columns(3)

for columna, fase in zip(columnas, FASES):
    with columna:
        st.image(
            imagenes[fase],
            caption=fase,
            clamp=True,
            width="stretch",
        )

with st.expander("Información de las imágenes"):
    st.dataframe(
        [
            {
                "Fase": fase,
                "Tamaño": "256 × 256",
                "Mínimo": int(imagenes[fase].min()),
                "Máximo": int(imagenes[fase].max()),
                "Media": round(float(imagenes[fase].mean()), 2),
            }
            for fase in FASES
        ],
        hide_index=True,
    )

    st.caption(
        "Las estadísticas describen los archivos originales; "
        "no permiten determinar por sí solas su calidad."
    )


st.subheader("3. Predicción")
st.warning(
    "Todavía no hay un modelo conectado. "
    "No se han calculado probabilidades ni etiquetas."
)

st.button(
    "Predecir respuesta",
    disabled=True,
    width="stretch",
)

st.caption(
    "Actualmente se visualiza un corte. "
    "La predicción por paciente requerirá cargar sus distintos "
    "cortes y promediar las probabilidades."
)