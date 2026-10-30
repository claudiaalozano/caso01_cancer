"use strict";

/* ARQUITECTURA 3D */

(() => {
  const etapas = [
    {
      nombre: "Entrada",
      canales: 3,
      resolucion: 256,
      alto: 180,
      profundidad: 8,
      texto:
        "PRE, EARLY y LATE se apilan como tres canales temporales, " +
        "con procesamiento espacial y escala de intensidad comunes.",
      forma: "3 × 256 × 256",
      operacion: "Apilado de las tres fases procesadas"
    },
    {
      nombre: "Etapa 1",
      canales: 16,
      resolucion: 128,
      alto: 145,
      profundidad: 16,
      texto:
        "La primera convolución combina las fases en 16 mapas aprendidos. " +
        "El pooling reduce a la mitad la resolución espacial.",
      forma: "16 × 128 × 128",
      operacion: "Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2"
    },
    {
      nombre: "Etapa 2",
      canales: 32,
      resolucion: 64,
      alto: 112,
      profundidad: 24,
      texto:
        "La red genera 32 canales a partir de las características anteriores. " +
        "Son mapas aprendidos, no imágenes originales adicionales.",
      forma: "32 × 64 × 64",
      operacion: "Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2"
    },
    {
      nombre: "Etapa 3",
      canales: 64,
      resolucion: 32,
      alto: 86,
      profundidad: 34,
      texto:
        "Se obtienen 64 canales. La reducción espacial permite combinar " +
        "información, pero también puede perder detalles pequeños.",
      forma: "64 × 32 × 32",
      operacion: "Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2"
    },
    {
      nombre: "Etapa 4",
      canales: 128,
      resolucion: 16,
      alto: 65,
      profundidad: 46,
      atencion: true,
      texto:
        "La atención utiliza el promedio y el máximo entre canales para " +
        "calcular una máscara espacial. Actúa sobre mapas de 32 × 32 " +
        "antes del pooling. No garantiza localizar el tumor.",
      forma: "128 × 16 × 16 después del pooling",
      operacion: "Conv → BatchNorm → ReLU → Atención espacial → MaxPool"
    },
    {
      nombre: "Etapa 5",
      canales: 128,
      resolucion: 8,
      alto: 48,
      profundidad: 46,
      texto:
        "Los últimos 128 mapas tienen una resolución de 8 × 8. " +
        "Después se resumen mediante Global Average Pooling.",
      forma: "128 × 8 × 8",
      operacion: "Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2"
    },
    {
      nombre: "Salida",
      alto: 24,
      profundidad: 24,
      salida: true,
      texto:
        "GAP obtiene 128 valores. La capa lineal produce un logit y la " +
        "sigmoide una puntuación. Dropout está desactivado en evaluación. " +
        "Para evaluar una paciente se promedian las puntuaciones de sus cortes.",
      forma: "128 características → 1 logit → 1 puntuación",
      operacion: "GAP → Dropout → Linear → Sigmoide"
    }
  ];

  const escena = document.querySelector("#network-scene");
  const tabs = document.querySelector("#network-tabs");
  const rotacion = document.querySelector("#network-rotation");
  const anterior = document.querySelector("#network-prev");
  const siguiente = document.querySelector("#network-next");
  const reproducir = document.querySelector("#network-play");

  let actual = 0;
  let temporizador = null;

  const volumenes = [];
  const botones = [];

  function pausar() {
    if (temporizador !== null) {
      clearInterval(temporizador);
      temporizador = null;
    }
    reproducir.textContent = "Reproducir";
  }

  function seleccionar(indice) {
    actual = indice;
    const etapa = etapas[indice];

    document.querySelector("#network-counter").textContent =
      `PASO ${indice + 1} / ${etapas.length}`;

    document.querySelector("#network-title").textContent = etapa.nombre;
    document.querySelector("#network-description").textContent = etapa.texto;
    document.querySelector("#network-shape").textContent = etapa.forma;
    document.querySelector("#network-operation").textContent = etapa.operacion;

    document.querySelector("#network-status").textContent =
      `${etapa.nombre}: ${etapa.forma}. Representación esquemática.`;

    botones.forEach((boton, posicion) => {
      if (posicion === indice) {
        boton.setAttribute("aria-current", "step");
      } else {
        boton.removeAttribute("aria-current");
      }
    });

    volumenes.forEach((boton, posicion) => {
      boton.setAttribute("aria-pressed", String(posicion === indice));
    });

    anterior.disabled = indice === 0;
    siguiente.disabled = indice === etapas.length - 1;
  }

  etapas.forEach((etapa, indice) => {
    const item = document.createElement("div");
    item.className = "volume-item";

    const volumen = document.createElement("button");
    volumen.type = "button";
    volumen.className = "volume-button";
    volumen.setAttribute("aria-pressed", "false");
    volumen.setAttribute(
      "aria-label",
      `${etapa.nombre}: ${etapa.forma}. Mostrar explicación.`
    );

    volumen.style.setProperty("--width", etapa.salida ? "24px" : "48px");
    volumen.style.setProperty("--height", `${etapa.alto}px`);
    volumen.style.setProperty("--depth", `${etapa.profundidad}px`);
    volumen.style.setProperty(
      "--block-color",
      etapa.atencion ? "#ac91ff" : "#72f5ee"
    );
    volumen.style.setProperty(
      "--face-background",
      etapa.atencion
        ? "rgba(115,75,190,0.55)"
        : "rgba(30,105,125,0.55)"
    );

    ["front", "back", "right", "left", "top", "bottom"].forEach(cara => {
      const panel = document.createElement("span");
      panel.className = `volume-face volume-${cara}`;
      panel.setAttribute("aria-hidden", "true");

      if (cara === "front") {
        panel.textContent = etapa.salida ? "1" : etapa.canales;
      }

      volumen.append(panel);
    });

    const etiqueta = document.createElement("div");
    etiqueta.className = "volume-caption";
    etiqueta.textContent = etapa.nombre;

    const dimensiones = document.createElement("small");
    dimensiones.textContent = etapa.salida
      ? "1 puntuación"
      : `${etapa.resolucion} × ${etapa.resolucion}`;

    etiqueta.append(dimensiones);
    item.append(volumen, etiqueta);
    escena.append(item);
    volumenes.push(volumen);

    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "network-tab";
    tab.textContent = etapa.nombre;
    tabs.append(tab);
    botones.push(tab);

    [volumen, tab].forEach(boton => {
      boton.addEventListener("click", () => {
        pausar();
        seleccionar(indice);
      });
    });
  });

  rotacion.addEventListener("input", () => {
    escena.style.setProperty("--rotation", `${rotacion.value}deg`);
  });

  anterior.addEventListener("click", () => {
    pausar();
    seleccionar(Math.max(0, actual - 1));
  });

  siguiente.addEventListener("click", () => {
    pausar();
    seleccionar(Math.min(etapas.length - 1, actual + 1));
  });

  reproducir.addEventListener("click", () => {
    if (temporizador !== null) {
      pausar();
      return;
    }

    if (actual === etapas.length - 1) seleccionar(0);

    reproducir.textContent = "Pausar";

    temporizador = setInterval(() => {
      seleccionar(Math.min(etapas.length - 1, actual + 1));
      if (actual === etapas.length - 1) pausar();
    }, 5000);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pausar();
  });

  window.addEventListener("pagehide", pausar);
  seleccionar(0);
})();

/* CARGA Y VALIDACIÓN DE IMÁGENES */

const FASES = ["PRE", "EARLY", "LATE"];
const LIMITE_BYTES = 2 * 1024 * 1024;
const estados = new Map();
const tarjetas = new Map();

const contenedor = document.querySelector("#uploads");
const feedback = document.querySelector("#feedback");
const selector = document.querySelector("#example-select");
const infoEjemplo = document.querySelector("#example-info");

let solicitudEjemplo = 0;

FASES.forEach((fase, indice) => {
  const tarjeta = document.createElement("article");
  tarjeta.className = "upload";

  tarjeta.innerHTML = `
    <div class="upload-top">
      <strong>${fase}</strong><span>0${indice + 1}</span>
    </div>
    <div class="preview">
      <span class="plus" aria-hidden="true">＋</span>
    </div>
    <label for="file-${fase}">Seleccionar fase ${fase}</label>
    <input
      id="file-${fase}"
      type="file"
      accept="image/png"
      aria-label="Cargar ${fase}"
    >
    <div class="status">Sin imagen</div>
  `;

  tarjeta.querySelector("input").addEventListener("change", async event => {
    solicitudEjemplo++;
    selector.value = "";
    infoEjemplo.textContent =
      "Carga manual. Verifica que las fases corresponden a la misma paciente y corte.";
    await cargarFase(fase, event.target.files[0]);
  });

  tarjetas.set(fase, tarjeta);
  contenedor.append(tarjeta);
});

function vaciarFase(fase) {
  const anterior = estados.get(fase);
  if (anterior?.url) URL.revokeObjectURL(anterior.url);
  estados.delete(fase);

  const tarjeta = tarjetas.get(fase);
  const simbolo = document.createElement("span");
  simbolo.className = "plus";
  simbolo.setAttribute("aria-hidden", "true");
  simbolo.textContent = "＋";

  tarjeta.querySelector(".preview").replaceChildren(simbolo);
  tarjeta.querySelector("input").value = "";
  tarjeta.querySelector(".status").textContent = "Sin imagen";
}

function actualizarVisor() {
  const n = FASES.filter(fase => estados.get(fase)?.valida).length;

  feedback.textContent = n === 3
    ? "Tres fases con formato válido. Esto no verifica su alineación anatómica. " +
      "La predicción sigue pendiente de conectar."
    : `${n} de 3 fases válidas.`;

  dibujarLaboratorio();
}

async function cargarFase(fase, archivo) {
  vaciarFase(fase);

  const tarjeta = tarjetas.get(fase);
  const status = tarjeta.querySelector(".status");
  const token = Symbol(fase);

  estados.set(fase, { token, valida: false });
  status.textContent = "Validando…";
  actualizarVisor();

  let url = null;

  try {
    if (!archivo) throw new Error("Sin imagen.");
    if (archivo.size > LIMITE_BYTES) throw new Error("Supera los 2 MB.");

    const bytes = new Uint8Array(await archivo.arrayBuffer());
    const firma = [137, 80, 78, 71, 13, 10, 26, 10];

    if (
      bytes.length < 33 ||
      !firma.every((valor, i) => bytes[i] === valor) ||
      String.fromCharCode(...bytes.slice(12, 16)) !== "IHDR"
    ) {
      throw new Error("PNG inválido.");
    }

    const cabecera = new DataView(bytes.buffer);

    if (
      cabecera.getUint32(16) !== 256 ||
      cabecera.getUint32(20) !== 256
    ) {
      throw new Error("Debe medir 256 × 256.");
    }

    if (bytes[24] !== 8 || bytes[25] !== 0) {
      throw new Error("Debe ser PNG gris de 8 bits sin alfa.");
    }

    url = URL.createObjectURL(archivo);

    const imagen = new Image();
    imagen.src = url;

    try {
      await imagen.decode();
    } catch {
      throw new Error("Imagen dañada o no compatible.");
    }

    if (estados.get(fase)?.token !== token) {
      URL.revokeObjectURL(url);
      return false;
    }

    imagen.alt = `Fase DCE ${fase}`;
    tarjeta.querySelector(".preview").replaceChildren(imagen);

    estados.set(fase, {
      token,
      valida: true,
      url,
      imagen,
      pixeles: null
    });

    status.textContent = `${archivo.name} · 256 × 256`;
    actualizarVisor();
    return true;
  } catch (error) {
    if (url) URL.revokeObjectURL(url);

    if (estados.get(fase)?.token === token) {
      estados.set(fase, { token, valida: false });
      status.textContent = error.message;
      actualizarVisor();
    }

    return false;
  }
}

/* EJEMPLOS INCLUIDOS */

const ejemplos = Array.isArray(window.DCE_EXAMPLES)
  ? window.DCE_EXAMPLES
  : [];

ejemplos.forEach((ejemplo, i) => {
  selector.add(new Option(ejemplo.label, String(i)));
});

if (!ejemplos.length) {
  selector.disabled = true;
  infoEjemplo.textContent =
    "No hay ejemplos configurados. Puedes cargar las imágenes manualmente.";
}

selector.addEventListener("change", async () => {
  const solicitud = ++solicitudEjemplo;

  FASES.forEach(vaciarFase);
  actualizarVisor();

  if (selector.value === "") {
    infoEjemplo.textContent = "Selecciona un ejemplo.";
    return;
  }

  const ejemplo = ejemplos[Number(selector.value)];
  infoEjemplo.textContent = "Cargando las tres fases…";

  try {
    if (location.protocol === "file:") {
      throw new Error(
        "Abre http://localhost:8000 mediante el servidor local, no index.html con doble clic."
      );
    }

    const archivos = await Promise.all(
      FASES.map(async fase => {
        const ruta = ejemplo.paths?.[fase];
        if (!ruta) throw new Error(`Falta la ruta de ${fase}.`);

        const respuesta = await fetch(ruta);
        if (!respuesta.ok) throw new Error(`No se pudo cargar ${fase}.`);

        const blob = await respuesta.blob();
        if (blob.size > LIMITE_BYTES) throw new Error(`${fase} supera 2 MB.`);

        return new File([blob], ruta.split("/").pop(), {
          type: "image/png"
        });
      })
    );

    if (solicitud !== solicitudEjemplo) return;

    const validas = await Promise.all(
      FASES.map((fase, i) => cargarFase(fase, archivos[i]))
    );

    if (solicitud !== solicitudEjemplo) return;

    infoEjemplo.textContent = validas.every(Boolean)
      ? `Ejemplo de entrenamiento: ${ejemplo.label}. No es una evaluación independiente.`
      : "Alguna fase no supera las comprobaciones.";
  } catch (error) {
    if (solicitud === solicitudEjemplo) {
      infoEjemplo.textContent = error.message;
    }
  }
});

document.querySelector("#reset").addEventListener("click", () => {
  solicitudEjemplo++;
  selector.value = "";
  FASES.forEach(vaciarFase);
  infoEjemplo.textContent = "Selecciona un ejemplo o carga imágenes propias.";
  actualizarVisor();
});

/* COMPARADOR DE FASES Y DIFERENCIAS */

const canvas = document.querySelector("#lab-canvas");
const ctx = canvas.getContext("2d");
const modo = document.querySelector("#lab-mode");
const slider = document.querySelector("#lab-slider");

function pixelesFase(fase) {
  const estado = estados.get(fase);

  if (!estado.pixeles) {
    const auxiliar = document.createElement("canvas");
    auxiliar.width = 256;
    auxiliar.height = 256;

    const contexto = auxiliar.getContext("2d", {
      willReadFrequently: true
    });

    contexto.drawImage(estado.imagen, 0, 0);
    estado.pixeles = contexto.getImageData(0, 0, 256, 256).data;
  }

  return estado.pixeles;
}

function dibujarLaboratorio() {
  const comparar = modo.value === "compare";

  document.querySelector("#lab-slider-control").hidden = !comparar;
  document.querySelector("#lab-legend").hidden = comparar;

  document.querySelector("#lab-title").textContent = comparar
    ? "Una imagen, dos momentos"
    : modo.value === "early"
      ? "Realce temprano"
      : "Cambio tardío";

  document.querySelector("#lab-explanation").textContent = comparar
    ? "PRE a la izquierda y EARLY a la derecha. Mueve el separador para comparar posiciones. No se aplica registro."
    : "Diferencia píxel a píxel con signo conservado y escala fija de −255 a +255. No permite diagnosticar ni identificar por sí sola el tejido.";

  const status = document.querySelector("#lab-status");

  if (!FASES.every(fase => estados.get(fase)?.valida)) {
    ctx.fillStyle = "#050913";
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = "#94a4bd";
    ctx.textAlign = "center";
    ctx.font = "12px sans-serif";
    ctx.fillText("Carga las tres fases", 128, 128);

    status.textContent = "Esperando imágenes.";
    canvas.setAttribute("aria-label", "Visor esperando imágenes.");
    return;
  }

  try {
    if (comparar) {
      const x = Math.round(256 * Number(slider.value) / 100);

      ctx.drawImage(estados.get("EARLY").imagen, 0, 0);

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, x, 256);
      ctx.clip();
      ctx.drawImage(estados.get("PRE").imagen, 0, 0);
      ctx.restore();

      ctx.strokeStyle = "#72f5ee";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, 256);
      ctx.stroke();

      ctx.fillStyle = "#07101fcc";
      ctx.fillRect(5, 5, 42, 20);
      ctx.fillRect(194, 5, 57, 20);
      ctx.fillStyle = "#edf4ff";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("PRE", 13, 19);
      ctx.fillText("EARLY", 201, 19);

      status.textContent =
        `Separador al ${slider.value} %. PRE izquierda / EARLY derecha.`;

      canvas.setAttribute(
        "aria-label",
        `Comparación PRE y EARLY. Separador al ${slider.value} por ciento.`
      );
      return;
    }

    const inicial = pixelesFase(modo.value === "early" ? "PRE" : "EARLY");
    const posterior = pixelesFase(modo.value === "early" ? "EARLY" : "LATE");
    const salida = ctx.createImageData(256, 256);

    const neutro = [18, 23, 35];
    let minimo = 255;
    let maximo = -255;
    let suma = 0;

    for (let i = 0; i < inicial.length; i += 4) {
      const diferencia = posterior[i] - inicial[i];
      const intensidad = Math.abs(diferencia) / 255;
      const color = diferencia >= 0 ? [255, 85, 70] : [45, 135, 255];

      for (let c = 0; c < 3; c++) {
        salida.data[i + c] = Math.round(
          neutro[c] + intensidad * (color[c] - neutro[c])
        );
      }

      salida.data[i + 3] = 255;
      minimo = Math.min(minimo, diferencia);
      maximo = Math.max(maximo, diferencia);
      suma += diferencia;
    }

    ctx.putImageData(salida, 0, 0);

    const nombre = modo.value === "early" ? "EARLY − PRE" : "LATE − EARLY";

    status.textContent =
      `${nombre} · mínimo ${minimo} · máximo ${maximo} · ` +
      `media de todos los píxeles ${(suma / 65536).toFixed(2)}. ` +
      "Unidades del PNG, no unidades físicas de resonancia.";

    canvas.setAttribute(
      "aria-label",
      `Diferencia ${nombre}: azul negativo, gris sin cambio, rojo positivo.`
    );
  } catch {
    ctx.clearRect(0, 0, 256, 256);
    status.textContent = "No se pudieron leer los píxeles.";
  }
}

modo.addEventListener("change", dibujarLaboratorio);
slider.addEventListener("input", dibujarLaboratorio);

/* NAVEGACIÓN Y LIMPIEZA */

document.querySelectorAll("nav a").forEach(enlace => {
  enlace.addEventListener("click", () => {
    document.querySelectorAll("nav a").forEach(item => {
      item.classList.remove("active");
    });
    enlace.classList.add("active");
  });
});

window.addEventListener("pagehide", () => {
  FASES.forEach(vaciarFase);
});

window.addEventListener("pageshow", actualizarVisor);

actualizarVisor();