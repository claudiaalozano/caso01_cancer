"use strict";

const FASES = ["PRE", "EARLY", "LATE"];
const LIMITE_BYTES = 2 * 1024 * 1024;

const contenedor = document.querySelector(".uploads");
const feedback = document.querySelector("#feedback");
const selector = document.querySelector("#example-select");
const infoEjemplo = document.querySelector("#example-info");
const botonLimpiar = document.querySelector("#reset");

const estados = new Map();
const tarjetas = new Map();

let solicitudEjemplo = 0;

/* CREAR LOS TRES VISORES */

FASES.forEach((fase, indice) => {
  const tarjeta = document.createElement("article");
  tarjeta.className = "upload";

  tarjeta.innerHTML = `
    <div class="upload-top">
      <strong>${fase}</strong>
      <span>0${indice + 1}</span>
    </div>

    <div class="preview">
      <span class="plus" aria-hidden="true">＋</span>
    </div>

    <label for="archivo-${fase}">
      <span class="filename">Seleccionar fase ${fase}</span>
    </label>

    <input
      id="archivo-${fase}"
      type="file"
      accept="image/png"
      aria-label="Cargar fase ${fase}"
    >

    <div class="filename status">Sin imagen</div>
  `;

  const input = tarjeta.querySelector("input");

  input.addEventListener("change", async () => {
    // Una carga manual cancela cualquier ejemplo pendiente.
    solicitudEjemplo++;
    selector.value = "";

    infoEjemplo.textContent =
      "Carga manual. Comprueba que las fases pertenecen " +
      "a la misma paciente y al mismo corte.";

    await cargarFase(fase, input.files[0]);
  });

  tarjetas.set(fase, tarjeta);
  contenedor.append(tarjeta);
});

/* ESTADO Y LIMPIEZA */

function liberarEstado(fase) {
  const anterior = estados.get(fase);

  if (anterior?.url) {
    URL.revokeObjectURL(anterior.url);
  }

  estados.delete(fase);
}

function vaciarVisor(fase) {
  liberarEstado(fase);

  const tarjeta = tarjetas.get(fase);
  const preview = tarjeta.querySelector(".preview");

  preview.replaceChildren();

  const simbolo = document.createElement("span");
  simbolo.className = "plus";
  simbolo.setAttribute("aria-hidden", "true");
  simbolo.textContent = "＋";

  preview.append(simbolo);

  tarjeta.querySelector("input").value = "";
  tarjeta.querySelector(".status").textContent = "Sin imagen";
}

function actualizarEstado() {
  const validas = FASES.filter(
    fase => estados.get(fase)?.valida
  ).length;

  if (validas === 3) {
    feedback.textContent =
      "Las tres fases tienen un formato válido. " +
      "Esto no confirma su correspondencia anatómica. " +
      "La predicción sigue pendiente de conectar al modelo.";
  } else {
    feedback.textContent =
      `${validas} de 3 fases válidas. ` +
      "Selecciona un ejemplo o carga las fases restantes.";
  }
}

/* VALIDACIÓN Y VISUALIZACIÓN */

async function cargarFase(fase, archivo) {
  vaciarVisor(fase);

  const tarjeta = tarjetas.get(fase);
  const status = tarjeta.querySelector(".status");
  const preview = tarjeta.querySelector(".preview");
  const token = Symbol(fase);

  estados.set(fase, {
    token,
    valida: false
  });

  status.textContent = "Validando imagen…";
  actualizarEstado();

  let url = null;

  try {
    if (!archivo) {
      throw new Error("Sin imagen seleccionada.");
    }

    if (archivo.size > LIMITE_BYTES) {
      throw new Error("El archivo supera el límite de 2 MB.");
    }

    const bytes = new Uint8Array(await archivo.arrayBuffer());

    const firmaPNG = [137, 80, 78, 71, 13, 10, 26, 10];

    const firmaValida = firmaPNG.every(
      (valor, indice) => bytes[indice] === valor
    );

    if (bytes.length < 33 || !firmaValida) {
      throw new Error("El archivo no es un PNG válido.");
    }

    const ihdr = String.fromCharCode(...bytes.slice(12, 16));

    if (ihdr !== "IHDR") {
      throw new Error("La cabecera del PNG no es válida.");
    }

    const cabecera = new DataView(bytes.buffer);

    const ancho = cabecera.getUint32(16);
    const alto = cabecera.getUint32(20);
    const profundidad = bytes[24];
    const tipoColor = bytes[25];

    if (ancho !== 256 || alto !== 256) {
      throw new Error("La imagen debe medir 256 × 256 píxeles.");
    }

    if (profundidad !== 8 || tipoColor !== 0) {
      throw new Error(
        "Se requiere PNG de 8 bits en escala de grises, sin alfa."
      );
    }

    url = URL.createObjectURL(archivo);

    const imagen = new Image();
    imagen.src = url;

    try {
      await imagen.decode();
    } catch {
      throw new Error("La imagen está dañada o no se puede leer.");
    }

    // Evita que una carga antigua sustituya una más reciente.
    if (estados.get(fase)?.token !== token) {
      URL.revokeObjectURL(url);
      return false;
    }

    imagen.alt = `Imagen DCE de la fase ${fase}`;

    preview.replaceChildren(imagen);

    estados.set(fase, {
      token,
      valida: true,
      url,
      archivo
    });

    status.textContent = `${archivo.name} · 256 × 256`;

    actualizarEstado();
    return true;
  } catch (error) {
    if (url) {
      URL.revokeObjectURL(url);
    }

    if (estados.get(fase)?.token === token) {
      estados.set(fase, {
        token,
        valida: false
      });

      status.textContent = error.message;
      actualizarEstado();
    }

    return false;
  }
}

/* EJEMPLOS YA INCLUIDOS EN LA WEB */

const ejemplos = Array.isArray(window.DCE_EXAMPLES)
  ? window.DCE_EXAMPLES
  : [];

ejemplos.forEach((ejemplo, indice) => {
  selector.add(
    new Option(ejemplo.label, String(indice))
  );
});

if (ejemplos.length === 0) {
  selector.disabled = true;

  infoEjemplo.textContent =
    "Todavía no hay ejemplos configurados. " +
    "Puedes cargar las tres fases manualmente.";
}

selector.addEventListener("change", async () => {
  const solicitud = ++solicitudEjemplo;

  // Limpia el estudio anterior para no mezclar pacientes.
  FASES.forEach(vaciarVisor);
  actualizarEstado();

  if (selector.value === "") {
    infoEjemplo.textContent = "Selecciona un ejemplo.";
    return;
  }

  const ejemplo = ejemplos[Number(selector.value)];

  infoEjemplo.textContent =
    "Cargando PRE, EARLY y LATE del ejemplo…";

  try {
    if (location.protocol === "file:") {
      throw new Error(
        "Abre la web con el servidor local o GitHub Pages. " +
        "Los ejemplos no se cargan abriendo index.html con doble clic."
      );
    }

    const archivos = await Promise.all(
      FASES.map(async fase => {
        const ruta = ejemplo.paths?.[fase];

        if (!ruta) {
          throw new Error(`Falta la ruta de la fase ${fase}.`);
        }

        const respuesta = await fetch(ruta);

        if (!respuesta.ok) {
          throw new Error(`No se pudo cargar la fase ${fase}.`);
        }

        const blob = await respuesta.blob();

        if (blob.size > LIMITE_BYTES) {
          throw new Error(`La fase ${fase} supera los 2 MB.`);
        }

        const nombre = ruta.split("/").pop();

        return new File([blob], nombre, {
          type: "image/png"
        });
      })
    );

    if (solicitud !== solicitudEjemplo) {
      return;
    }

    const validaciones = await Promise.all(
      FASES.map((fase, indice) =>
        cargarFase(fase, archivos[indice])
      )
    );

    if (solicitud !== solicitudEjemplo) {
      return;
    }

    infoEjemplo.textContent = validaciones.every(Boolean)
      ? `Ejemplo de entrenamiento: ${ejemplo.label}. ` +
        "No representa una evaluación independiente."
      : "Alguna fase no supera la validación. Revisa los mensajes.";
  } catch (error) {
    if (solicitud === solicitudEjemplo) {
      infoEjemplo.textContent = error.message;
    }
  }
});

/* BOTÓN LIMPIAR */

botonLimpiar.addEventListener("click", () => {
  solicitudEjemplo++;
  selector.value = "";

  FASES.forEach(vaciarVisor);

  infoEjemplo.textContent =
    "Selecciona un ejemplo incluido o carga imágenes propias.";

  actualizarEstado();
});

/* NAVEGACIÓN */

const enlaces = document.querySelectorAll("nav a");

enlaces.forEach(enlace => {
  enlace.addEventListener("click", () => {
    enlaces.forEach(item => item.classList.remove("active"));
    enlace.classList.add("active");
  });
});

/* LIBERAR RECURSOS */

window.addEventListener("pagehide", () => {
  FASES.forEach(liberarEstado);
});

actualizarEstado();