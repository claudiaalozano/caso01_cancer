(() => {
  "use strict";

  const escena = document.querySelector("#network-scene");
  if (!escena) {
    console.error("No se encuentra #network-scene");
    return;
  }

  /*
   * Sustituimos el visor anterior.
   * Clonar el contenedor elimina los manejadores antiguos asociados a él.
   */
  const visor = escena.cloneNode(false);
  escena.replaceWith(visor);

  const estilo = document.createElement("style");

  estilo.textContent = `
    #network-scene {
      display: block !important;
      position: relative;
      width: 100%;
      min-width: 0 !important;
      height: auto !important;
      min-height: 650px;
      padding: 0 !important;
      overflow: hidden !important;
      transform: none !important;
      perspective: none !important;
      background:
        radial-gradient(ellipse at 50% 40%, #12465d66, transparent 62%),
        linear-gradient(145deg, #07131f, #060a15);
      border: 1px solid #7beaff26;
      border-radius: 24px;
      color: #ecfaff;
      isolation: isolate;
    }

    .morph-header {
      position: relative;
      z-index: 2;
      padding: 26px 28px 0;
    }

    .morph-kicker {
      color: #6fe7ff;
      font-size: 11px;
      letter-spacing: .18em;
      text-transform: uppercase;
    }

    .morph-title {
      margin: 10px 0 6px;
      font-size: clamp(24px, 4vw, 36px);
      line-height: 1.15;
    }

    .morph-description {
      max-width: 650px;
      min-height: 44px;
      margin: 0;
      color: #a8bdcd;
      line-height: 1.6;
      font-size: 14px;
    }

    .morph-stage {
      position: relative;
      height: 370px;
      perspective: 1000px;
      touch-action: pan-y;
      cursor: grab;
    }

    .morph-stage:active {
      cursor: grabbing;
    }

    .morph-floor {
      position: absolute;
      left: 50%;
      top: 70%;
      width: 560px;
      height: 320px;
      transform: translate(-50%, -50%) rotateX(70deg);
      background-image:
        linear-gradient(#64dfff18 1px, transparent 1px),
        linear-gradient(90deg, #64dfff18 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: radial-gradient(ellipse, black, transparent 72%);
      pointer-events: none;
    }

    .morph-object {
      --w: 230px;
      --h: 230px;
      --d: 16px;
      --accent: #63eaff;
      position: absolute;
      left: 50%;
      top: 46%;
      width: var(--w);
      height: var(--h);
      transform-style: preserve-3d;
      transform:
        translate(-50%, -50%)
        rotateX(-15deg)
        rotateY(-30deg);
      transition:
        width 1400ms cubic-bezier(.22, 1, .36, 1),
        height 1400ms cubic-bezier(.22, 1, .36, 1);
    }

    .morph-face {
      position: absolute;
      box-sizing: border-box;
      border: 1px solid var(--accent);
      backface-visibility: visible;
      background-color: color-mix(in srgb, var(--accent) 12%, #071420 88%);
      background-image:
        linear-gradient(#ffffff18 1px, transparent 1px),
        linear-gradient(90deg, #ffffff18 1px, transparent 1px);
      background-size: 18px 18px;
      box-shadow:
        inset 0 0 35px color-mix(in srgb, var(--accent) 12%, transparent),
        0 0 15px color-mix(in srgb, var(--accent) 15%, transparent);
      transition:
        width 1400ms cubic-bezier(.22, 1, .36, 1),
        height 1400ms cubic-bezier(.22, 1, .36, 1),
        transform 1400ms cubic-bezier(.22, 1, .36, 1),
        background-color 1000ms,
        border-color 1000ms;
    }

    .morph-front,
    .morph-back {
      width: var(--w);
      height: var(--h);
    }

    .morph-front {
      transform: translateZ(calc(var(--d) / 2));
      display: grid;
      place-items: center;
    }

    .morph-front span {
      padding: 7px 10px;
      background: #06101bd9;
      border-radius: 7px;
      font: 600 13px ui-monospace, monospace;
      white-space: nowrap;
    }

    .morph-back {
      transform: rotateY(180deg) translateZ(calc(var(--d) / 2));
    }

    .morph-right,
    .morph-left {
      width: var(--d);
      height: var(--h);
      left: calc((var(--w) - var(--d)) / 2);
    }

    .morph-right {
      transform: rotateY(90deg) translateZ(calc(var(--w) / 2));
    }

    .morph-left {
      transform: rotateY(-90deg) translateZ(calc(var(--w) / 2));
    }

    .morph-top,
    .morph-bottom {
      width: var(--w);
      height: var(--d);
      top: calc((var(--h) - var(--d)) / 2);
    }

    .morph-top {
      transform: rotateX(90deg) translateZ(calc(var(--h) / 2));
    }

    .morph-bottom {
      transform: rotateX(-90deg) translateZ(calc(var(--h) / 2));
    }

    .morph-shape {
      position: absolute;
      bottom: 12px;
      width: 100%;
      text-align: center;
      color: #def9ff;
      font: 600 21px ui-monospace, monospace;
    }

    .morph-controls {
      position: relative;
      z-index: 2;
      padding: 0 28px 26px;
    }

    .morph-progress {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 7px;
      margin-bottom: 20px;
    }

    .morph-step {
      appearance: none;
      min-width: 0;
      padding: 11px 3px;
      border: 1px solid #ffffff16;
      border-radius: 10px;
      color: #8fa7ba;
      background: #ffffff04;
      cursor: pointer;
      font: inherit;
      font-size: 11px;
      transition: .25s;
    }

    .morph-step[aria-current="step"] {
      color: #8af0ff;
      border-color: #6fe7ff99;
      background: #43d9ff14;
      box-shadow: 0 0 22px #43d9ff12;
    }

    .morph-actions {
      display: flex;
      align-items: center;
      justify-content: center;
      flex-wrap: wrap;
      gap: 10px;
    }

    .morph-actions button {
      appearance: none;
      padding: 10px 16px;
      border: 1px solid #78e9ff40;
      border-radius: 10px;
      background: #102b3c;
      color: #e8faff;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }

    .morph-note {
      margin: 15px 0 0;
      color: #7f99ad;
      text-align: center;
      font-size: 11px;
      line-height: 1.6;
    }

    @media (max-width: 600px) {
      #network-scene {
        min-height: 620px;
      }

      .morph-header {
        padding: 22px 18px 0;
      }

      .morph-stage {
        height: 340px;
      }

      .morph-controls {
        padding: 0 14px 22px;
      }

      .morph-step {
        font-size: 9px;
      }

      .morph-description {
        min-height: 66px;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      .morph-object,
      .morph-face {
        transition-duration: 0ms;
      }
    }
  `;

  document.head.appendChild(estilo);

  const etapas = [
    {
      nombre: "Las tres fases de entrada",
      corto: "Entrada",
      descripcion:
        "PRE, EARLY y LATE forman tres canales. Son fases de resonancia, no colores RGB.",
      forma: "3 × 256 × 256",
      etiqueta: "PRE · EARLY · LATE",
      w: 230, h: 230, d: 16,
      color: "#63eaff"
    },
    {
      nombre: "Etapa 1 · primeros patrones",
      corto: "Etapa 1",
      descripcion:
        "Convolución, BatchNorm, ReLU y MaxPooling. Se obtienen 16 canales y la resolución baja a 128 × 128.",
      forma: "16 × 128 × 128",
      etiqueta: "16 canales",
      w: 205, h: 205, d: 36,
      color: "#63eaff"
    },
    {
      nombre: "Etapa 2 · combinación de patrones",
      corto: "Etapa 2",
      descripcion:
        "La siguiente convolución combina características y produce 32 canales. MaxPooling reduce la resolución a 64 × 64.",
      forma: "32 × 64 × 64",
      etiqueta: "32 canales",
      w: 177, h: 177, d: 60,
      color: "#6edbcf"
    },
    {
      nombre: "Etapa 3 · características más complejas",
      corto: "Etapa 3",
      descripcion:
        "La red obtiene 64 canales. Después del tercer MaxPooling, cada mapa mide 32 × 32.",
      forma: "64 × 32 × 32",
      etiqueta: "64 canales",
      w: 146, h: 146, d: 92,
      color: "#9295ff"
    },
    {
      nombre: "Etapa 4 · atención espacial",
      corto: "Etapa 4",
      descripcion:
        "La convolución genera 128 canales. La atención pondera las posiciones en mapas de 32 × 32, antes del pooling que los reduce a 16 × 16.",
      forma: "128 × 16 × 16",
      etiqueta: "Atención",
      w: 113, h: 113, d: 135,
      color: "#c084fc"
    },
    {
      nombre: "Etapa 5 · representación final",
      corto: "Etapa 5",
      descripcion:
        "La última etapa conserva 128 canales y reduce los mapas a 8 × 8. El volumen contiene las características que resumirá la cabeza.",
      forma: "128 × 8 × 8",
      etiqueta: "128 canales",
      w: 82, h: 82, d: 135,
      color: "#f49ac2"
    },
    {
      nombre: "Cabeza · puntuación por corte",
      corto: "Salida",
      descripcion:
        "Global Average Pooling resume cada canal. Dropout y una capa lineal producen un logit; la sigmoide lo convierte en una puntuación de pCR=1. La evaluación por paciente usa la media de sus cortes.",
      forma: "128 características → 1 puntuación",
      etiqueta: "pCR",
      w: 58, h: 58, d: 58,
      color: "#f6c76c"
    }
  ];

  visor.innerHTML = `
    <div class="morph-header">
      <div class="morph-kicker">Recorrido por la arquitectura</div>
      <h3 class="morph-title"></h3>
      <p class="morph-description"></p>
    </div>

    <div class="morph-stage"
         aria-label="Representación tridimensional giratoria de los mapas">
      <div class="morph-floor"></div>

      <div class="morph-object" aria-hidden="true">
        <div class="morph-face morph-front"><span></span></div>
        <div class="morph-face morph-back"></div>
        <div class="morph-face morph-right"></div>
        <div class="morph-face morph-left"></div>
        <div class="morph-face morph-top"></div>
        <div class="morph-face morph-bottom"></div>
      </div>

      <div class="morph-shape"></div>
    </div>

    <div class="morph-controls">
      <div class="morph-progress" aria-label="Etapas de la CNN"></div>

      <div class="morph-actions">
        <button type="button" data-action="prev">← Anterior</button>
        <button type="button" data-action="play">Pausar</button>
        <button type="button" data-action="next">Siguiente →</button>
      </div>

      <p class="morph-note">
        Arrastra para girar el volumen.
        Dimensiones visuales esquemáticas, no proporcionales.
        Animación explicativa, sin inferencia.
      </p>
    </div>
  `;

  /*
   * Ocultamos los controles del visor anterior.
   * Los nuevos controles están dentro de esta escena.
   */
  [
    "network-tabs",
    "network-rotation",
    "network-prev",
    "network-next",
    "network-play",
    "network-counter",
    "network-title",
    "network-description",
    "network-shape",
    "network-operation",
    "network-status"
  ].forEach(id => {
    const elemento = document.getElementById(id);
    if (elemento) elemento.hidden = true;
  });

  const objeto = visor.querySelector(".morph-object");
  const superficie = visor.querySelector(".morph-stage");
  const titulo = visor.querySelector(".morph-title");
  const descripcion = visor.querySelector(".morph-description");
  const forma = visor.querySelector(".morph-shape");
  const etiqueta = visor.querySelector(".morph-front span");
  const progreso = visor.querySelector(".morph-progress");
  const botonPlay = visor.querySelector('[data-action="play"]');

  const reducirMovimiento =
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let actual = 0;
  let reproduciendo = !reducirMovimiento;
  let temporizador = null;
  let giroX = -15;
  let giroY = -30;
  let arrastre = null;

  const botones = etapas.map((etapa, indice) => {
    const boton = document.createElement("button");

    boton.type = "button";
    boton.className = "morph-step";
    boton.textContent = etapa.corto;
    boton.addEventListener("click", () => {
      mostrar(indice);
      programar();
    });

    progreso.appendChild(boton);
    return boton;
  });

  function aplicarGiro() {
    objeto.style.transform = `
      translate(-50%, -50%)
      rotateX(${giroX}deg)
      rotateY(${giroY}deg)
    `;
  }

  function mostrar(indice) {
    actual = (indice + etapas.length) % etapas.length;
    const etapa = etapas[actual];

    objeto.style.setProperty("--w", `${etapa.w}px`);
    objeto.style.setProperty("--h", `${etapa.h}px`);
    objeto.style.setProperty("--d", `${etapa.d}px`);
    objeto.style.setProperty("--accent", etapa.color);

    titulo.textContent = etapa.nombre;
    descripcion.textContent = etapa.descripcion;
    forma.textContent = etapa.forma;
    etiqueta.textContent = etapa.etiqueta;

    botones.forEach((boton, i) => {
      if (i === actual) boton.setAttribute("aria-current", "step");
      else boton.removeAttribute("aria-current");
    });
  }

  function programar() {
    clearTimeout(temporizador);
    temporizador = null;

    botonPlay.textContent = reproduciendo ? "Pausar" : "Reproducir";
    botonPlay.setAttribute("aria-pressed", String(reproduciendo));

    if (!reproduciendo || document.hidden || arrastre) return;

    temporizador = setTimeout(() => {
      mostrar(actual + 1);
      programar();
    }, 4500);
  }

  visor.querySelector('[data-action="prev"]')
    .addEventListener("click", () => {
      mostrar(actual - 1);
      programar();
    });

  visor.querySelector('[data-action="next"]')
    .addEventListener("click", () => {
      mostrar(actual + 1);
      programar();
    });

  botonPlay.addEventListener("click", () => {
    reproduciendo = !reproduciendo;
    programar();
  });

  superficie.addEventListener("pointerdown", evento => {
    if (!evento.isPrimary || evento.button !== 0) return;

    arrastre = {
      id: evento.pointerId,
      x: evento.clientX,
      y: evento.clientY,
      giroX,
      giroY
    };

    superficie.setPointerCapture(evento.pointerId);
    programar();
  });

  superficie.addEventListener("pointermove", evento => {
    if (!arrastre || evento.pointerId !== arrastre.id) return;

    giroY = arrastre.giroY + (evento.clientX - arrastre.x) * 0.4;
    giroX = Math.max(
      -65,
      Math.min(65, arrastre.giroX - (evento.clientY - arrastre.y) * 0.3)
    );

    aplicarGiro();
  });

  function terminarArrastre(evento) {
    if (!arrastre || evento.pointerId !== arrastre.id) return;
    arrastre = null;
    programar();
  }

  superficie.addEventListener("pointerup", terminarArrastre);
  superficie.addEventListener("pointercancel", terminarArrastre);

  document.addEventListener("visibilitychange", programar);
  window.addEventListener("pagehide", () => clearTimeout(temporizador));
  window.addEventListener("pageshow", programar);

  mostrar(0);
  aplicarGiro();
  programar();
})();