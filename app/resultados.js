"use strict";

(function () {
  function iniciarResultados() {
    const tituloCorte = document.getElementById("prediction-title");
    const puntuacionCorte = document.getElementById("prediction-score");
    const resultadoCombinado = document.getElementById("combined-result");

    if (!tituloCorte || !puntuacionCorte || !resultadoCombinado) {
      console.error("No se encuentran los elementos de resultados.");
      return;
    }

    // Se lee el umbral del mismo archivo que utiliza la CNN.
    let umbralCNN = null;

    fetch("./models/modelo_config.json", { cache: "no-store" })
      .then(function (respuesta) {
        if (!respuesta.ok) {
          throw new Error("No se pudo leer la configuración de la CNN.");
        }

        return respuesta.json();
      })
      .then(function (configuracion) {
        const valor = Number(configuracion.umbral);

        if (!Number.isFinite(valor) || valor < 0 || valor > 1) {
          throw new Error("El umbral de la CNN no es válido.");
        }

        umbralCNN = valor;
        actualizarResultadoCorte();
      })
      .catch(function (error) {
        console.error(error);
      });

    // Tarjetas independientes: no modifican el cálculo del modelo.
    const zonaTarjetas = document.createElement("div");
    zonaTarjetas.id = "tarjetas-predicciones";
    zonaTarjetas.hidden = true;

    resultadoCombinado.insertAdjacentElement("afterend", zonaTarjetas);

    const estilo = document.createElement("style");

    estilo.textContent = `
      #tarjetas-predicciones[hidden] {
        display: none !important;
      }

      #tarjetas-predicciones {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 20px;
        margin: 24px 0;
      }

      #tarjetas-predicciones .prediccion-tarjeta {
        min-width: 0;
        padding: 26px;
        border: 1px solid var(--color-prediccion);
        border-radius: 18px;
        background: #101a2c;
      }

      #tarjetas-predicciones .prediccion-modelo {
        display: block;
        margin-bottom: 18px;
        font-size: 14px;
        letter-spacing: 1px;
        color: #bcc9df;
      }

      #tarjetas-predicciones .prediccion-clase {
        display: block;
        margin-bottom: 16px;
        font-size: clamp(24px, 3vw, 36px);
        line-height: 1.2;
        color: var(--color-prediccion);
      }

      #tarjetas-predicciones .prediccion-significado {
        margin: 0 0 18px;
        line-height: 1.6;
        color: #edf3ff;
      }

      #tarjetas-predicciones .prediccion-puntuacion,
      #tarjetas-predicciones .prediccion-aviso {
        margin: 10px 0 0;
        line-height: 1.6;
        color: #bcc9df;
      }

      #tarjetas-predicciones .prediccion-aviso {
        font-size: 13px;
      }

      @media (max-width: 760px) {
        #tarjetas-predicciones {
          grid-template-columns: 1fr;
        }
      }
    `;

    document.head.append(estilo);

    function actualizarResultadoCorte() {
      if (umbralCNN === null) {
        return;
      }

      const tituloActual = tituloCorte.textContent.trim();

      // Solo cambia el título de un resultado calculado.
      // No reemplaza mensajes de error, espera o carga.
      const coincidencia = tituloActual.match(
        /^(?:Puntuación de pCR=1|Predicción del corte: pCR=[01])\s*·\s*corte\s+(.+)$/
      );

      if (!coincidencia) {
        return;
      }

      const textoPuntuacion = puntuacionCorte.textContent
        .trim()
        .replace(",", ".");

      const porcentaje = Number.parseFloat(textoPuntuacion);

      if (!Number.isFinite(porcentaje)) {
        return;
      }

      const clase = porcentaje / 100 >= umbralCNN ? 1 : 0;
      const nuevoTitulo =
        "Predicción del corte: pCR=" +
        clase +
        " · corte " +
        coincidencia[1];

      if (tituloCorte.textContent !== nuevoTitulo) {
        tituloCorte.textContent = nuevoTitulo;
      }
    }

    function crearTarjeta(nombre, porcentaje, clase) {
      const tarjeta = document.createElement("article");
      tarjeta.className = "prediccion-tarjeta";

      tarjeta.style.setProperty(
        "--color-prediccion",
        clase === 1 ? "#c6b3ff" : "#83eeee"
      );

      const etiqueta = document.createElement("span");
      etiqueta.className = "prediccion-modelo";
      etiqueta.textContent = nombre;

      const prediccion = document.createElement("strong");
      prediccion.className = "prediccion-clase";
      prediccion.textContent = "Predicción: pCR=" + clase;

      const significado = document.createElement("p");
      significado.className = "prediccion-significado";

      significado.textContent =
        clase === 1
          ? "Predice respuesta patológica completa tras el tratamiento."
          : "Predice que no se alcanzará respuesta patológica completa.";

      const puntuacion = document.createElement("p");
      puntuacion.className = "prediccion-puntuacion";

      puntuacion.textContent =
        "Puntuación de pCR=1: " +
        porcentaje.toFixed(2).replace(".", ",") +
        " %";

      const aviso = document.createElement("p");
      aviso.className = "prediccion-aviso";

      aviso.textContent =
        "Puntuación no calibrada. No indica cuánto cáncer hay. " +
        "Solo se incluyen los cortes aportados.";

      tarjeta.append(
        etiqueta,
        prediccion,
        significado,
        puntuacion,
        aviso
      );

      return tarjeta;
    }

    function actualizarTarjetas() {
      // Lee las predicciones ya calculadas por el motor original.
      // No ejecuta inferencia ni cambia puntuaciones o umbrales.
      const texto = resultadoCombinado.textContent;

      const cnn = texto.match(
        /Solo imágenes:\s*([\d.,]+)\s*%\s*→\s*pCR=([01])/
      );

      const combinado = texto.match(
        /CNN\s*\+\s*datos:\s*([\d.,]+)\s*%\s*→\s*pCR=([01])/
      );

      zonaTarjetas.replaceChildren();

      if (!cnn || !combinado) {
        zonaTarjetas.hidden = true;
        return;
      }

      const puntuacionCNN = Number(cnn[1].replace(",", "."));
      const puntuacionCombinada = Number(
        combinado[1].replace(",", ".")
      );

      if (
        !Number.isFinite(puntuacionCNN) ||
        !Number.isFinite(puntuacionCombinada)
      ) {
        zonaTarjetas.hidden = true;
        return;
      }

      zonaTarjetas.append(
        crearTarjeta(
          "SOLO IMÁGENES · CNN",
          puntuacionCNN,
          Number(cnn[2])
        ),
        crearTarjeta(
          "CNN + DATOS CLÍNICOS",
          puntuacionCombinada,
          Number(combinado[2])
        )
      );

      zonaTarjetas.hidden = false;
    }

    const observadorCorte = new MutationObserver(
      actualizarResultadoCorte
    );

    observadorCorte.observe(tituloCorte, {
      childList: true,
      characterData: true,
      subtree: true
    });

    observadorCorte.observe(puntuacionCorte, {
      childList: true,
      characterData: true,
      subtree: true
    });

    const observadorCombinado = new MutationObserver(
      actualizarTarjetas
    );

    observadorCombinado.observe(resultadoCombinado, {
      childList: true,
      characterData: true,
      subtree: true
    });

    actualizarResultadoCorte();
    actualizarTarjetas();
  }

  if (document.readyState === "loading") {
    document.addEventListener(
      "DOMContentLoaded",
      iniciarResultados,
      { once: true }
    );
  } else {
    iniciarResultados();
  }
})();