/* Inferencia local. La predicción se habilita tras verificar referencias. */
(function () {
  "use strict";
  const $ = s => document.querySelector(s);
  const preparar = $("#model-prepare");
  const analizar = $("#predict");
  const estado = $("#model-status");
  const detalle = $("#model-checks");
  const titulo = $("#prediction-title");
  const explicacion = $("#prediction-description");
  const puntuacion = $("#prediction-score");
  const pacienteInput = $("#patient-id");
  const corteInput = $("#slice-id");
  const tabla = $("#patient-slices");
  const resumen = $("#patient-summary");
  const limpiar = $("#patient-clear");
  const exportar = $("#patient-export");
  let sesion = null, config = null, verificado = false, ocupado = false;
  let pacienteActivo = null;
  const cortes = new Map();
  const VERSION = "1.22.0";
  const CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${VERSION}/dist/`;
  const texto = (el, v) => { el.textContent = v; };

  async function obtener(ruta, tipo="json") {
    const r = await fetch(ruta,{cache:"no-store"});
    if(!r.ok) throw Error(`No se encuentra ${ruta} (HTTP ${r.status}).`);
    return tipo==="json" ? r.json() : r.arrayBuffer();
  }
  async function hash(bytes) {
    const digest=await crypto.subtle.digest("SHA-256",bytes);
    return Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,"0")).join("");
  }
  function iguales(a,b) {
    if(a===b) return true;
    if(!a||!b||typeof a!=="object"||typeof b!=="object") return false;
    const ka=Object.keys(a),kb=Object.keys(b);
    return ka.length===kb.length&&ka.every(k=>Object.hasOwn(b,k)&&iguales(a[k],b[k]));
  }
  async function cargarRuntime() {
    if(window.ort) return;
    await new Promise((resolve,reject)=>{
      const script=document.createElement("script");
      script.src=CDN+"ort.min.js";
      script.crossOrigin="anonymous";
      script.onload=resolve;
      script.onerror=()=>{script.remove();reject(Error("No se pudo descargar ONNX Runtime. Comprueba tu conexión."));};
      document.head.append(script);
    });
  }
  async function predecir(datos) {
    const input=new ort.Tensor("float32",datos,[1,3,256,256]);
    let salidas;
    try {
      salidas=await sesion.run({imagenes:input});
      const salida=salidas.probabilidad_pcr1;
      if(!salida||salida.data.length!==1) throw Error("Salida del modelo inesperada.");
      const valor=Number(salida.data[0]);
      if(!Number.isFinite(valor)||valor<0||valor>1) throw Error("Puntuación del modelo no válida.");
      return valor;
    } finally {
      input.dispose?.();
      if(salidas) Object.values(salidas).forEach(t=>t.dispose?.());
    }
  }
  function comprobarManifest(c) {
    if(c.experimento!=="cnn5_5pool_otsu_spatial_f1"||c.activacion!=="ReLU"||
       c.entrada?.tipo!=="float32"||!iguales(c.entrada.forma,[1,3,256,256])||
       !iguales(c.entrada.orden_canales,["PRE","EARLY","LATE"])||
       c.entrada.nombre!=="imagenes"||c.salida?.nombre!=="probabilidad_pcr1"||
       c.salida.sigmoide_incluida!==true||c.umbral!==0.3) {
      throw Error("El manifiesto no corresponde a la CNN y umbral seleccionados.");
    }
    PreprocesadoDCE.validarConfig(c.preprocesado);
  }
  function registrar(textoLinea) {
    const li=document.createElement("li");li.textContent=textoLinea;detalle.append(li);
  }
  function identificadores(entrada) {
    if(!entrada) return null;
    const encontrados=entrada.nombres.map((nombre,i)=>{
      const sufijo=`_${["PRE","EARLY","LATE"][i]}.png`;
      if(!nombre.endsWith(sufijo)) return null;
      return nombre.slice(0,-sufijo.length).match(/^(.*)_z(\d+)$/);
    });
    if(encontrados.every(Boolean)) {
      const [primero]=encontrados;
      if(encontrados.some(m=>m[1]!==primero[1]||m[2]!==primero[2])) throw Error("Las tres fases tienen identificadores de paciente o corte diferentes.");
      return {paciente:primero[1],corte:"z"+primero[2]};
    }
    return null;
  }
  function actualizarBoton() {
    analizar.disabled=!verificado||ocupado||!window.DCEEntrada?.obtener();
    preparar.disabled=ocupado||verificado;
  }
  function alCambiarEntrada() {
    const entrada=window.DCEEntrada?.obtener();
    let ids=null;
    try {ids=identificadores(entrada);} catch(error) {texto(explicacion,error.message);}
    pacienteInput.value=ids?.paciente||"";
    corteInput.value=ids?.corte||"";
    pacienteInput.readOnly=!!ids;corteInput.readOnly=!!ids;
    texto(puntuacion,"—");
    texto(titulo,verificado ? "Listo para analizar un corte" : "Modelo sin verificar en este navegador");
    if(entrada) {
      try {
        identificadores(entrada);
        texto(explicacion,verificado ? "Analiza el corte y añádelo al resumen de una paciente." : "Pulsa «Preparar modelo» para comprobar el pipeline con las referencias de Python.");
      } catch(error) {texto(explicacion,error.message);}
    } else texto(explicacion,"Selecciona las tres fases de la misma paciente y corte.");
    actualizarBoton();
  }
  function dibujarPaciente() {
    tabla.replaceChildren();
    for(const [id,corte] of cortes) {
      const tr=document.createElement("tr");
      for(const valor of [id,corte.probabilidad.toFixed(6)]) {
        const td=document.createElement("td");td.textContent=valor;tr.append(td);
      }
      const td=document.createElement("td"),boton=document.createElement("button");
      boton.type="button";boton.className="secondary";boton.textContent="Retirar";
      boton.setAttribute("aria-label",`Retirar corte ${id}`);
      boton.addEventListener("click",()=>{
        if(ocupado) return;
        cortes.delete(id);if(!cortes.size) pacienteActivo=null;dibujarPaciente();
      });
      td.append(boton);tr.append(td);tabla.append(tr);
    }
    limpiar.disabled=ocupado||!cortes.size;
    exportar.disabled=!cortes.size;
    if(!cortes.size) {texto(resumen,"Todavía no has añadido cortes de una paciente.");return;}
    const media=Array.from(cortes.values()).reduce((s,c)=>s+c.probabilidad,0)/cortes.size;
    const clase=media>=config.umbral ? 1 : 0;
    texto(resumen,
      `${pacienteActivo} · ${cortes.size} corte(s) aportado(s) · media ${media.toFixed(6)} · `+
      `clasificación pCR=${clase} con umbral ${config.umbral}. `+
      "La media solo incluye los cortes de esta tabla. Para reproducir la evaluación de una paciente debes aportar todos sus cortes previstos."
    );
  }

  preparar.addEventListener("click",async()=>{
    ocupado=true;verificado=false;detalle.replaceChildren();actualizarBoton();
    try {
      if(location.protocol==="file:") throw Error("Abre la web mediante http://localhost:8000, no con doble clic.");
      if(!crypto.subtle) throw Error("La comprobación necesita HTTPS o localhost.");
      texto(estado,"Leyendo modelo y referencias…");
      const [manifest,refs]=await Promise.all([
        obtener("models/modelo_config.json"),obtener("verificacion/referencias.json")
      ]);
      comprobarManifest(manifest);
      if(refs.experimento!==manifest.experimento||refs.onnx_sha256!==manifest.onnx_sha256||!iguales(refs.preprocesado,manifest.preprocesado)||!Array.isArray(refs.ejemplos)||refs.ejemplos.length!==6) {
        throw Error("Las referencias no corresponden al modelo y preprocesado cargados.");
      }
      const bytes=await obtener("models/cnn_pcr.onnx","bytes");
      if(await hash(bytes)!==manifest.onnx_sha256) throw Error("SHA-256 del ONNX incorrecto.");
      registrar("Identidad del modelo y configuración: correctas.");
      texto(estado,"Cargando motor de inferencia local… La primera vez puede tardar.");
      await cargarRuntime();
      ort.env.wasm.wasmPaths=CDN;
      ort.env.wasm.numThreads=1;
      ort.env.wasm.proxy=false;
      if(sesion) await sesion.release();
      sesion=await ort.InferenceSession.create(bytes,{executionProviders:["wasm"]});
      if(!iguales(sesion.inputNames,["imagenes"])||!iguales(sesion.outputNames,["probabilidad_pcr1"])) throw Error("Nombres de entrada o salida incompatibles.");
      let maxTensor=0,maxProb=0;
      for(const [i,ref] of refs.ejemplos.entries()) {
        texto(estado,`Verificando ${i+1}/${refs.ejemplos.length}: ${ref.ejemplo}…`);
        const [fases,binario]=await Promise.all([
          Promise.all(["PRE","EARLY","LATE"].map(async f=>PreprocesadoDCE.decodificarPNG(await obtener(ref.fases[f],"bytes")))),
          obtener(ref.tensor,"bytes")
        ]);
        if(await hash(binario)!==ref.tensor_sha256) throw Error("Tensor de referencia modificado: "+ref.ejemplo);
        if(binario.byteLength!==3*256*256*4||!iguales(ref.forma,[1,3,256,256])) throw Error("Tensor de referencia incompatible.");
        const {datos}=PreprocesadoDCE.procesar(fases,manifest.preprocesado);
        const vista=new DataView(binario);let errorTensor=0;
        for(let j=0;j<datos.length;j++) {
          const esperado=vista.getFloat32(j*4,true);
          if(!Number.isFinite(esperado)) throw Error("Referencia no finita.");
          errorTensor=Math.max(errorTensor,Math.abs(datos[j]-esperado));
        }
        if(errorTensor>1e-6) throw Error(`Preprocesado diferente en ${ref.ejemplo}: ${errorTensor}. No se habilita inferencia.`);
        const p=await predecir(datos),errorProb=Math.abs(p-ref.pytorch);
        if(!Number.isFinite(ref.pytorch)||errorProb>1e-5+1e-4*Math.abs(ref.pytorch)) throw Error(`Inferencia diferente en ${ref.ejemplo}: ${errorProb}.`);
        maxTensor=Math.max(maxTensor,errorTensor);maxProb=Math.max(maxProb,errorProb);
        registrar(`${ref.ejemplo}: tensor Δ ${errorTensor.toExponential(2)} · salida Δ ${errorProb.toExponential(2)}.`);
        await new Promise(resolve=>setTimeout(resolve,0));
      }
      config=manifest;verificado=true;
      texto(estado,`Modelo preparado. Seis referencias verificadas. Diferencia máxima: tensor ${maxTensor.toExponential(2)}; puntuación ${maxProb.toExponential(2)}.`);
      preparar.textContent="Modelo verificado";
      alCambiarEntrada();
    } catch(error) {
      texto(estado,"No se ha habilitado la predicción: "+error.message);
      preparar.textContent="Reintentar preparación";
      console.error(error);
    } finally {ocupado=false;actualizarBoton();}
  });

  analizar.addEventListener("click",async()=>{
    if(!verificado||ocupado) return;
    const entrada=window.DCEEntrada.obtener();
    if(!entrada) return;
    ocupado=true;actualizarBoton();dibujarPaciente();
    try {
      const auto=identificadores(entrada);
      const paciente=auto?.paciente||pacienteInput.value.trim();
      const id=auto?.corte||corteInput.value.trim();
      if(!paciente||!id) throw Error("Indica el identificador de paciente y de corte para agrupar correctamente.");
      if(pacienteActivo&&paciente!==pacienteActivo) throw Error("Ya hay cortes de otra paciente. Pulsa «Vaciar paciente» antes de cambiar.");
      texto(titulo,"Procesando el corte…");
      const inicio=performance.now();
      const {datos,informe}=PreprocesadoDCE.procesar(entrada.fases,config.preprocesado);
      const p=await predecir(datos);
      if(window.DCEEntrada.obtener()?.revision!==entrada.revision) {
        texto(titulo,"Las imágenes han cambiado");
        texto(explicacion,"Vuelve a analizar las fases actuales. No se ha añadido el resultado anterior.");
        return;
      }
      pacienteActivo=paciente;
      cortes.set(id,{probabilidad:p,nombres:entrada.nombres.slice(),informe,tiempo_ms:performance.now()-inicio});
      texto(puntuacion,(100*p).toFixed(2)+" %");
      texto(titulo,"Puntuación de pCR=1 · corte "+id);
      texto(explicacion,
        `Corte añadido a ${paciente}. Preprocesado e inferencia: ${(performance.now()-inicio).toFixed(0)} ms. `+
        "La salida es una puntuación del modelo, no una probabilidad clínica calibrada."
      );
      dibujarPaciente();
    } catch(error) {
      texto(titulo,"No se ha analizado el corte");texto(explicacion,error.message);texto(puntuacion,"—");
    } finally {ocupado=false;actualizarBoton();dibujarPaciente();}
  });

  limpiar.addEventListener("click",()=>{
    if(ocupado) return;
    cortes.clear();pacienteActivo=null;dibujarPaciente();
  });
  exportar.addEventListener("click",()=>{
    if(!cortes.size) return;
    const lista=Array.from(cortes,([corte,valor])=>({corte,...valor}));
    const media=lista.reduce((s,c)=>s+c.probabilidad,0)/lista.length;
    const reporte={experimento:config.experimento,onnx_sha256:config.onnx_sha256,paciente:pacienteActivo,
      numero_cortes:lista.length,media,umbral:config.umbral,clase_predicha:media>=config.umbral?1:0,
      cortes:lista,alcance:"Solo los cortes aportados por el usuario; no verifica que la paciente esté completa.",uso:"Educativo, sin validez clínica"};
    const url=URL.createObjectURL(new Blob([JSON.stringify(reporte,null,2)],{type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="resultado_paciente.json";a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.addEventListener("dce-input-change",alCambiarEntrada);
  alCambiarEntrada();dibujarPaciente();
})();
