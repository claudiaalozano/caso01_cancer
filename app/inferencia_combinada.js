"use strict";
document.addEventListener("DOMContentLoaded", function () {
  try {
/* Pipeline del experimento cnn5_5pool_otsu_spatial_f1.
 * No utiliza canvas para leer intensidades: conserva los valores del PNG.
 * Configuración admitida: Otsu + componentes 8-conexas + caja común,
 * resize bilineal uint8 compatible con Pillow + escala_comun.
 */
(function (global) {
  "use strict";
  const SIZE = 256, N = SIZE * SIZE, F = Math.fround;
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let v = i;
    for (let j = 0; j < 8; j++) v = (v & 1) ? (0xedb88320 ^ (v >>> 1)) : (v >>> 1);
    crcTable[i] = v >>> 0;
  }
  function crc32(bytes, start, end) {
    let crc = 0xffffffff;
    for (let i = start; i < end; i++) crc = crcTable[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }
  function paeth(a, b, c) {
    const p = a + b - c, pa = Math.abs(p-a), pb = Math.abs(p-b), pc = Math.abs(p-c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  }

  async function decodificarPNG(fuente) {
    const bytes = fuente instanceof Uint8Array ? fuente : new Uint8Array(fuente);
    const firma = [137,80,78,71,13,10,26,10];
    if (bytes.length < 33 || !firma.every((x,i) => bytes[i] === x)) throw Error("PNG inválido.");
    if (bytes.length > 2*1024*1024) throw Error("El PNG supera los 2 MB.");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let pos = 8, entrelazado = 0, cabecera = false, final = false;
    const partes = [];
    while (pos + 12 <= bytes.length) {
      const len = view.getUint32(pos);
      if (pos+12+len > bytes.length) throw Error("PNG truncado.");
      const tipo = String.fromCharCode(...bytes.subarray(pos+4,pos+8));
      if (crc32(bytes,pos+4,pos+8+len) !== view.getUint32(pos+8+len)) throw Error("CRC del PNG incorrecto.");
      const inicio = pos+8;
      if (!cabecera && tipo !== "IHDR") throw Error("Falta la cabecera PNG.");
      if (tipo === "IHDR") {
        if (cabecera || len !== 13 || view.getUint32(inicio) !== SIZE || view.getUint32(inicio+4) !== SIZE ||
            bytes[inicio+8] !== 8 || bytes[inicio+9] !== 0 || bytes[inicio+10] !== 0 || bytes[inicio+11] !== 0 || bytes[inicio+12] > 1) {
          throw Error("Se requiere PNG gris de 8 bits, 256 × 256.");
        }
        entrelazado = bytes[inicio+12]; cabecera = true;
      } else if (tipo === "IDAT") partes.push(bytes.subarray(inicio,inicio+len));
      else if (tipo === "IEND") { final = true; break; }
      else if (tipo === "acTL" || tipo === "tRNS") throw Error("No se admiten PNG animados o con transparencia.");
      else if (tipo[0] === tipo[0].toUpperCase() && tipo !== "PLTE") throw Error("Chunk PNG no compatible: " + tipo);
      pos += len+12;
    }
    if (!cabecera || !final || !partes.length) throw Error("PNG incompleto.");
    if (typeof DecompressionStream === "undefined") throw Error("Este navegador no permite descomprimir PNG. Usa Chrome o Edge actualizado.");
    const stream = new Blob(partes).stream().pipeThrough(new DecompressionStream("deflate"));
    // Límite de salida: evita expandir un archivo malformado sin límite.
    const reader = stream.getReader(), trozos = [];
    let total = 0;
    while (true) {
      const {value,done} = await reader.read();
      if (done) break;
      total += value.length;
      if (total > 70000) { await reader.cancel(); throw Error("Datos PNG demasiado grandes."); }
      trozos.push(value);
    }
    const raw = new Uint8Array(total);
    let offset=0;
    for (const trozo of trozos) { raw.set(trozo,offset); offset += trozo.length; }
    const salida = new Uint8Array(N);
    const pasos = entrelazado ? [[0,0,8,8],[4,0,8,8],[0,4,4,8],[2,0,4,4],[0,2,2,4],[1,0,2,2],[0,1,1,2]] : [[0,0,1,1]];
    pos=0;
    for (const [x0,y0,dx,dy] of pasos) {
      const w=Math.ceil((SIZE-x0)/dx), h=Math.ceil((SIZE-y0)/dy);
      let anterior = new Uint8Array(w);
      for (let y=0;y<h;y++) {
        if (pos+w+1 > raw.length) throw Error("Datos PNG incompletos.");
        const filtro=raw[pos++], fila=new Uint8Array(w);
        if (filtro > 4) throw Error("Filtro PNG desconocido.");
        for (let x=0;x<w;x++) {
          const a=x ? fila[x-1] : 0, b=anterior[x], c=x ? anterior[x-1] : 0;
          const predictor = filtro===0 ? 0 : filtro===1 ? a : filtro===2 ? b : filtro===3 ? Math.floor((a+b)/2) : paeth(a,b,c);
          fila[x]=(raw[pos++]+predictor)&255;
          salida[(y0+y*dy)*SIZE+x0+x*dx]=fila[x];
        }
        anterior=fila;
      }
    }
    if (pos !== raw.length) throw Error("Longitud PNG inesperada.");
    return salida;
  }

  // Otsu con la misma recurrencia y criterio de desempate de OpenCV.
  function otsu(imagen) {
    const hist = new Uint32Array(256);
    for (const p of imagen) hist[p]++;
    const escala=1/imagen.length;
    let media=0;
    for (let i=0;i<256;i++) media+=i*hist[i];
    media*=escala;
    let mu1=0,q1=0,maximo=0,umbral=0;
    const eps=1.1920928955078125e-7;
    for (let i=0;i<256;i++) {
      const pi=hist[i]*escala;
      mu1*=q1; q1+=pi;
      const q2=1-q1;
      if (Math.min(q1,q2)<eps || Math.max(q1,q2)>1-eps) continue;
      mu1=(mu1+i*pi)/q1;
      const mu2=(media-q1*mu1)/q2;
      const sigma=q1*q2*(mu1-mu2)*(mu1-mu2);
      if (sigma>maximo) { maximo=sigma; umbral=i; }
    }
    return umbral;
  }
  function medianaBorde(imagen) {
    const valores=[];
    for (let y=0;y<SIZE;y++) for(let x=0;x<SIZE;x++) {
      if(y<8) valores.push(imagen[y*SIZE+x]);
      if(y>=SIZE-8) valores.push(imagen[y*SIZE+x]);
      if(x<8) valores.push(imagen[y*SIZE+x]);
      if(x>=SIZE-8) valores.push(imagen[y*SIZE+x]);
    }
    valores.sort((a,b)=>a-b);
    return (valores[valores.length/2-1]+valores[valores.length/2])/2;
  }
  function cajaROI(fases,cfg) {
    const mascara=new Uint8Array(N), visitado=new Uint8Array(N), cola=new Int32Array(N);
    const umbrales=fases.map(otsu);
    if(fases.some(f=>medianaBorde(f)>200)) return {caja:[0,0,256,256],umbrales,motivo:"Borde claro: imagen completa"};
    for(let i=0;i<N;i++) mascara[i]=fases.some((f,j)=>f[i]>umbrales[j]) ? 1 : 0;
    let xmin=256,ymin=256,xmax=-1,ymax=-1,areaTotal=0;
    for(let p=0;p<N;p++) {
      if(!mascara[p] || visitado[p]) continue;
      let ini=0,fin=1;
      cola[0]=p;visitado[p]=1;
      let xa=256,ya=256,xb=-1,yb=-1;
      while(ini<fin) {
        const v=cola[ini++],y=Math.floor(v/SIZE),x=v%SIZE;
        xa=Math.min(xa,x);xb=Math.max(xb,x);ya=Math.min(ya,y);yb=Math.max(yb,y);
        for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
          const xx=x+dx,yy=y+dy;
          if(xx<0||xx>=SIZE||yy<0||yy>=SIZE) continue;
          const vecino=yy*SIZE+xx;
          if(mascara[vecino]&&!visitado[vecino]) {visitado[vecino]=1;cola[fin++]=vecino;}
        }
      }
      if(fin>=cfg.area_minima_roi) {
        areaTotal+=fin;xmin=Math.min(xmin,xa);xmax=Math.max(xmax,xb);ymin=Math.min(ymin,ya);ymax=Math.max(ymax,yb);
      }
    }
    if(areaTotal/N<0.05) return {caja:[0,0,256,256],umbrales,motivo:"ROI insuficiente: imagen completa"};
    return {caja:[Math.max(0,xmin-cfg.margen),Math.max(0,ymin-cfg.margen),Math.min(SIZE,xmax+1+cfg.margen),Math.min(SIZE,ymax+1+cfg.margen)],umbrales,motivo:"Caja común Otsu"};
  }
  function redondeoPython(x) {
    const a=Math.floor(x),d=x-a;
    return d===0.5 ? (a%2===0 ? a : a+1) : Math.round(x);
  }
  // Coeficientes bilineales normalizados a 22 bits como Pillow L/uint8.
  function coeficientes(entrada,salida) {
    const escala=entrada/salida, filtro=Math.max(1,escala), result=[];
    for(let i=0;i<salida;i++) {
      const centro=(i+0.5)*escala;
      const inicio=Math.max(0,Math.trunc(centro-filtro+0.5));
      const fin=Math.min(entrada,Math.trunc(centro+filtro+0.5));
      const pesos=[];let suma=0;
      for(let j=inicio;j<fin;j++) {const w=Math.max(0,1-Math.abs((j-centro+0.5)/filtro));pesos.push(w);suma+=w;}
      result.push({inicio,pesos:pesos.map(w=>Math.trunc(0.5+w/suma*4194304))});
    }
    return result;
  }
  function resizePillow(imagen,w,h,nw,nh) {
    let intermedio=imagen;
    if(nw!==w) {
      intermedio=new Uint8Array(nw*h);const cs=coeficientes(w,nw);
      for(let y=0;y<h;y++) for(let x=0;x<nw;x++) {
        const c=cs[x];let suma=2097152;
        for(let k=0;k<c.pesos.length;k++) suma+=imagen[y*w+c.inicio+k]*c.pesos[k];
        intermedio[y*nw+x]=Math.max(0,Math.min(255,Math.floor(suma/4194304)));
      }
    }
    if(nh===h) return intermedio;
    const result=new Uint8Array(nw*nh),cs=coeficientes(h,nh);
    for(let y=0;y<nh;y++) for(let x=0;x<nw;x++) {
      const c=cs[y];let suma=2097152;
      for(let k=0;k<c.pesos.length;k++) suma+=intermedio[(c.inicio+k)*nw+x]*c.pesos[k];
      result[y*nw+x]=Math.max(0,Math.min(255,Math.floor(suma/4194304)));
    }
    return result;
  }
  function validarConfig(cfg) {
    if(cfg.registrar!==false || cfg.recortar!==true || cfg.roi!==true || cfg.metodo_roi!=="otsu" || cfg.normalizacion!=="escala_comun" || cfg.clahe!==false ||
       !Number.isInteger(cfg.margen)||cfg.margen<0||!Number.isInteger(cfg.area_minima_roi)||cfg.area_minima_roi<1) {
      throw Error("Configuración no compatible con este pipeline web. No se aplicará una aproximación.");
    }
  }
  function procesar(fases,cfg) {
    validarConfig(cfg);
    if(fases.length!==3||fases.some(f=>!(f instanceof Uint8Array)||f.length!==N)) throw Error("Se necesitan tres fases uint8 de 256 × 256.");
    const informe=cajaROI(fases,cfg),[x0,y0,x1,y1]=informe.caja;
    const h=y1-y0,w=x1-x0,escala=Math.min(SIZE/h,SIZE/w);
    const nh=Math.min(SIZE,Math.max(1,redondeoPython(h*escala)));
    const nw=Math.min(SIZE,Math.max(1,redondeoPython(w*escala)));
    const top=Math.floor((SIZE-nh)/2),left=Math.floor((SIZE-nw)/2);
    const salida=new Float32Array(3*N),valores=[];
    for(let c=0;c<3;c++) {
      const recorte=new Uint8Array(w*h);
      for(let y=0;y<h;y++) recorte.set(fases[c].subarray((y+y0)*SIZE+x0,(y+y0)*SIZE+x1),y*w);
      const ampliada=resizePillow(recorte,w,h,nw,nh);
      for(let y=0;y<nh;y++) for(let x=0;x<nw;x++) {
        const v=F(ampliada[y*nw+x]/255);
        salida[c*N+(y+top)*SIZE+x+left]=v;
        if(v>0) valores.push(v);
      }
    }
    valores.sort((a,b)=>a-b);
    let limite=0;
    if(valores.length) {
      const indice=(valores.length-1)*0.995,lo=Math.floor(indice),hi=Math.ceil(indice);
      // NumPy interpola sobre valores float32; la división conserva float32.
      const a=valores[lo],b=valores[hi],peso=indice-lo,d=F(b-a);
      limite=peso<0.5 ? a+d*peso : b-d*(1-peso);
      const divisor=F(limite);
      if(divisor>0) for(let i=0;i<salida.length;i++) salida[i]=Math.max(0,Math.min(1,F(salida[i]/divisor)));
    }
    informe.limite_comun_01=limite;
    informe.redimensionado=[nh,nw];informe.padding=[top,left];
    return {datos:salida,informe};
  }
  global.PreprocesadoDCE=Object.freeze({decodificarPNG,procesar,validarConfig,resizePillow,otsu});
})(globalThis);

"use strict";

/* La arquitectura animada se gestiona en cnn3d.js. */

/* CARGA Y VALIDACIÓN DE IMÁGENES */

const FASES = ["PRE", "EARLY", "LATE"];
const LIMITE_BYTES = 2 * 1024 * 1024;
const estados = new Map();
let revisionEntrada = 0;
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
      "Prepara el modelo para verificar y habilitar la inferencia local."
    : `${n} de 3 fases válidas.`;

  dibujarLaboratorio();
  revisionEntrada++;
  window.dispatchEvent(new Event("dce-input-change"));
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

    const grises = await PreprocesadoDCE.decodificarPNG(bytes);

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
      grises,
      nombre: archivo.name,
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
    const rgba = new Uint8ClampedArray(256 * 256 * 4);
    for (let i = 0; i < estado.grises.length; i++) {
      const valor = estado.grises[i];
      rgba[4*i] = rgba[4*i+1] = rgba[4*i+2] = valor;
      rgba[4*i+3] = 255;
    }
    estado.pixeles = rgba;
  }
  return estado.pixeles;
}

// Solo se expone una copia de los píxeles de las tres fases válidas.
window.DCEEntrada = Object.freeze({
  obtener() {
    if (!FASES.every(fase => estados.get(fase)?.valida)) return null;
    return {
      revision: revisionEntrada,
      fases: FASES.map(fase => estados.get(fase).grises.slice()),
      nombres: FASES.map(fase => estados.get(fase).nombre)
    };
  }
});

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
    const control = new AbortController();
    const tiempo = setTimeout(() => control.abort(), 20000);
    let r;
    try {
      r = await fetch(ruta, {cache:"no-store", signal: control.signal});
    } catch (error) {
      throw Error(`No se pudo leer ${ruta}: ${error.name === "AbortError" ? "tiempo de espera agotado" : error.message}`);
    } finally {
      clearTimeout(tiempo);
    }
    if(!r.ok) throw Error(`No se encuentra ${ruta} (HTTP ${r.status}).`);
    return tipo==="json" ? r.json() : r.arrayBuffer();
  }
  async function hash(bytes) {
    if (!globalThis.crypto?.subtle) {
      if (typeof globalThis.sha256 !== "function") throw Error("SHA-256 no disponible.");
      return globalThis.sha256(bytes);
    }
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
      const tiempo=setTimeout(()=>{script.remove();reject(Error("La descarga del motor ONNX ha superado 45 segundos. Comprueba que la red permite acceder a cdn.jsdelivr.net."));},45000);
      script.src=CDN+"ort.min.js";
      script.crossOrigin="anonymous";
      script.onload=()=>{clearTimeout(tiempo);resolve();};
      script.onerror=()=>{clearTimeout(tiempo);script.remove();reject(Error("No se pudo descargar ONNX Runtime. Comprueba tu conexión."));};
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
    botonCombinado.disabled=ocupado||!verificado||!cortes.size;
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
    botonCombinado.disabled=ocupado||!verificado||!cortes.size;
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
        cortes.delete(id);invalidarCombinado();if(!cortes.size) pacienteActivo=null;dibujarPaciente();
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

/* Extensión insertada dentro del controlador de inferencia existente. */
const botonCombinado = $("#combined-predict");
const estadoCombinado = $("#combined-status");
const salidaCombinada = $("#combined-result");
let configCombinada = null;
let sesionCaracteristicas = null;
let resultadoCombinado = null;

function invalidarCombinado() {
  resultadoCombinado = null;
  texto(salidaCombinada, "La comparación clínica todavía no se ha calculado para estos cortes y datos.");
}

function clasificarCombinado(entrada, c) {
  if (entrada.length !== 132 || c.imputacion_mediana.length !== 132) {
    throw Error("Dimensiones del clasificador combinado incompatibles.");
  }
  const faltantes = entrada.map(v => v === null || !Number.isFinite(v));
  const x = entrada.map((v,i) => faltantes[i] ? c.imputacion_mediana[i] : v);
  for (const i of c.indicadores_faltantes) x.push(faltantes[i] ? 1 : 0);
  if (x.length !== c.coeficientes.length || x.length !== c.media_escalador.length || x.length !== c.escala_escalador.length) {
    throw Error("Escalador y coeficientes incompatibles.");
  }
  let z = c.intercepto;
  for (let i=0; i<x.length; i++) {
    z += ((x[i] - c.media_escalador[i]) / c.escala_escalador[i]) * c.coeficientes[i];
  }
  if (!Number.isFinite(z)) throw Error("Resultado combinado no finito.");
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

async function prepararCombinado() {
  if (configCombinada && sesionCaracteristicas) return;
  texto(estadoCombinado, "Cargando el clasificador clínico guardado…");
  const c = await obtener("models/clasificador_combinado.json");
  if (c.version !== "cnn_clinica_volumen_web_v1" || c.cnn_original_sha256 !== config.onnx_sha256 ||
      c.numero_caracteristicas_imagen !== 128 || c.umbral !== 0.3 ||
      !iguales(c.clases,[0,1]) || !iguales(c.variables_clinicas,["HR","HER2","age","tum_vol"]) ||
      c.agregacion_caracteristicas !== "media por paciente" ||
      c.columnas?.length !== 132 || c.columnas.some((nombre,i) => nombre !== (i<128 ? `imagen_${String(i).padStart(3,"0")}` : ["HR","HER2","age","tum_vol"][i-128]))) {
    throw Error("El combinado no corresponde a la CNN o a las variables seleccionadas.");
  }
  let maximo=0;
  for(const ref of c.referencias_clasificador) {
    const p=clasificarCombinado(ref.entrada,c);
    maximo=Math.max(maximo,Math.abs(p-ref.probabilidad));
  }
  if(maximo>1e-10) throw Error("El clasificador web no reproduce sus referencias de Python.");
  texto(estadoCombinado,"Clasificador verificado. Cargando extractor de características…");
  const bytes=await obtener("models/"+c.cnn_features_archivo,"bytes");
  if(await hash(bytes)!==c.cnn_features_sha256) throw Error("Huella del extractor de características incorrecta.");
  sesionCaracteristicas=await ort.InferenceSession.create(bytes,{executionProviders:["wasm"]});
  if(!iguales(sesionCaracteristicas.inputNames,["imagenes"]) ||
     !iguales(sesionCaracteristicas.outputNames,["probabilidad_pcr1","caracteristicas_cnn"])) {
    await sesionCaracteristicas.release(); sesionCaracteristicas=null;
    throw Error("Salidas del extractor incompatibles.");
  }
  configCombinada=c;
}

function leerClinica() {
  const hr=$("#clinical-hr").value,her2=$("#clinical-her2").value;
  const edad=$("#clinical-age").value.trim(),volumen=$("#clinical-volume").value.trim();
  const age=edad==="" ? null : Number(edad);
  const vol=volumen==="" ? null : Number(volumen);
  if(age!==null && (!Number.isFinite(age)||age<=0||age>120)) throw Error("Introduce una edad válida en años o deja el campo vacío.");
  if(vol!==null && (!Number.isFinite(vol)||vol<0)) throw Error("Introduce un volumen no negativo o deja el campo vacío.");
  return {HR:hr===""?null:Number(hr),HER2:her2===""?null:Number(her2),age,tum_vol:vol};
}

$("#clinical-fields").addEventListener("input",invalidarCombinado);
$("#clinical-fields").addEventListener("change",invalidarCombinado);

botonCombinado.addEventListener("click",async()=>{
  if(ocupado || !verificado || !cortes.size) return;
  ocupado=true;actualizarBoton();dibujarPaciente();
  invalidarCombinado();
  const campos=Array.from($("#clinical-fields").querySelectorAll("input,select"));
  campos.forEach(c=>c.disabled=true);
  try {
    const clinica=leerClinica();
    await prepararCombinado();
    const media=new Float64Array(128);
    let numero=0;
    for(const [id,corte] of cortes) {
      texto(estadoCombinado,`Extrayendo características: corte ${id} (${numero+1}/${cortes.size})…`);
      if(!corte.caracteristicas) {
        if(!corte.fases) throw Error("Faltan imágenes del corte "+id+". Vuelve a analizarlo.");
        const {datos}=PreprocesadoDCE.procesar(corte.fases,config.preprocesado);
        const input=new ort.Tensor("float32",datos,[1,3,256,256]);
        let salida;
        try {
          salida=await sesionCaracteristicas.run({imagenes:input});
          const p=Number(salida.probabilidad_pcr1.data[0]);
          if(Math.abs(p-corte.probabilidad)>1e-5+1e-4*Math.abs(corte.probabilidad)) throw Error("El extractor no reproduce la CNN del corte "+id);
          const valores=Array.from(salida.caracteristicas_cnn.data);
          if(valores.length!==128 || valores.some(v=>!Number.isFinite(v))) throw Error("Características no válidas.");
          corte.caracteristicas=valores;
        } finally {
          input.dispose?.();
          if(salida) Object.values(salida).forEach(t=>t.dispose?.());
        }
      }
      for(let i=0;i<128;i++) media[i]+=corte.caracteristicas[i];
      numero++;
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    const entrada=Array.from(media,v=>v/numero).concat([clinica.HR,clinica.HER2,clinica.age,clinica.tum_vol]);
    const p=clasificarCombinado(entrada,configCombinada);
    const pImagen=Array.from(cortes.values()).reduce((s,c)=>s+c.probabilidad,0)/numero;
    resultadoCombinado={experimento:configCombinada.experimento,paciente:pacienteActivo,numero_cortes:numero,
      datos_clinicos:clinica,puntuacion:p,umbral:configCombinada.umbral,
      clase_predicha:p>=configCombinada.umbral?1:0,cnn_solo_imagen:pImagen,
      cnn_clase_predicha:pImagen>=config.umbral?1:0,
      agregacion:"Media de 128 características por paciente antes del clasificador combinado",
      clasificador_sha256:configCombinada.clasificadores_sha256};
    const faltantes=Object.entries(clinica).filter(([,v])=>v===null).map(([k])=>k);
    texto(salidaCombinada,
      `Paciente ${pacienteActivo} · ${numero} corte(s). `+
      `Solo imágenes: ${(pImagen*100).toFixed(2)} % → pCR=${resultadoCombinado.cnn_clase_predicha}. `+
      `CNN + datos: ${(p*100).toFixed(2)} % → pCR=${resultadoCombinado.clase_predicha}. `+
      `Umbral fijo 0,3 en ambos. `+
      (faltantes.length ? "Se han imputado con las medianas de entrenamiento: "+faltantes.join(", ")+". " : "")+
      "Puntuaciones no calibradas. El resultado solo utiliza los cortes aportados."
    );
    texto(estadoCombinado,"Comparación calculada con el clasificador entrenado, sin modificar sus coeficientes.");
  } catch(error) {
    texto(estadoCombinado,"No se pudo calcular el combinado: "+error.message);
  } finally {
    campos.forEach(c=>c.disabled=false);
    ocupado=false;actualizarBoton();dibujarPaciente();
  }
});


  preparar.addEventListener("click",async()=>{
    texto(estado,"Conectando CNN…");
    ocupado=true;verificado=false;detalle.replaceChildren();actualizarBoton();
    try {
      if(location.protocol==="file:") throw Error("Abre la web mediante http://localhost:8000, no con doble clic.");
      if (!globalThis.crypto?.subtle && typeof globalThis.sha256 !== "function") throw Error("SHA-256 no disponible en este navegador.");
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
      invalidarCombinado();
      cortes.set(id,{probabilidad:p,nombres:entrada.nombres.slice(),informe,tiempo_ms:performance.now()-inicio,fases:entrada.fases.map(f=>f.slice())});
      texto(puntuacion,(100*p).toFixed(2)+" %");
      texto(
  titulo,
  `Predicción del corte: pCR=${p >= config.umbral ? 1 : 0} · corte ${id}`
);
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
    cortes.clear();invalidarCombinado();pacienteActivo=null;dibujarPaciente();
  });
  exportar.addEventListener("click",()=>{
    if(!cortes.size) return;
    const lista=Array.from(cortes,([corte,valor])=>{const {fases,...resto}=valor;return {corte,...resto};});
    const media=lista.reduce((s,c)=>s+c.probabilidad,0)/lista.length;
    const reporte={experimento:config.experimento,onnx_sha256:config.onnx_sha256,paciente:pacienteActivo,
      numero_cortes:lista.length,media,umbral:config.umbral,clase_predicha:media>=config.umbral?1:0,
      cortes:lista,alcance:"Solo los cortes aportados por el usuario; no verifica que la paciente esté completa.",comparacion_cnn_clinica:resultadoCombinado,uso:"Educativo, sin validez clínica"};
    const url=URL.createObjectURL(new Blob([JSON.stringify(reporte,null,2)],{type:"application/json"}));
    const a=document.createElement("a");a.href=url;a.download="resultado_paciente.json";a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.addEventListener("dce-input-change",alCambiarEntrada);
  alCambiarEntrada();dibujarPaciente();
})();

    window.DCEConexionIntegrada = true;
    const aviso = document.getElementById("model-status");
    aviso.textContent = "Interfaz preparada. Pulsa Conectar CNN.";
  } catch (error) {
    const aviso = document.getElementById("model-status");
    if (aviso) aviso.textContent = "No se pudo iniciar la conexión: " + error.message;
    const boton = document.getElementById("model-prepare");
    if (boton) boton.disabled = true;
  }
}, {once: true});