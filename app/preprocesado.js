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
