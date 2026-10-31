"""Audita PNG de desarrollo sin modificar imágenes ni consultar etiquetas pCR.

Las alertas son heurísticas. Una fase LATE más oscura no demuestra un error.
Una posible inversión debe verificarse con los datos fuente antes de corregirla.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FASES = ('PRE', 'EARLY', 'LATE')


def estadisticas(im):
    im = im.astype(float)
    # Esquinas pequeñas: no equivalen necesariamente a fondo anatómico.
    esquinas = np.concatenate([im[:8,:8].ravel(), im[:8,-8:].ravel(),
                               im[-8:,:8].ravel(), im[-8:,-8:].ravel()])
    return dict(media=float(im.mean()), p01=float(np.percentile(im,1)),
                p99=float(np.percentile(im,99)), esquinas=float(np.median(esquinas)),
                fraccion_250=float(np.mean(im>=250)), std=float(im.std()))


def analizar(fases):
    stats = [estadisticas(im) for im in fases]
    filas=[]
    for i, nombre in enumerate(FASES):
        st=stats[i]; alertas=[]
        otras=[s['esquinas'] for j,s in enumerate(stats) if j!=i]
        if st['esquinas'] > 200 and st['esquinas']-np.median(otras)>100:
            alertas.append('posible_polaridad_distinta')
        if st['fraccion_250'] > .9:
            alertas.append('valores_altos_en_mas_del_90_por_ciento')
        if st['std'] < 1:
            alertas.append('imagen_casi_constante')
        referencia=fases[0].astype(float).ravel()
        movil=fases[i].astype(float).ravel()
        corr=float(np.corrcoef(referencia,movil)[0,1]) if np.std(referencia)>0 and np.std(movil)>0 else None
        if i and corr is not None and corr < -.3:
            alertas.append('correlacion_negativa_con_PRE')
        filas.append(dict(fase=nombre, **st, correlacion_PRE=corr,
                          alerta=';'.join(alertas)))
    return filas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raiz-datos',type=Path,default=Path(__file__).resolve().parents[1]/'data/breastdcedl')
    p.add_argument('--paciente',default=None,help='Opcional: revisar solo este patient_id')
    p.add_argument('--salida',type=Path,default=Path('resultados/auditoria_intensidades'))
    p.add_argument('--max-figuras',type=int,default=24)
    a=p.parse_args()
    if a.max_figuras < 0: p.error('max-figuras debe ser >= 0')
    rows=pd.read_csv(a.raiz_datos/'metadata/samples.csv')
    # El split train incluye los folds internos de desarrollo. Test queda reservado.
    rows=rows[rows['split']=='train']
    if a.paciente is not None: rows=rows[rows.patient_id.astype(str)==a.paciente]
    if rows.empty: raise ValueError('No hay cortes de desarrollo para esta selección')
    a.salida.mkdir(parents=True,exist_ok=True)
    registros=[]; errores=[]; figuras=0; casos=0
    for n,r in enumerate(rows.itertuples(index=False),1):
        try:
            fases=[]
            for campo in ('path_pre','path_early','path_late'):
                with Image.open(a.raiz_datos/getattr(r,campo)) as im:
                    if im.mode!='L' or im.format!='PNG' or im.size!=(256,256):
                        raise ValueError(f'Formato inesperado: {getattr(r,campo)}')
                    im.load(); fases.append(np.array(im,dtype=np.uint8))
            fases=np.stack(fases)
            info=analizar(fases)
            sospechoso=any(i['alerta'] for i in info)
            casos+=int(sospechoso)
            for i,campo in zip(info,('path_pre','path_early','path_late')):
                registros.append(dict(sample_id=str(r.sample_id),patient_id=str(r.patient_id),
                                      ruta=getattr(r,campo),**i))
            if (sospechoso or a.paciente is not None) and figuras<a.max_figuras:
                fig,axes=plt.subplots(1,3,figsize=(10,4))
                for ax,im,i in zip(axes,fases,info):
                    ax.imshow(im,cmap='gray',vmin=0,vmax=255)
                    ax.set_title(f"{i['fase']} | media {i['media']:.2f}\nesquinas {i['esquinas']:.1f}")
                    ax.axis('off')
                fig.suptitle(f'{r.sample_id} · PNG original; escala fija 0–255',fontsize=10)
                fig.tight_layout();fig.savefig(a.salida/f'caso_{figuras:03d}.png',dpi=120);plt.close(fig)
                figuras+=1
        except (OSError,ValueError) as e:
            errores.append(dict(sample_id=str(r.sample_id),error=str(e)))
        if n%200==0: print(f'Auditados {n}/{len(rows)} cortes',flush=True)
    tabla=pd.DataFrame(registros)
    tabla.to_csv(a.salida/'intensidades.csv',index=False)
    if not tabla.empty:
        tabla[tabla.alerta!=''].to_csv(a.salida/'alertas.csv',index=False)
    resumen=dict(cortes_seleccionados=len(rows),cortes_con_alerta=casos,errores=errores,
                 figuras=figuras,test_consultado=False,
                 nota='Alertas heurísticas, sin inversión ni exclusión automática. Revisar datos fuente.')
    (a.salida/'resumen.json').write_text(json.dumps(resumen,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(resumen,ensure_ascii=False,indent=2))
    print(a.salida)

if __name__=='__main__':main()
