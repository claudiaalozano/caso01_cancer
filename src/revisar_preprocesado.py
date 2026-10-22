"""Genera una revisión visual de cortes de TRAIN, sin usar las etiquetas pCR."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from preprocessing import ConfigEspacial, procesar_espacial, _bordes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path)
    p.add_argument('--raiz-datos', type=Path, default=Path(__file__).resolve().parents[1]/'data/breastdcedl')
    p.add_argument('--max-cortes', type=int, default=12)
    p.add_argument('--salida', type=Path, default=Path('resultados/revision_espacial'))
    args=p.parse_args()
    if args.max_cortes<1:p.error('--max-cortes debe ser positivo')
    cfg=ConfigEspacial.desde_json(args.config) if args.config else ConfigEspacial()
    rows=pd.read_csv(args.raiz_datos/'metadata/samples.csv')
    # Pacientes distribuidas por la lista, con un corte intermedio por paciente.
    # No se usan etiquetas pCR ni muestras de test.
    train=rows[rows['split']=='train'].sort_values(['patient_id','slice_index'])
    grupos=list(train.groupby('patient_id',sort=True))
    indices=np.linspace(0,len(grupos)-1,min(args.max_cortes,len(grupos)),dtype=int)
    seleccion=[grupos[i][1].iloc[len(grupos[i][1])//2] for i in indices]
    rows=pd.DataFrame(seleccion)
    if rows.empty:raise ValueError('No hay muestras de entrenamiento')
    args.salida.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=[]
    for num,(_,r) in enumerate(rows.iterrows()):
        phases=[]
        for key in ('path_pre','path_early','path_late'):
            with Image.open(args.raiz_datos/r[key]) as im:
                if im.format!='PNG' or im.mode!='L' or im.size!=(256,256):
                    raise ValueError(f'Formato incorrecto: {r[key]}')
                im.load();phases.append(np.array(im))
        x=np.stack(phases)
        y,info=procesar_espacial(x,cfg,diagnostico=True)
        fig,axes=plt.subplots(2,6,figsize=(18,7))
        for row,(a,label) in enumerate(((x,'Original'),(y,'Procesado'))):
            for c,name in enumerate(('PRE','EARLY','LATE')):
                axes[row,c].imshow(a[c],cmap='gray',vmin=0,vmax=255)
                axes[row,c].set_title(f'{label}: {name}')
            for col,i in ((3,1),(4,2)):
                pre=_bordes(a[0]);mov=_bordes(a[i])
                pre=pre/max(float(pre.max()),1e-12);mov=mov/max(float(mov.max()),1e-12)
                axes[row,col].imshow(np.stack((pre,mov,mov),axis=-1))
                axes[row,col].set_title(f'Bordes PRE / {("EARLY" if i==1 else "LATE")}')
            axes[row,5].imshow((a[1].astype(float)-a[0])/255,cmap='coolwarm',vmin=-.5,vmax=.5)
            axes[row,5].set_title('EARLY - PRE')
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'{r.sample_id} | Bordes: PRE rojo, otra fase cian. Revisar visualmente.')
        fig.tight_layout(h_pad=3, rect=(0,0,1,.95))
        name=f'corte_{num:03d}.png';fig.savefig(args.salida/name,dpi=120);plt.close(fig)
        report.append({'sample_id':str(r.sample_id),'figura':name,**info})
        print(f'{name}: {r.sample_id}')
        for fase,registro in info['registro'].items():
            print(f"  {fase}: desplazamiento candidato {registro['desplazamiento_yx']}; aplicado={registro['aplicado']}; {registro['motivo']}")
        print(f"  Recorte: {info['recorte']}")
    (args.salida/'diagnostico.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Revisados {len(report)} cortes de train. Resultados: {args.salida}')
    print('Los desplazamientos candidatos son heurísticos; revisar antes de activar registro.')

if __name__=='__main__':main()
