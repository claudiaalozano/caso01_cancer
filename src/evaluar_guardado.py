"""Evalúa pesos guardados sin volver a entrenar. Test solo con --conjunto test."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import model2 as m
import preprocessing as pp
from distribucion_probabilidades import dibujar_distribucion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experimento',required=True)
    p.add_argument('--conjunto',choices=['train','val','test'],default='val')
    p.add_argument('--umbral',type=float,default=.3)
    p.add_argument('--batch',type=int,default=32)
    p.add_argument('--workers',type=int,default=0)
    a=p.parse_args()
    if not 0<=a.umbral<=1 or a.batch<1 or a.workers<0:p.error('Argumentos fuera de rango')
    carpeta=pp.REPO/'resultados/entrenamiento'/a.experimento
    c=json.loads((carpeta/'config.json').read_text(encoding='utf-8'))
    arq=c['arquitectura']; entrenamiento=c['entrenamiento']
    if len(arq['canales'])!=4:raise ValueError('Este script solo carga la CNN actual de cuatro etapas')
    if entrenamiento.get('modo_rapido'):raise ValueError('No se evalúan pesos de modo rápido')
    if 'preprocesado_espacial' not in entrenamiento:
        raise ValueError('Falta el preprocesado exacto del experimento: no se adivina')
    cfg=pp.ConfigEspacial(**entrenamiento['preprocesado_espacial'])
    pesos=pp.REPO/c['pesos']['ruta']
    if not pesos.exists():raise FileNotFoundError(f'Faltan los pesos: {pesos}')
    if m.sha256(pesos)!=c['pesos']['sha256']:raise ValueError('El SHA-256 de los pesos no coincide')
    salida=carpeta/f'evaluacion_{a.conjunto}_u{a.umbral:g}'
    if salida.exists():raise FileExistsError(f'Ya existe {salida}; reutiliza el CSV para dibujar sin repetir inferencia')
    marca=carpeta/'evaluacion_test_iniciada.json'
    if a.conjunto=='test':
        if c.get('test') is not None or (carpeta/'test_pacientes.csv').exists() or marca.exists():
            raise RuntimeError('Test ya consultado o iniciado. Usa el CSV guardado; no repitas inferencia para ajustar decisiones.')
        # Fija y registra el umbral antes de abrir el conjunto test.
        marca.write_text(json.dumps(dict(umbral=a.umbral,pesos_sha256=c['pesos']['sha256'])),encoding='utf-8')
    dispositivo=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo=m.CNNpCR(canales=tuple(arq['canales']),dropout=arq['dropout'],
                    representacion=arq['representacion'],atencion=arq.get('atencion','ninguna')).to(dispositivo)
    salida.mkdir(parents=True)
    if a.conjunto=='test':
        resultado=m.evaluar_test(modelo,pesos,a.umbral,
                                SimpleNamespace(batch=a.batch,workers=a.workers),
                                dispositivo,salida,cfg)
    else:
        modelo.load_state_dict(torch.load(pesos,map_location=dispositivo,weights_only=True))
        d=pp.crear_dataloaders(fold_val=entrenamiento['fold_validacion'],batch_size=a.batch,
                              aumentar=False,num_workers=a.workers,config_espacial=cfg)
        loader=d['train_eval'] if a.conjunto=='train' else d['val']
        filas=d['filas_train'] if a.conjunto=='train' else d['filas_val']
        perdida,prob=m.predecir(modelo,loader,nn.BCEWithLogitsLoss(),dispositivo)
        met,tabla=m.metricas_paciente(prob,filas,a.umbral)
        tabla.to_csv(salida/f'{a.conjunto}_pacientes.csv')
        m.dibujar_matriz(met,salida/'matriz_confusion.png',f'{a.conjunto} · umbral {a.umbral:g}')
        dibujar_distribucion(tabla,salida/'distribucion_probabilidades.png',a.umbral,a.conjunto)
        resultado=dict(conjunto=a.conjunto,perdida_BCE_sin_ponderar=perdida,
                       umbral=a.umbral,metricas=met,modo='eval sin aumentos')
    (salida/'metricas.json').write_text(json.dumps(m.limpiar_json(resultado),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(m.limpiar_json(resultado),indent=2,ensure_ascii=False))
    print(salida)

if __name__=='__main__':main()
