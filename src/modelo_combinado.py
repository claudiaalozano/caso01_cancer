"""Comparación exploratoria en validación: CNN congelada + HR/HER2/edad. Nunca carga test."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, precision_score,
                             recall_score, balanced_accuracy_score, log_loss)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model3 as m
import preprocessing as pp


VARIABLES_CLINICAS = ["HR", "HER2", "age"]
VERSION = "hr_her2_edad_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def guardar(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                   allow_nan=False), encoding='utf-8')


def metricas(y, p, u):
    pred = (p >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=float(roc_auc_score(y, p)),
                bce_paciente=float(log_loss(y, p, labels=[0, 1])),
                recall_0=float(recall_score(y, pred, pos_label=0)),
                recall_1=float(recall_score(y, pred, pos_label=1)),
                precision_0=float(precision_score(y, pred, pos_label=0, zero_division=0)),
                precision_1=float(precision_score(y, pred, pos_label=1, zero_division=0)),
                balanceada=float(balanced_accuracy_score(y, pred)),
                VN=int(tn), FP=int(fp), FN=int(fn), VP=int(tp))


def extraer(modelo, filas, cfg, dispositivo, batch, workers):
    dataset = pp.DatasetCortes(filas, transform=None, cache=None,
                               indice_cache=None, config_espacial=cfg)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=dispositivo.type == 'cuda')
    rasgos, probs = [], []
    modelo.eval()
    with torch.inference_mode():
        for i, (x, _) in enumerate(loader):
            x = x.to(dispositivo)
            mapas = modelo.extractor(m.representar_entrada(x, modelo.representacion))
            z = modelo.cabeza[1](modelo.cabeza[0](mapas))
            logits = modelo.cabeza[3](modelo.cabeza[2](z))
            rasgos.append(z.cpu().numpy())
            probs.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
            if (i + 1) % 20 == 0 or i + 1 == len(loader):
                print(f'  Lotes {i + 1}/{len(loader)}', flush=True)
    z = np.concatenate(rasgos)
    p = np.concatenate(probs)
    if not np.isfinite(z).all() or not np.isfinite(p).all():
        raise ValueError('La CNN ha producido valores no finitos')
    cols = [f'imagen_{i:03}' for i in range(z.shape[1])]
    tabla = pd.DataFrame(z, columns=cols)
    tabla['patient_id'] = filas.patient_id.to_numpy()
    tabla['pCR'] = filas.pCR.astype(int).to_numpy()
    tabla['prob_cnn'] = p
    # La CNN original promedia probabilidades. La fusión promedia características.
    return tabla.groupby('patient_id', sort=True).mean(), cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experimento', default='cnn5_5pool_otsu_spatial_f1')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--umbral', type=float, default=0.3)
    ap.add_argument('--C', type=float, default=0.1,
                    help='Regularización logística: menor C implica más regularización')
    a = ap.parse_args()
    print(f'Script: {Path(__file__).resolve()}', flush=True)
    print(f'Versión: {VERSION} | Variables clínicas: {VARIABLES_CLINICAS}', flush=True)
    if a.batch < 1 or a.workers < 0 or not 0 < a.umbral < 1 or a.C <= 0:
        ap.error('Argumentos fuera de rango')
    carpeta = pp.REPO / 'resultados' / 'entrenamiento' / a.experimento
    config_path = carpeta / 'config.json'
    c = json.loads(config_path.read_text(encoding='utf-8'))
    arq, ent = c['arquitectura'], c['entrenamiento']
    if ent.get('modo_rapido') or len(arq['canales']) != 5:
        raise ValueError('Se requiere el modelo original de cinco etapas, no rápido')
    pesos = pp.REPO / Path(c['pesos']['ruta'].replace('\\', '/'))
    if sha(pesos) != c['pesos']['sha256']:
        raise ValueError('Los pesos no coinciden con el SHA registrado')
    cfg = pp.ConfigEspacial(**ent['preprocesado_espacial'])
    torch.manual_seed(42)
    np.random.seed(42)
    dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo = m.CNNpCR(canales=tuple(arq['canales']), dropout=arq['dropout'],
                     representacion=arq['representacion'],
                     atencion=arq.get('atencion', 'ninguna'))
    if any(isinstance(v, torch.nn.LeakyReLU) for v in modelo.modules()):
        raise ValueError('Este experimento requiere ReLU, no LeakyReLU')
    if sum(p.numel() for p in modelo.parameters()) != arq['parametros']:
        raise ValueError('La arquitectura no coincide con config.json')
    modelo.load_state_dict(torch.load(pesos, map_location='cpu', weights_only=True), strict=True)
    modelo.requires_grad_(False)
    modelo.to(dispositivo).eval()
    pp._requiere_utils()
    # Se conserva la partición original: no aplica exclusiones nuevas posteriores.
    samples = pp.uc.cargar_samples(pp.RAIZ_DATOS)
    desarrollo = samples.loc[samples['split'].eq('train')].copy()
    tr, va = pp.uc.particion(desarrollo, fold_val=ent['fold_validacion'])
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
    assert set(tr.patient_id).isdisjoint(va.patient_id)
    pacientes_path = pp.RAIZ_DATOS / 'metadata' / 'patients.csv'
    pacientes = pd.read_csv(pacientes_path, dtype={'pid': str})
    if pacientes.pid.duplicated().any():
        raise ValueError('patients.csv contiene pacientes duplicadas')
    pacientes = pacientes.loc[pacientes['pid'].isin(set(tr.patient_id) | set(va.patient_id))].copy()
    clin = pacientes.set_index('pid')[VARIABLES_CLINICAS + ['pCR']].copy()
    clin['age'] = pd.to_numeric(clin['age'], errors='raise')
    if not np.isfinite(clin['age'].dropna()).all():
        raise ValueError('La edad contiene valores no finitos')
    for col in ['HR', 'HER2']:
        clin[col] = pd.to_numeric(clin[col], errors='raise')
        if not clin[col].dropna().isin([0, 1]).all():
            raise ValueError(f'{col} debe ser 0, 1 o ausente; revisar codificación')
    tablas = {}
    for nombre, filas in [('train', tr), ('val', va)]:
        print(f'{nombre}: {filas.patient_id.nunique()} pacientes; dispositivo {dispositivo}', flush=True)
        tabla, cols_imagen = extraer(modelo, filas, cfg, dispositivo, a.batch, a.workers)
        if not tabla.index.isin(clin.index).all():
            raise ValueError('Faltan pacientes en patients.csv')
        if not np.array_equal(tabla.pCR.to_numpy(), clin.loc[tabla.index, 'pCR'].to_numpy()):
            raise ValueError('Etiquetas discordantes entre samples y patients')
        tabla[VARIABLES_CLINICAS] = clin.loc[tabla.index, VARIABLES_CLINICAS]
        tablas[nombre] = tabla
    # Verifica la CNN contra las probabilidades originales antes de ajustar la fusión.
    original = pd.read_csv(carpeta / 'val_pacientes.csv', dtype={'patient_id': str}).set_index('patient_id')
    actual = tablas['val']
    if set(original.index) != set(actual.index):
        raise ValueError('No coinciden las pacientes de validación originales')
    if not np.allclose(original.loc[actual.index, 'prob'], actual.prob_cnn, atol=1e-3, rtol=1e-3):
        raise ValueError('La CNN no reproduce las probabilidades originales; revisar antes de continuar')
    print('CNN original verificada. Ajustando clasificadores por paciente...', flush=True)
    salida = pp.REPO / 'resultados' / 'combinados' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    salida.mkdir(parents=True)
    resultados, modelos = [], {}
    for nombre, columnas in [('solo_cnn', None), ('solo_clinica', VARIABLES_CLINICAS),
                                ('cnn_clinica', cols_imagen + VARIABLES_CLINICAS)]:
        if columnas is None:
            predicciones = {k: t.prob_cnn.to_numpy() for k, t in tablas.items()}
        else:
            if tablas['train'][columnas].isna().all().any():
                raise ValueError('Hay una variable completamente ausente en entrenamiento')
            pipe = Pipeline([
                ('imputar', SimpleImputer(strategy='median', add_indicator=True)),
                ('escalar', StandardScaler()),
                ('clasificador', LogisticRegression(C=a.C, class_weight='balanced',
                                                     max_iter=3000, random_state=42))])
            pipe.fit(tablas['train'][columnas], tablas['train'].pCR.astype(int))
            predicciones = {k: pipe.predict_proba(t[columnas])[:, 1] for k, t in tablas.items()}
            modelos[nombre] = dict(pipeline=pipe, columnas=columnas)
        for conjunto, tabla in tablas.items():
            y = tabla.pCR.astype(int).to_numpy()
            prob = predicciones[conjunto]
            met = metricas(y, prob, a.umbral)
            resultados.append(dict(modelo=nombre, conjunto=conjunto, umbral=a.umbral, **met))
            pd.DataFrame(dict(patient_id=tabla.index, pCR=y, prob=prob)).to_csv(
                salida / f'{nombre}_{conjunto}_pacientes.csv', index=False)
            if conjunto == 'val':
                fig, ax = plt.subplots(figsize=(5, 4))
                matriz = np.array([[met['VN'], met['FP']], [met['FN'], met['VP']]])
                ax.imshow(matriz, cmap='Blues')
                for (i, j), v in np.ndenumerate(matriz):
                    ax.text(j, i, str(v), ha='center', va='center', fontsize=16,
                            color='white' if v > matriz.max() / 2 else 'black')
                ax.set(xticks=[0, 1], yticks=[0, 1], xlabel='Predicción pCR',
                       ylabel='Real pCR', title=f'{nombre} · val · umbral {a.umbral:g}')
                fig.tight_layout(); fig.savefig(salida / f'{nombre}_val_matriz.png'); plt.close(fig)
    resumen = pd.DataFrame(resultados)
    resumen.to_csv(salida / 'comparacion.csv', index=False)
    for conjunto, tabla in tablas.items():
        tabla.to_csv(salida / f'caracteristicas_{conjunto}.csv')
    joblib.dump(modelos, salida / 'clasificadores.joblib')
    guardar(salida / 'config.json', dict(experimento_cnn=a.experimento,
        pesos_sha256=sha(pesos), config_cnn=c, config_cnn_sha256=sha(config_path),
        patients_sha256=sha(pacientes_path), variables_clinicas=VARIABLES_CLINICAS, version_script=VERSION,
        script_sha256=sha(Path(__file__)),
        columnas_por_modelo={k: v['columnas'] for k, v in modelos.items()},
        umbral=a.umbral, C=a.C, class_weight='balanced',
        agregacion_caracteristicas='media por paciente', cnn_congelada=True,
        exclusiones='Mismas pacientes que en la validación original',
        test_consultado_por_este_script=False,
        limitacion='Experimento exploratorio posterior al test del modelo de imagen; no evaluación final independiente',
        faltantes_train=tablas['train'][VARIABLES_CLINICAS].isna().sum().astype(int).to_dict(),
        faltantes_val=tablas['val'][VARIABLES_CLINICAS].isna().sum().astype(int).to_dict()))
    print(resumen.loc[resumen.conjunto.eq('val')].to_string(index=False))
    print(f'Resultados: {salida}')
    print('No se ha realizado inferencia en test. Las métricas de train son de ajuste, no de generalización.')


if __name__ == '__main__':
    main()
