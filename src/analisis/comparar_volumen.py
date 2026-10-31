"""Compara añadir tum_vol usando características guardadas. CPU; no imágenes ni test."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

CLINICAS = ['HR', 'HER2', 'age']


def metricas(y, prob, umbral):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(prob) >= umbral
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    def ratio(n, d):
        return float(n / d) if d else None
    r0, r1 = ratio(vn, vn + fp), ratio(vp, vp + fn)
    return dict(pacientes=len(y), auc=float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None,
        bce_paciente=float(log_loss(y, prob, labels=[0, 1])), recall_0=r0, recall_1=r1,
        precision_0=ratio(vn, vn + fn), precision_1=ratio(vp, vp + fp),
        balanceada=(r0 + r1) / 2 if r0 is not None and r1 is not None else None,
        VN=int(vn), FP=int(fp), FN=int(fn), VP=int(vp))


def leer_caracteristicas(ruta):
    t = pd.read_csv(ruta, dtype={'patient_id': str})
    if 'patient_id' not in t or t.patient_id.duplicated().any():
        raise ValueError(f'Identificadores ausentes o duplicados en {ruta}')
    t = t.set_index('patient_id')
    if not t.pCR.isin([0, 1]).all():
        raise ValueError('Etiquetas inválidas en las características')
    return t


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True, help='Carpeta del combinado con características guardadas')
    ap.add_argument('--datos', type=Path, default=repo / 'data' / 'breastdcedl')
    ap.add_argument('--umbral', type=float, default=0.3)
    a = ap.parse_args()
    if not 0 < a.umbral < 1:
        ap.error('El umbral debe estar entre 0 y 1')
    base = a.base.resolve()
    cfg = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    fold = cfg['config_cnn']['entrenamiento']['fold_validacion']
    regularizacion = float(cfg.get('C', 0.1))
    tablas = {k: leer_caracteristicas(base / f'caracteristicas_{k}.csv') for k in ['train', 'val']}
    if set(tablas['train'].index) & set(tablas['val'].index):
        raise ValueError('Solapamiento de pacientes')
    # Solo se inspeccionan etiquetas/particiones de desarrollo, nunca se evalúa test.
    samples = pd.read_csv(a.datos / 'metadata' / 'samples.csv', dtype={'patient_id': str})
    dev = samples.loc[samples['split'].eq('train')].copy()
    if dev.groupby('patient_id')[['pCR', 'fold']].nunique().gt(1).any().any():
        raise ValueError('Metadatos inconsistentes por paciente')
    for k in ['train', 'val']:
        esperado = dev.loc[dev.fold.ne(fold) if k == 'train' else dev.fold.eq(fold)]
        if set(esperado.patient_id) != set(tablas[k].index):
            raise ValueError(f'Pacientes de {k} distintas a la partición original; revisar exclusiones')
        etiquetas = esperado.drop_duplicates('patient_id').set_index('patient_id').pCR
        if not np.array_equal(tablas[k].pCR, etiquetas.loc[tablas[k].index]):
            raise ValueError('Etiquetas discordantes con samples.csv')
    clin = pd.read_csv(a.datos / 'metadata' / 'patients.csv', dtype={'pid': str})
    ids = set(tablas['train'].index) | set(tablas['val'].index)
    clin = clin.loc[clin.pid.isin(ids)].set_index('pid')
    if clin.index.duplicated().any() or set(clin.index) != ids:
        raise ValueError('Pacientes clínicas ausentes o duplicadas')
    for v in CLINICAS + ['tum_vol']:
        clin[v] = pd.to_numeric(clin[v], errors='raise')
        if np.isinf(clin[v].to_numpy()).any():
            raise ValueError(f'Valores infinitos en {v}')
    for v in ['HR', 'HER2']:
        if not clin[v].dropna().isin([0, 1]).all():
            raise ValueError(f'Codificación inválida en {v}')
    if clin.tum_vol.dropna().lt(0).any():
        raise ValueError('Volumen negativo: revisar fuente')
    imagen = [v for v in tablas['train'].columns if v.startswith('imagen_')]
    if not imagen or imagen != [v for v in tablas['val'].columns if v.startswith('imagen_')]:
        raise ValueError('Características de imagen ausentes o distintas entre conjuntos')
    for k, tabla in tablas.items():
        if not np.isfinite(tabla[imagen].to_numpy()).all():
            raise ValueError('Características no finitas')
        if not np.array_equal(tabla.pCR, clin.loc[tabla.index, 'pCR']):
            raise ValueError('Etiquetas discordantes con patients.csv')
        tabla[CLINICAS + ['tum_vol', 'dataset']] = clin.loc[tabla.index, CLINICAS + ['tum_vol', 'dataset']]
    versiones = {
        'clinica_sin_volumen': CLINICAS,
        'clinica_con_volumen': CLINICAS + ['tum_vol'],
        'cnn_clinica_sin_volumen': imagen + CLINICAS,
        'cnn_clinica_con_volumen': imagen + CLINICAS + ['tum_vol']}
    salida = repo / 'resultados' / 'comparacion_volumen' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    salida.mkdir(parents=True)
    resultados, cohortes, modelos = [], [], {}
    print(f'CPU | Train {len(tablas["train"])} pacientes | Val {len(tablas["val"])} pacientes', flush=True)
    print(f'Umbral {a.umbral:g} | C={regularizacion} | class_weight=balanced', flush=True)
    for nombre, cols in versiones.items():
        if tablas['train'][cols].isna().all().any():
            raise ValueError('Hay una variable totalmente ausente en entrenamiento')
        modelo = Pipeline([
            ('imputar', SimpleImputer(strategy='median', add_indicator=True)),
            ('escalar', StandardScaler()),
            ('clasificador', LogisticRegression(C=regularizacion, class_weight='balanced',
                                                max_iter=3000, random_state=42))])
        modelo.fit(tablas['train'][cols], tablas['train'].pCR.astype(int))
        modelos[nombre] = dict(pipeline=modelo, columnas=cols)
        for conjunto, t in tablas.items():
            prob = modelo.predict_proba(t[cols])[:, 1]
            resultados.append(dict(modelo=nombre, conjunto=conjunto, **metricas(t.pCR, prob, a.umbral)))
            pred = pd.DataFrame(dict(patient_id=t.index, pCR=t.pCR.to_numpy(), prob=prob,
                                     dataset=t.dataset.to_numpy()))
            pred.to_csv(salida / f'{nombre}_{conjunto}_pacientes.csv', index=False)
            if conjunto == 'val':
                for cohorte, g in pred.groupby('dataset', dropna=False):
                    cohortes.append(dict(modelo=nombre, cohorte=str(cohorte),
                                         **metricas(g.pCR, g.prob, a.umbral)))
    resumen = pd.DataFrame(resultados)
    resumen.to_csv(salida / 'comparacion.csv', index=False)
    pd.DataFrame(cohortes).to_csv(salida / 'validacion_por_cohorte.csv', index=False)
    joblib.dump(modelos, salida / 'clasificadores.joblib')
    faltantes = pd.concat([t.groupby('dataset', dropna=False).tum_vol.agg(
        pacientes='size', faltantes=lambda x: int(x.isna().sum()), mediana='median').assign(conjunto=k)
        for k, t in tablas.items()])
    faltantes.to_csv(salida / 'volumen_por_cohorte.csv')
    info = dict(base=str(base), config_base=cfg, variables_por_modelo=versiones,
        umbral=a.umbral, C=regularizacion, class_weight='balanced',
        volumen='Valor original del CSV, imputación y escalado ajustados solo con train',
        cnn_congelada=True, usa_caracteristicas_guardadas=True, test_evaluado=False,
        advertencias=['Comparación exploratoria posterior al test del modelo original.',
            'Procedencia y comparabilidad de tum_vol entre cohortes pendientes de aclarar.',
            'Las características de la CNN proceden de un checkpoint elegido con validación.',
            'No interpretar mejora de validación como resultado independiente de test.'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (salida / 'config.json').write_text(json.dumps(info, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print('\nVALIDACIÓN\n' + resumen.loc[resumen.conjunto.eq('val')].to_string(index=False))
    print(f'\nResultados: {salida}')
    print('No se han procesado imágenes ni consultado predicciones de test.')


if __name__ == '__main__':
    main()
