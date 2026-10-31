"""Explora metadatos con árboles pequeños; solo entrenamiento, nunca test ni validación externa."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

EXCLUIR = {'pid', 'pCR', 'split', 'test', 'fold', 'patient_id', 'sample_id'}
CLINICAS = ['age', 'menopause', 'HR', 'HER2', 'HR_HER2_STATUS',
            'TripleNeg', 'HER2pos', 'HRposHER2neg', 'tum_vol']
CATEGORICAS = {'dataset', 'HR_HER2_STATUS'}


def grupo(col):
    if col in ['race_white', 'race_black']:
        return 'demografia: solo auditoria de sesgo'
    if col == 'tum_vol':
        return 'tumoral: verificar momento y metodo de medicion'
    if col in CLINICAS:
        return 'clinica (subtipos redundantes con HR/HER2)'
    return 'procedencia/adquisicion/localizacion: auditoria'


def crear_modelo(columnas, semilla):
    cat = [c for c in columnas if c in CATEGORICAS]
    num = [c for c in columnas if c not in CATEGORICAS]
    transformadores = []
    if num:
        transformadores.append(('num', SimpleImputer(strategy='median',
            add_indicator=True, keep_empty_features=True), num))
    if cat:
        transformadores.append(('cat', Pipeline([
            ('imputar', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('codificar', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    return Pipeline([
        ('pre', ColumnTransformer(transformadores)),
        ('arbol', DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                     class_weight='balanced', random_state=semilla))])


def analizar(X, y, salida):
    salida.mkdir(parents=True)
    columnas = list(X.columns)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    importancias, resultados = [], []
    for k, (it, iv) in enumerate(cv.split(X, y), 1):
        modelo = crear_modelo(columnas, 42 + k).fit(X.iloc[it], y.iloc[it])
        prob = modelo.predict_proba(X.iloc[iv])[:, 1]
        resultados.append(dict(particion=k, auc=float(roc_auc_score(y.iloc[iv], prob)),
            balanceada=float(balanced_accuracy_score(y.iloc[iv], prob >= 0.5))))
        perm = permutation_importance(modelo, X.iloc[iv], y.iloc[iv], scoring='roc_auc',
                                      n_repeats=10, random_state=42 + k)
        importancias.append(perm.importances_mean)
        print(f'  Particion interna {k}/15: AUC {resultados[-1]["auc"]:.3f}', flush=True)
    imp = np.asarray(importancias)
    tabla = pd.DataFrame(dict(variable=columnas, grupo=[grupo(c) for c in columnas],
        descenso_auc_media=imp.mean(axis=0), descenso_auc_std=imp.std(axis=0, ddof=1),
        fraccion_particiones_descenso_positivo=(imp > 0).mean(axis=0),
        faltantes_train=X.isna().sum().to_numpy(),
        porcentaje_faltantes=100 * X.isna().mean().to_numpy())).sort_values(
            'descenso_auc_media', ascending=False)
    tabla.to_csv(salida / 'importancia_variables.csv', index=False)
    pd.DataFrame(resultados).to_csv(salida / 'cv_interna.csv', index=False)
    final = crear_modelo(columnas, 42).fit(X, y)
    nombres = final['pre'].get_feature_names_out().tolist()
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(final['arbol'], feature_names=nombres, class_names=['pCR=0', 'pCR=1'],
              filled=True, rounded=True, impurity=False, ax=ax, fontsize=8)
    ax.set_title('Solo entrenamiento. Valores de clase ponderados, no recuentos reales.')
    fig.tight_layout(); fig.savefig(salida / 'arbol.png', dpi=150); plt.close(fig)
    (salida / 'reglas.txt').write_text(export_text(final['arbol'], feature_names=nombres), encoding='utf-8')
    joblib.dump(final, salida / 'arbol.joblib')
    resumen = dict(variables=columnas, auc_media=float(np.mean([r['auc'] for r in resultados])),
        auc_std_entre_particiones=float(np.std([r['auc'] for r in resultados], ddof=1)))
    print(tabla[['variable', 'descenso_auc_media', 'fraccion_particiones_descenso_positivo']].to_string(index=False))
    print(f'AUC interna media: {resumen["auc_media"]:.3f}', flush=True)
    return resumen


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datos', type=Path, default=repo / 'data' / 'breastdcedl')
    ap.add_argument('--fold', type=int, choices=range(5), default=1)
    ap.add_argument('--salida', type=Path, default=repo / 'resultados' / 'todas_variables')
    a = ap.parse_args()
    samples = pd.read_csv(a.datos / 'metadata' / 'samples.csv', dtype={'patient_id': str})
    dev = samples.loc[samples['split'].eq('train')].copy()
    if dev.groupby('patient_id')[['fold', 'pCR']].nunique().gt(1).any().any():
        raise ValueError('Etiqueta/fold inconsistente por paciente')
    tr = dev.loc[dev.fold.ne(a.fold), ['patient_id', 'pCR']].drop_duplicates()
    if set(tr.patient_id) & set(dev.loc[dev.fold.eq(a.fold), 'patient_id']):
        raise ValueError('Solapamiento con validacion externa')
    clin = pd.read_csv(a.datos / 'metadata' / 'patients.csv', dtype={'pid': str})
    clin = clin.loc[clin.pid.isin(tr.patient_id)].copy()
    variables = [c for c in clin.columns if c not in EXCLUIR]
    # Fallar ante nuevas columnas sin revisar, para no incorporar posibles fugas.
    conocidas = set(CLINICAS) | {'dataset', 'race_white', 'race_black', 'n_xy', 'n_z',
        'n_times', 'pre', 'post_early', 'post_late', 'slice_thick', 'xy_spacing',
        'mask_start', 'mask_end', 'sraw', 'eraw', 'scol', 'ecol'}
    if set(variables) - conocidas:
        raise ValueError(f'Columnas nuevas que requieren revision: {set(variables) - conocidas}')
    tabla = tr.merge(clin, left_on='patient_id', right_on='pid', how='left',
                     validate='one_to_one', suffixes=('_samples', '_patients'))
    if tabla.pid.isna().any() or not tabla.pCR_samples.eq(tabla.pCR_patients).all():
        raise ValueError('Paciente ausente o etiqueta discordante')
    X = tabla[variables].copy()
    for c in variables:
        if c in CATEGORICAS:
            X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors='raise')
            if np.isinf(X[c].to_numpy()).any():
                raise ValueError(f'Valores infinitos en {c}')
    # Quitar constantes/vacias usando solo entrenamiento, registrandolas.
    omitidas = [c for c in variables if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=omitidas)
    y = tabla.pCR_samples.astype(int)
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError('Etiquetas binarias invalidas')
    salida = a.salida / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    salida.mkdir(parents=True)
    print(f'Solo entrenamiento: {len(tabla)} pacientes; fold externo {a.fold} reservado.', flush=True)
    print(f'Omitidas por constantes o vacias: {omitidas}', flush=True)
    resumen = {}
    for nombre, columnas in [('clinicas_tumorales', [c for c in CLINICAS if c in X]),
                              ('todas_auditoria', list(X.columns))]:
        print(f'\nANALISIS {nombre}: {columnas}', flush=True)
        resumen[nombre] = analizar(X[columnas], y, salida / nombre)
    tabla[['patient_id', 'pCR_samples']].to_csv(salida / 'pacientes_entrenamiento.csv', index=False)
    config = dict(fold_externo=a.fold, pacientes=len(tabla), excluidas=sorted(EXCLUIR),
        omitidas_constantes_o_vacias=omitidas, max_depth=3, min_samples_leaf=30,
        cv='5 particiones estratificadas por paciente, 3 repeticiones, solo entrenamiento',
        resumen=resumen, validacion_externa_evaluada=False, test_evaluado=False,
        advertencias=[
            'Analisis exploratorio: no selecciona automaticamente variables para la CNN.',
            'Subtipos e indicadores son redundantes con HR/HER2: pueden repartirse la importancia.',
            'Importancia negativa o nula no demuestra ausencia de informacion.',
            'Procedencia, adquisicion, coordenadas y faltantes pueden reflejar sesgos de cohorte.',
            'Raza: auditoria descriptiva, no causalidad ni recomendacion de incluirla.',
            'Verificar que volumen y variables tumorales esten disponibles antes del tratamiento.',
            'Las metricas de CV son descriptivas; compararlas para seleccionar requiere validacion adicional.',
            'La desviacion entre particiones no es un intervalo de confianza.' ])
    (salida / 'resumen.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nResultados: {salida}')
    print('No se ha evaluado validacion externa ni test. No se ha modificado la CNN.')


if __name__ == '__main__':
    main()
