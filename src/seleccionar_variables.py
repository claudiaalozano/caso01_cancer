"""Árbol clínico exploratorio. Solo entrenamiento; no evalúa validación externa ni test."""
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

VARIABLES = ['age', 'HR', 'HER2', 'menopause']


def crear_modelo(semilla):
    pre = ColumnTransformer([
        ('edad', SimpleImputer(strategy='median', add_indicator=True), ['age']),
        ('marcadores', SimpleImputer(strategy='most_frequent', add_indicator=True),
         ['HR', 'HER2', 'menopause'])
    ])
    return Pipeline([
        ('preprocesado', pre),
        ('arbol', DecisionTreeClassifier(max_depth=3, min_samples_leaf=30,
                    class_weight='balanced', random_state=semilla))
    ])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    p.add_argument('--datos', type=Path, default=repo / 'data' / 'breastdcedl')
    p.add_argument('--fold', type=int, choices=range(5), default=1)
    p.add_argument('--salida', type=Path, default=repo / 'resultados' / 'seleccion_variables')
    a = p.parse_args()
    # Solo se usan etiquetas y variables de las pacientes de entrenamiento.
    samples = pd.read_csv(a.datos / 'metadata' / 'samples.csv', dtype={'patient_id': str})
    dev = samples.loc[samples['split'].eq('train')].copy()
    if dev.groupby('patient_id')[['fold', 'pCR']].nunique().gt(1).any().any():
        raise ValueError('Fold o etiqueta inconsistentes dentro de una paciente')
    tr = dev.loc[dev.fold.ne(a.fold), ['patient_id', 'pCR']].drop_duplicates()
    reservadas = set(dev.loc[dev.fold.eq(a.fold), 'patient_id'])
    if set(tr.patient_id) & reservadas:
        raise ValueError('Solapamiento con validación externa')
    clin = pd.read_csv(a.datos / 'metadata' / 'patients.csv', dtype={'pid': str})
    clin = clin.loc[clin.pid.isin(tr.patient_id), ['pid', 'pCR'] + VARIABLES].copy()
    tabla = tr.merge(clin, left_on='patient_id', right_on='pid', how='left',
                     validate='one_to_one', suffixes=('_samples', '_patients'))
    if tabla.pid.isna().any() or not tabla.pCR_samples.eq(tabla.pCR_patients).all():
        raise ValueError('Faltan pacientes o hay etiquetas discordantes')
    X = tabla[VARIABLES].apply(pd.to_numeric, errors='raise')
    y = tabla.pCR_samples.astype(int)
    if not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError('Se requieren las dos clases pCR')
    if np.isinf(X.to_numpy()).any() or X.isna().all().any():
        raise ValueError('Variable infinita o totalmente ausente')
    for col in ['HR', 'HER2', 'menopause']:
        if not X[col].dropna().isin([0, 1]).all():
            raise ValueError(f'Revisar codificación binaria de {col}')
    print(f'Entrenamiento: {len(tabla)} pacientes. Fold externo {a.fold} reservado.', flush=True)
    print('Faltantes en entrenamiento:\n' + X.isna().sum().to_string())
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    resultados, importancias, usos = [], [], []
    for k, (it, iv) in enumerate(cv.split(X, y), start=1):
        modelo = crear_modelo(42 + k)
        modelo.fit(X.iloc[it], y.iloc[it])
        prob = modelo.predict_proba(X.iloc[iv])[:, 1]
        resultados.append(dict(particion=k, auc=roc_auc_score(y.iloc[iv], prob),
            balanceada=balanced_accuracy_score(y.iloc[iv], prob >= 0.5)))
        perm = permutation_importance(modelo, X.iloc[iv], y.iloc[iv], scoring='roc_auc',
                                      n_repeats=10, random_state=42 + k)
        importancias.append(perm.importances_mean)
        nombres = modelo['preprocesado'].get_feature_names_out()
        usadas = {nombres[i].split('__')[-1] for i in modelo['arbol'].tree_.feature if i >= 0}
        usos.append([int(v in usadas) for v in VARIABLES])
        print(f'Partición interna {k}/15: AUC {resultados[-1]["auc"]:.3f}', flush=True)
    imp = np.asarray(importancias)
    resumen = pd.DataFrame(dict(variable=VARIABLES,
        descenso_auc_permutar_media=imp.mean(axis=0),
        descenso_auc_permutar_std=imp.std(axis=0, ddof=1),
        fraccion_particiones_descenso_positivo=(imp > 0).mean(axis=0),
        fraccion_arboles_usa_variable=np.asarray(usos).mean(axis=0),
        faltantes_train=X.isna().sum().to_numpy())).sort_values(
            'descenso_auc_permutar_media', ascending=False)
    salida = a.salida / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    salida.mkdir(parents=True)
    resumen.to_csv(salida / 'importancia_variables.csv', index=False)
    pd.DataFrame(resultados).to_csv(salida / 'validacion_cruzada_interna.csv', index=False)
    tabla[['patient_id', 'pCR_samples']].to_csv(salida / 'pacientes_entrenamiento.csv', index=False)
    final = crear_modelo(42).fit(X, y)
    nombres = final['preprocesado'].get_feature_names_out().tolist()
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(final['arbol'], feature_names=nombres, class_names=['pCR=0', 'pCR=1'],
              filled=True, rounded=True, impurity=False, proportion=False, ax=ax, fontsize=9)
    ax.set_title('Árbol clínico: solo entrenamiento. Valores de clase ponderados por balanced.')
    fig.tight_layout(); fig.savefig(salida / 'arbol_clinico.png', dpi=160); plt.close(fig)
    (salida / 'reglas_arbol.txt').write_text(export_text(final['arbol'], feature_names=nombres), encoding='utf-8')
    joblib.dump(final, salida / 'arbol_clinico.joblib')
    info = dict(variables=VARIABLES, fold_externo_reservado=a.fold, pacientes=len(tabla),
        max_depth=3, min_samples_leaf=30, class_weight='balanced',
        cv='5 particiones estratificadas por paciente, 3 repeticiones dentro de entrenamiento',
        auc_cv_media=float(np.mean([r['auc'] for r in resultados])),
        auc_cv_std=float(np.std([r['auc'] for r in resultados], ddof=1)),
        validacion_externa_evaluada=False, test_evaluado=False,
        nota='Importancias exploratorias, no selección automática. La desviación entre particiones no es un intervalo de confianza. Las variables correlacionadas pueden compartir importancia; no demuestra aporte adicional a la CNN. Los nodos contienen recuentos ponderados, no prevalencias clínicas.')
    (salida / 'resumen.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n' + resumen.to_string(index=False))
    print(f'AUC interna media: {info["auc_cv_media"]:.3f}; desviación entre particiones: {info["auc_cv_std"]:.3f}')
    print(f'Resultados: {salida}')
    print('No se han evaluado validación externa ni test. No se ha modificado la CNN.')


if __name__ == '__main__':
    main()
