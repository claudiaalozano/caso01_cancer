"""Distribución empírica por paciente. No requiere inferencia ni entrenamiento."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dibujar_distribucion(tabla, ruta, umbral=0.3, titulo='Distribución por paciente'):
    t = tabla.copy()
    if 'patient_id' in t.columns and t.patient_id.duplicated().any():
        raise ValueError('Se necesita una sola fila por paciente')
    if t.empty or not {'prob', 'pCR'}.issubset(t.columns):
        raise ValueError('CSV vacío o sin columnas prob y pCR')
    if not np.isfinite(t.prob).all() or not t.prob.between(0, 1).all():
        raise ValueError('Probabilidades fuera de [0,1] o no finitas')
    if not t.pCR.isin([0, 1]).all() or not 0 <= umbral <= 1:
        raise ValueError('Etiquetas o umbral inválidos')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 21)
    for clase, color in ((0, '#2a78d6'), (1, '#eb6834')):
        p = t.loc[t.pCR == clase, 'prob'].to_numpy()
        if not len(p):
            continue
        etiqueta = f'pCR={clase} (n={len(p)})'
        # Cada clase suma 100%; permite comparar clases desbalanceadas.
        axes[0].hist(p, bins=bins, weights=np.full(len(p), 100 / len(p)),
                     histtype='step', linewidth=2, color=color, label=etiqueta)
        orden = np.sort(p)
        axes[1].step(np.r_[0, orden, 1], np.r_[0, np.arange(1, len(p)+1)/len(p), 1],
                     where='post', color=color, label=etiqueta)
    axes[0].set_ylabel('% de pacientes de cada clase por intervalo')
    axes[0].set_title('Histograma de probabilidades')
    axes[1].set_ylabel('Fracción acumulada de cada clase')
    axes[1].set_title('Distribución acumulada empírica')
    for ax in axes:
        ax.axvline(umbral, ls='--', color='black', label=f'Umbral {umbral:g}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probabilidad predicha de pCR=1')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--umbral', type=float, default=.3)
    p.add_argument('--salida', type=Path, required=True)
    a = p.parse_args()
    t = pd.read_csv(a.csv)
    dibujar_distribucion(t, a.salida, a.umbral, a.csv.stem)
    y = t.pCR.astype(int)
    for u in dict.fromkeys([.5, a.umbral]):
        pred = t.prob >= u
        vp=int((pred & (y==1)).sum()); fp=int((pred & (y==0)).sum())
        vn=int((~pred & (y==0)).sum()); fn=int((~pred & (y==1)).sum())
        div=lambda n,d:n/d if d else float('nan')
        print(f'Umbral {u:g}: recall={div(vp,vp+fn):.3f}, precisión={div(vp,vp+fp):.3f}, '
              f'especificidad={div(vn,vn+fp):.3f}; VP={vp} VN={vn} FP={fp} FN={fn}')
    print(a.salida)

if __name__ == '__main__':
    main()
