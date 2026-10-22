#!/usr/bin/env python3
"""Actualiza solo el recorte en src/preprocessing.py. Crea copia de seguridad."""
import ast
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

NUEVA_FUNCION = 'def procesar_espacial(\n    fases: np.ndarray,\n    config: ConfigEspacial | None = None,\n    diagnostico: bool = False,\n) -> tuple[np.ndarray, dict]:\n    """Registro opcional y recorte rectangular común, sin deformar proporciones.\n\n    El relleno negro conserva 256x256; puede seguir existiendo fondo. Solo se\n    eliminan márgenes exteriores de valor exactamente cero en TODAS las fases.\n    """\n    fases = _validar_fases(fases)\n    config = config or ConfigEspacial()\n    informe = {\n        "version": "recorte_rectangular_v1",\n        "config": config.to_dict(),\n        "registro": {},\n        "recorte": None,\n        "advertencia": "Diagnóstico heurístico; no certifica alineación anatómica.",\n    }\n    if not (config.registrar or config.recortar or diagnostico):\n        return fases, informe\n    salida = fases.copy()\n    if config.registrar or diagnostico:\n        for indice, nombre in ((1, "EARLY"), (2, "LATE")):\n            resultado = _estimar_desplazamiento(fases[0], fases[indice], config)\n            informe["registro"][nombre] = resultado\n            if resultado["aplicado"]:\n                salida[indice] = _mover(fases[indice], *resultado["desplazamiento_yx"])\n    if config.recortar:\n        alto, ancho = salida.shape[1:]\n        filas, columnas = np.where(np.any(salida > 0, axis=0))\n        if len(filas):\n            y0 = max(0, int(filas.min()) - config.margen)\n            y1 = min(alto, int(filas.max()) + 1 + config.margen)\n            x0 = max(0, int(columnas.min()) - config.margen)\n            x1 = min(ancho, int(columnas.max()) + 1 + config.margen)\n            if (y0, y1, x0, x1) != (0, alto, 0, ancho):\n                from PIL import Image\n                recorte = salida[:, y0:y1, x0:x1]\n                h, w = recorte.shape[1:]\n                escala = min(alto / h, ancho / w)\n                nuevo_h = min(alto, max(1, round(h * escala)))\n                nuevo_w = min(ancho, max(1, round(w * escala)))\n                top = (alto - nuevo_h) // 2\n                left = (ancho - nuevo_w) // 2\n                antes = float(np.mean(salida == 0))\n                lienzo = np.zeros_like(salida)\n                for indice, imagen in enumerate(recorte):\n                    if (nuevo_h, nuevo_w) == (h, w):\n                        redimensionada = imagen\n                    else:\n                        redimensionada = np.asarray(\n                            Image.fromarray(imagen).resize(\n                                (nuevo_w, nuevo_h), Image.Resampling.BILINEAR\n                            )\n                        )\n                    lienzo[indice, top:top+nuevo_h, left:left+nuevo_w] = redimensionada\n                salida = lienzo\n                informe["recorte"] = {\n                    "bbox_xyxy": [x0, y0, x1, y1],\n                    "tamano_recorte_hw": [h, w],\n                    "tamano_redimensionado_hw": [nuevo_h, nuevo_w],\n                    "padding_tblr": [top, alto-nuevo_h-top, left, ancho-nuevo_w-left],\n                    "escala": escala,\n                    "fraccion_negra_antes": antes,\n                    "fraccion_negra_despues": float(np.mean(salida == 0)),\n                    "nota": ("Se amplía el campo conservado" if escala > 1\n                             else "Sin ampliación: puede limitarse a recentrar el tejido"),\n                }\n    return salida, informe\n'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archivo', type=Path, default=Path('src/preprocessing.py'))
    args = parser.parse_args()
    ruta = args.archivo.resolve()
    texto = ruta.read_text(encoding='utf-8')
    arbol = ast.parse(texto)
    nombres = {n.name for n in arbol.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    necesarios = {'ConfigEspacial', '_validar_fases', '_estimar_desplazamiento', '_mover', 'procesar_espacial'}
    if not necesarios <= nombres:
        parser.error('Este actualizador requiere la versión de preprocessing.py autocontenida que pegamos en el chat. No se ha modificado nada.')
    funciones = [n for n in arbol.body if isinstance(n, ast.FunctionDef) and n.name == 'procesar_espacial']
    if len(funciones) != 1:
        parser.error('Hay varias definiciones de procesar_espacial. Corrige duplicados antes de actualizar; no se ha modificado nada.')
    funcion = funciones[0]
    lineas = texto.splitlines(keepends=True)
    nuevo = ''.join(lineas[:funcion.lineno-1]) + NUEVA_FUNCION + '\n' + ''.join(lineas[funcion.end_lineno:])
    if ast.dump(funcion) == ast.dump(ast.parse(NUEVA_FUNCION).body[0]):
        nuevo = texto
    ast.parse(nuevo)
    if texto != nuevo:
        sello = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = ruta.with_name(ruta.name + '.' + sello + '.bak')
        shutil.copy2(ruta, backup)
        temporal = ruta.with_name(ruta.name + '.rectangular.tmp')
        temporal.write_text(nuevo, encoding='utf-8')
        temporal.replace(ruta)
        print('Copia anterior:', backup)
    print('Actualizado:', ruta)
    raiz = ruta.parent.parent
    carpeta = raiz / 'configs'
    carpeta.mkdir(exist_ok=True)
    config = carpeta / 'solo_recorte_rectangular.json'
    if not config.exists():
        config.write_text(json.dumps({'registrar': False, 'recortar': True}, indent=2) + '\n', encoding='utf-8')
    print('Configuración para revisión:', config)
    print('Registro desactivado en la nueva configuración. Revisa los PNG antes de entrenar.')
    print('Guarda la versión recorte_rectangular_v1 con los experimentos nuevos; los pesos antiguos requieren su preprocesado original.')

if __name__ == '__main__':
    main()
