# caso01_cancer

Predicción de respuesta al tratamiento en cáncer de mama
Proyecto académico de Aprendizaje Automático · Claudia Lozano Pérez

Estudio de una red neuronal convolucional (CNN) para predecir la respuesta patológica completa (pCR) al tratamiento a partir de imágenes de resonancia magnética de mama. El proyecto incluye el entrenamiento y la evaluación por paciente, una comparación exploratoria con datos clínicos y una aplicación web para visualizar las imágenes y ejecutar el modelo en el navegador.
Resultado principal: la CNN obtuvo un AUC de 0,518 en el test independiente. Su capacidad para distinguir entre ambas clases fue limitada en este conjunto. El proyecto tiene finalidad educativa y de investigación; sus predicciones no deben utilizarse para tomar decisiones clínicas.

Enlaces
- Aplicación web: abrir en GitHub Pages.
- Código: repositorio del proyecto.
- Modelo ONNX utilizado para desplegar la web: release modelo-v1.
- Repositorio del modelo en Hugging Face: claudiaalozano/caso01_cancer.
¿Qué intentamos predecir?
El objetivo es anticipar la respuesta al tratamiento, no detectar si una imagen contiene cáncer.
Etiqueta	Significado en el proyecto
pCR = 1	Respuesta patológica completa tras el tratamiento.
pCR = 0	No se alcanza respuesta patológica completa.


El modelo asigna una puntuación a cada corte. Para obtener una predicción por paciente se calcula la media de las puntuaciones de sus cortes y se aplica el umbral fijado en validación:
- Puntuación mayor o igual que 0,30 → pCR = 1.
- Puntuación menor que 0,30 → pCR = 0.
Por ejemplo, una puntuación del 3,89 % queda por debajo del umbral del 30 % y corresponde a una predicción de pCR = 0. La puntuación no mide cuánto cáncer hay ni constituye una probabilidad clínica calibrada.
Datos y separación por pacientes
Se utilizaron imágenes de BreastDCEDL, procedentes de las cohortes I-SPY1, I-SPY2 y Duke. Cada ejemplo reúne tres fases del mismo corte: PRE, EARLY y LATE.
Conjunto del modelo original	Pacientes	Cortes tras el preprocesado
Entrenamiento	878	8.755
Validación	219	2.190
Test independiente	176	1.758


La separación se realiza por paciente: los cortes de una misma paciente no se reparten entre entrenamiento, validación y test.

El conjunto de desarrollo —entrenamiento y validación— contiene 322 pacientes con pCR = 1 y 775 con pCR = 0. En el test hay 53 con pCR = 1 y 123 con pCR = 0. En ambos casos, aproximadamente 3 de cada 10 pacientes alcanzan pCR = 1.
Modelo y preparación de las imágenes
El modelo principal seleccionado es cnn5_5pool_otsu_spatial_f1. Es una CNN 2D propia, entrenada desde cero, con cinco bloques convolucionales, activaciones ReLU, atención espacial y una salida binaria.

Sus tres canales de entrada son PRE, EARLY y LATE. No son canales de color RGB y, en este modelo final, no se construyen explícitamente como diferencias entre fases.

La preparación incluye la estimación de una región de interés mediante Otsu, un recorte común para las tres fases, ajuste a 256 × 256 píxeles conservando la proporción con relleno y normalización con una escala común. La aplicación debe reproducir el preprocesado utilizado en Python.

Durante el entrenamiento se utilizaron aumentos conjuntos de las tres fases: volteos horizontales, rotaciones y desplazamientos. Estos aumentos no se aplican en validación, test ni en la aplicación web.

El entrenamiento utiliza AdamW y una pérdida binaria ponderada para aumentar la contribución de los ejemplos positivos. El checkpoint original seleccionado corresponde a la época 12, elegida por el AUC de validación por paciente.
Resultados de la CNN
Validación y test
Evaluación	AUC ROC
Entrenamiento del checkpoint seleccionado	0,769
Validación del checkpoint seleccionado	0,556
Test independiente	0,518


El AUC resume la capacidad para ordenar a las pacientes según su respuesta: 0,5 es la referencia de una ordenación al azar y 1 representa una separación perfecta. La diferencia entre entrenamiento y validación indica que el rendimiento observado al entrenar no se mantuvo en pacientes distintas.

El intervalo de confianza bootstrap del 95 % del AUC de test fue [0,430; 0,607], con 5.000 remuestreos por paciente. El intervalo incluye 0,5, por lo que estos resultados no permiten afirmar que el modelo discrimine por encima del azar en la población de origen.

También se realizaron experimentos de validación cruzada con cinco particiones: el AUC medio de validación de los checkpoints fue 0,566 ± 0,014. Esta cifra describe la variabilidad entre esos experimentos; no sustituye al resultado del test ni corresponde al mismo checkpoint original.
Matriz de confusión del test
Resultados por paciente con el umbral fijo de 0,30:
Resultado real	Predice pCR = 0	Predice pCR = 1
pCR = 0	61 verdaderos negativos	62 falsos positivos
pCR = 1	29 falsos negativos	24 verdaderos positivos


Métrica	Resultado
Accuracy	48,3 %
Exactitud balanceada	47,4 %
Precisión de pCR = 1	27,9 %
Recall de pCR = 1	45,3 %
F1-score de pCR = 1	0,345
F1-score de pCR = 0	0,573
F1-score macro	0,459


La CNN identificó 24 de las 53 pacientes que alcanzaron pCR = 1. De las 86 pacientes que clasificó como pCR = 1, 24 pertenecían realmente a esa clase.

Predecir siempre pCR = 0 habría acertado 123 de 176 pacientes (69,9 % de accuracy), pero no habría detectado ninguna paciente con pCR = 1. Esto ilustra por qué conviene interpretar varias métricas cuando las clases están desbalanceadas. Con el umbral fijado, la CNN tampoco superó el 50 % de exactitud balanceada de esa referencia.
Comparación exploratoria con datos clínicos
Se estudiaron HR, HER2, edad y volumen tumoral, tanto con un clasificador exclusivamente clínico como combinándolos con la media de 128 características de una CNN congelada.
Variables	AUC solo clínica	AUC CNN + clínica
HR + HER2	0,646	0,598
HR + HER2 + edad	0,697	0,612
HR + HER2 + edad + volumen	0,715	0,613


Estos resultados corresponden a las mismas 219 pacientes de validación, no al test independiente. En esta comparación, añadir las características de la CNN no mejoró el resultado del modelo clínico. La combinación con volumen obtuvo un AUC de 0,871 en entrenamiento y de 0,613 en validación.

La comparación se realizó de forma exploratoria después de la evaluación del modelo de imagen. Sus cifras no deben presentarse como resultados de test. Además, queda por aclarar la comparabilidad del volumen tumoral entre cohortes.
Análisis posteriores de las predicciones guardadas
Se analizaron la incertidumbre del AUC, los resultados por cohorte, la calibración y ejemplos de errores utilizando las predicciones de test ya guardadas. Estos análisis no implicaron volver a entrenar la CNN ni modificar el umbral.
Resultados por procedencia
Cohorte	Pacientes	pCR = 0	pCR = 1	AUC
Duke	42	33	9	0,727
I-SPY1	35	23	12	0,344
I-SPY2	99	67	32	0,483


El rendimiento observado fue diferente entre grupos. Su tamaño limitado, especialmente el número de pacientes positivas, impide considerar esas diferencias como definitivas o atribuirlas automáticamente a una causa concreta.
Calibración y revisión de errores
El Brier score global fue 0,247. Esta medida resume el error de las puntuaciones respecto a las etiquetas y no mide exclusivamente la calibración. Los intervalos de puntuación más altos contenían pocas pacientes, por lo que no permiten interpretar las salidas como probabilidades clínicas fiables.

También se revisaron ejemplos de falsos positivos y falsos negativos, comparando las tres fases originales con las entradas preprocesadas. La revisión permite formular hipótesis sobre calidad, recorte e intensidades, pero no demuestra por sí sola la causa de un error.

Las pacientes ISPY1_1148 e ISPY1_1193 se conservaron en la validación original de 219 pacientes del modelo final. Se probaron exclusiones por calidad en otros experimentos, que deben identificarse por separado. Sus cortes de aspecto atípico requieren revisión adicional; una diferencia de brillo no demuestra por sí sola que una imagen sea incorrecta.
Aplicación web y distribución del modelo
La aplicación se publica con GitHub Pages y ejecuta la inferencia en el navegador mediante ONNX Runtime Web.
1. Seleccionar un ejemplo o cargar las fases PRE, EARLY y LATE del mismo corte y paciente.
2. Preparar la inferencia con el botón Conectar CNN.
3. Analizar los cortes y consultar su agregación por paciente.
4. Opcionalmente, introducir datos clínicos y calcular la comparación CNN + datos.
La interfaz espera PNG en escala de grises de 8 bits, de 256 × 256 píxeles y un máximo de 2 MB por archivo. Si solo se aportan algunos cortes, la media utiliza únicamente esos cortes y puede diferir de la evaluación completa de la paciente.

Antes de habilitar el análisis, la aplicación comprueba seis referencias generadas en Python. Estas comprobaciones verifican la equivalencia técnica del preprocesado y de la inferencia; no significan que el modelo esté limitado a seis pacientes ni demuestran utilidad clínica.
GitHub y Hugging Face
- El release de GitHub distribuye el ONNX que se incorpora a la web durante el despliegue.
- GitHub Pages aloja la aplicación y sirve el archivo incorporado al sitio.
- Hugging Face contiene un repositorio del modelo y su documentación. No aloja la web ni es la fuente de descarga utilizada por esta aplicación.
Para la comparación clínica, la web necesita además el clasificador original clasificador_combinado.json y el modelo de extracción de características cnn_pcr_features.onnx. Sus parámetros deben mantenerse coherentes con los del experimento exportado.
Organización del proyecto
Ubicación	Contenido
app/	Interfaz web, scripts, ejemplos y recursos de inferencia.
src/	Código Python de preprocesado, entrenamiento y análisis.
src/archivos_app/	Scripts auxiliares para los análisis y recursos de la aplicación.
resultados/	Historiales, métricas, figuras y análisis generados en los experimentos.
.github/workflows/pages.yml	Flujo de despliegue de GitHub Pages.


Algunos datos, pesos y resultados generados pueden conservarse fuera del repositorio de código. La estructura anterior describe la organización del proyecto y no implica que todos los archivos estén distribuidos públicamente.
Conclusiones y límites
- La CNN permitió construir y comprobar un flujo completo de predicción por paciente, pero mostró una capacidad de discriminación limitada en el test.
- En validación, el clasificador clínico superó a la combinación de clínica e imagen estudiada.
- Los resultados varían entre cohortes y presentan incertidumbre.
- El modelo analiza cortes 2D; no representa directamente las relaciones espaciales del volumen completo.
- Las puntuaciones no están validadas como probabilidades clínicas calibradas.
- La búsqueda entre múltiples experimentos sobre una misma validación puede favorecer estimaciones optimistas. Los análisis posteriores al test son exploratorios.
El proyecto documenta tanto los resultados como sus limitaciones para facilitar su revisión y orientar futuros experimentos con una evaluación independiente.
Autoría
Claudia Lozano Pérez · Grado en Ingeniería Matemática · Proyecto académico de Aprendizaje Automático.

La disponibilidad pública del código no implica que los datos de origen puedan redistribuirse libremente. Deben respetarse las condiciones de uso de sus fuentes.
