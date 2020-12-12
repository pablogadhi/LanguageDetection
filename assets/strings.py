BLEU_SCORES = 'Puntajes BLEU:'
METEOR_SCORES = 'Puntajes METEOR:'
TABLE_GEN_EXPL = """Con dicho motor de traducciones se generó una tabla con las
                    diferencias de una serie de Traducciones-Inversas creadas sobre un
                    conjunto de párrafos de al menos 100 palabras. Para esto, se generaron
                    traducciones en los 3 idiomas distintos al idioma de cada párrafo de entrada.
                    Luego se tradujeron de regreso estos textos resultantes al mismo idioma de entrada
                    y se calcularon las diferencias entre ellos y el texto de entrada utilizando la métrica BLEU.
                    Dicho proceso se repite una vez más, pero comparando las Traducciones-Inversas de esta iteración
                    con las de la anterior. El siguiente diagrama representa todo el proceso:"""
TABLE_EXAMPLE = 'La tabla resultante de este proceso se ve como la siguiente:'
CLASSIFIER_INTRO = """Dicha tabla se utilizó para alimentar 4 algoritmos de clasificación: una Red Neuronal,
                        una SVM, un árbol de decisiones y una algoritmo KNN. A continuación se puede 
                        seleccionar cada uno de los algoritmos para ver sus resultados y utilizarlo en la
                        sección de pruebas:"""
DETECT_BTN_LABEL = 'Detectar'
TRANSLATE_BTN_LABEL = 'Traducir'
EN_PROBABILITY = 'Probabilidad de que el texto esté en inglés: {:.3f}'
ES_PROBABILITY = 'Probabilidad de que el texto esté en español: {:.3f}'
FR_PROBABILITY = 'Probabilidad de que el texto esté en francés: {:.3f}'
DE_PROBABILITY = 'Probabilidad de que el texto esté en alemán: {:.3f}'
FINAL_PRED = 'Predicción final: {}'
