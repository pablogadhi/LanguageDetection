## Detección del idioma original en textos traducidos por máquina

El siguiente trabajo propone un método para la identificación del idioma original de un texto
traducido automáticamente a español, inglés, alemán o francés. Esto se hizo con la
justificación de expandir algunas investigaciones previas sobre el área de traducciones
automáticas. Para esto se buscó desarrollar un sistema capaz de proporcionar información que
pueda ser relevante para aplicaciones como encontrar la fuente de origen de un texto o
mejorar una traducción automática producida por un motor de traducciones conocido. El
proyecto se realizó utilizando un motor propio de traducciones con el cual se generaron varias
Traducciones-Inversas de un conjunto de textos iterativamente, para cada idioma distinto al
de entrada. Luego se calcularon las diferencias de cada Traducción-Inversa producida en las
iteraciones utilizando el puntaje BLEU, con el fin producir una tabla que contiene dichas
diferencias para cada idioma del conjunto propuesto. Por último se entrenaron 4 clasificadores
utilizando la información de dicha tabla, de manera que intentaran clasificar un serie de
traducciones como textos traducido desde el idioma español, inglés, alemán o francés. La
mejor precisión obtenida por uno de estos clasificadores fue de 58%.

Para el motor de traducciones se entrenó un modelo de tipo Transformer que fuera capaz de traduccir oraciones
entre cualquier permutación de pares del conjunto de idiomas mencionado. A continuación se presentan los mejores resultados del rendimiento que presentó dicho modelo en las métricas BLEU y METEOR:
