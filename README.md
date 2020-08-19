# Detección del lenguaje original de un texto traducido por máquina

## Requerimientos
* [Python 3 >=3.6](https://www.python.org/)
* [virtualenv](https://pypi.org/project/virtualenv/)

## Instalación
Crear un nuevo ambiente virtual utilizando virtualenv y activarlo:
```console
virtualenv venv
source venv/bin/activate
```
Instalar los paquetes definidos en el archivo requirements.txt:
```console
pip install -r requirements.txt
```

## Pasos para replicar la investigación
Descargar el conjunto de datos del parlamento europeo en [https://www.statmt.org/europarl/v7/europarl.tgz](https://www.statmt.org/europarl/v7/europarl.tgz).
Descomprimir el archivo y ejecutar el script de alineación **sentence-align-corpusl.perl** que se incluye dentro de la carpeta descomprimida. Este script debe ejecutarse para cada par de idiomas entre inglés y los idiomas que se desean utilizar. El idioma inglés es el idioma "eje" que se utilizará posteriormente para crear un conjunto de oraciones totalmente alineado.

### Preprocesamiento de los datos para el motor de traducciones
El procesamiento de los datos consiste en generar un conjunto de oraciones totalmente alinieadas que tienen injectadas un token con la información de la traducción que debe realizar el motor de traducciones, como se explica en el artículo [Training Romance Multi-Way model](https://forum.opennmt.net/t/training-romance-multi-way-model/86). Este conjunto se utilizará para entrenar dicho motor.

Para realizar este procesamiento es necesario ejecutar la primera sección del archivo **preprocess.ipynb**, nombrada *Preprocessing europarl data*, hasta la sub-sección *Write new file(s)*.

Luego se debe ejecutar el script para la alineación múltiple:
```console
python tools/multialign.py
```
A continuación se debe ejecutar el script para entrenar el modelo de BPE que se utilizará para la tokenización de las oraciones y el script de tokenización:
```console
python tools/learn_bpe.py
./bpe_preprocess.sh
```
Este último (bpe_preprocess.sh) es un recopilación de llamadas al script **tokenize.py** que se encuentra en la carpeta tools. Si se desea utilizar este programa para otra cosa se puede ver su documentación con la bandera `-h`.

Por último se debe ejecutar la última sub-sección del procesamiento del conjunto de datos del parlamento europeo llamada *Join language files and add translation token* en el archivo **preprocess.ipynb**.

### Entrenamiento del motor de traducciones
Para comenzar a entrenar el motor de traducciones, primero se deben transformar los datos generados en los pasos anterior al formato que requiere **OpenNMT-py**
con el siguiente commando:
```console
onmt_preprocess -train_src data/europarl_train_src.txt -train_tgt data/europarl_train_tgt.txt -valid_src data/europarl_validation_src.txt -valid_tgt data/europarl_validation_tgt.txt -save_data data/multi-europarl -overwrite -shard_size 500000
```
Luego se puede ejecutar el script de entrenamiento de la siguiente manera:
```console
./onmt_train.sh data/multi-europarl models/multi-europarl 100000 100001 4096 2500 {checkpoint_file_to_load}
```
(Ver el archivo **onmt_train.sh** para ver qué significa cada parámetro)
El último parametro de este script es opcional.

Si se desea utilizar **Google Colaboratory** puede subir los archivos generados por el comando `onmt_preprocess` a **Google Drive** y abrir el archivo **colab_sandbox.ipynb** desde su navegador con el bótón que se encuentra en la parte superior del mismo.

Por últitmo se debe convertir el modelo resultante o el último checkpoint guardado al formato utilizado por el servidor con el siguiente comando:
```console
ct2-opennmt-py-converter --model_path models/multi-europarl_step_200000.pt --model_spec TransformerBase --output_dir models/multi_lang_final
```

### Ejecución del servidor de traducciones
Para ejecutar el servidor de traducciones utilizando el CPU basta con ejecutar el script **server.py** que se encuentra en la carpeta **translation_server** de la siguiente manera:
```console
python translation_server/server.py --beam_size 5 --max_batch_size 300
```
Si se desea realizar las traducciones en un GPU se debe compilar la imagen de [Docker](https://www.docker.com/) descrita en el **Dockerfile** y ejecutar un contenedor que la utilice.
Ejemplo:
```console
docker build --pull --rm -f "Dockerfile" -t languagedetection:latest "."
docker run --gpus=all -p 8080:8080 -it --rm languagedetection --beam_size 5 --batch_size 60
```
Los valores máximos de **beam_size** y **batch_size** soportados dependerán de las características de su GPU.

### Preprocesamiento del clasificador
El preprocesamiento del clasificador consiste en generar una tabla con los puntajes BLEU de las traducciones generada a partir de las Back-Translations de las oraciones del conjunto de datos. Para esto se debe ejecutar la sección llamada *Preprocessing classifier data* del archivo **preprocess.ipynb** sobre una carpeta de archivos traducidos por máquina. En este caso se utilizó el script **google_translator.py** para generar dichos archivos de la siguiente manera:
```console
py google_translator.py -s data/aligned/validation/fr.txt -o data/aligned/validation/fr_de.txt -tgt de
```
para cada par de idiomas de los archivos generados para la validación en el procesamiento del conjunto de datos del motor de traducciones (ver documentación del script con `google_translator -h` para más información).

### Entrenamiento del clasificador
Para el entrenamiento del clasificador se puede ver y ejecutar el archivo **classifier_training.ipynb**.

### Dashboard y ambiente de pruebas en tiempo real (EN CONSTRUCCIÓN)
Ejecutar el programa de pruebas:
```console
python real_time_test.py
```
El programa levantará un servidor en la dirección: [http://127.0.0.1:8050/](http://127.0.0.1:8050/)
