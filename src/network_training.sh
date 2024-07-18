#!/bin/bash

# Instrucciones: cambie el nombre de trabajo para identificarlo
#               fácilmente, el nombre del ejecutable y sus parámetros.

# Grupo en el que va a ejecutar. Sólo el grupo GPU sirve para haver ejecuciones CUDA.
#SBATCH --partition=GPU

# Nombre del trabajo. Puede cambiarse para identificar de forma más sencilla.
#SBATCH --job-name=BrD-GRU

# Numero de hilos a ejecutar
# Valor recomendado mínimo: 12 para C1, 24 para C2. 24 para GPU.
#SBATCH --cpus-per-task=48

# Solo es necesario cambiarlo si se va a hacer paralelos híbridos que usen paralelización
# en CPUs y las tarjetas gŕaficas.
#SBATCH --ntasks=1

# El archivo de log donde quedará lo que imprima el software en pantalla.
# Lo recomendable es que el programa no imprama nada a pantalla, sino a fichero directamente
#SBATCH --output=test_brainweb_raw_ensemble_pd.log

# Especificar la cantidad máxima de memoria que podrá utilizar cada nodo.
# El valor 0 indica que podrá utilizar toda la memoria.
#SBATCH --mem=0

# Si el proceso tarda más del tiempo especificado en time, automáticamente se terminará,
# lo que permitirá evitar que un proceso en el que hubo algún tipo de error, como un bucle
# infinito, se quede ejecutando por mucho tiempo. EL valor 0 indica que no se mata el proceso
# nunca, con lo que en caso de errores de ese tipo lo deberá de terminal manualmente
# (con scancel).
#SBATCH --time=0

# En la siguiente linea cambie la ruta, el nombre del ejecutable compilado con nvcc e
# incluya al final los parámetros de su ejecutable si procede.
# ruta/del/ejecutable.out

# Primero se activará el anaconda, luego se ativará el entorno virtual y finalmente se mandará
# a ejecutar el script de python. Además, este scipt se almacena junto al archivo que main que
# se manda ejecutar.

source /opt/anaconda3_titan/bin/activate
source activate pytorch
# nvidia-smi
# python main_train.py
# python main_test_brainweb.py
# python main_test_ixi.py
python main_test_brainweb_rawb.py

# Para enviarlo en la bash escriba sbatch nombreDeEsteFichero
