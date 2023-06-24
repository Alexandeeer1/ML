import os
import pickle

carpeta = r'C:\Users\alexa\OneDrive\Desktop\ML- proyecto\Dataset'  # Ruta de la carpeta a convertir'  # Ruta de la carpeta a convertir
datos = []

# Recorrer los archivos de la carpeta
for archivo in os.listdir(carpeta):
    ruta_archivo = os.path.join(carpeta, archivo)
    with open(ruta_archivo, 'rb') as file:
        # Leer el contenido de cada archivo y agregarlo a la lista de datos
        contenido = file.read()
        datos.append(contenido)

ruta_pickle = 'DataSet.pickle'  # Ruta y nombre del archivo .pickle de salida

# Guardar la lista de datos en el archivo .pickle
with open(ruta_pickle, 'wb') as file:
    pickle.dump(datos, file)
