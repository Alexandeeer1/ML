import os
import pickle

carpeta = r'C:\Users\alexa\OneDrive\Desktop\ML- proyecto\Dataset'
datos = []

for archivo in os.listdir(carpeta):
    ruta_archivo = os.path.join(carpeta, archivo)
    with open(ruta_archivo, 'rb') as file:
        contenido = file.read()
        datos.append(contenido)

ruta_pickle = 'Dataset.pickle'

with open(ruta_pickle, 'wb') as file:
    pickle.dump(datos, file)
