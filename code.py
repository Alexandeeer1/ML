import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Ruta al directorio de nuestro dataset
dataset_dir = "C:\Users\alexa\OneDrive\Desktop\ML- proyecto\Dataset"  # Carpeta en donde están guardadas las imágenes
classes = os.listdir(dataset_dir)
num_classes = len(classes)

# Pre-procesamiento de imágenes y etiquetas
images = []
labels = []
for i, sign_class in enumerate(classes):
    class_dir = os.path.join(dataset_dir, sign_class)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue  # Salta las imágenes que no se puedan leer correctamente
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))  # Ajustamos el tamaño
        images.append(image)
        labels.append(i)

# Convertimos las listas a arrays numpy
images = np.array(images)
labels = np.array(labels)

# División del dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalización de píxeles
X_train = X_train / 255.0
X_test = X_test / 255.0

# Agregamos una dimensión de canal para las imágenes en escala de grises
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Creamos el modelo LeNet-5
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compilamos y entrenamos el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Precisión en el conjunto de prueba:", test_accuracy)