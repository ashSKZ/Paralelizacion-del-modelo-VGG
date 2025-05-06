import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

#parámetros para el modelo VGG
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = r"C:\Users\samas\OneDrive\Documents\6to semestre\computo pararelo\Proyecto paralelizacion VGG\resultantes"

#aumentar imagenes para el enteno
train_augmented_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

#imágenes sin aumento para validación
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

#entrenamiento
train_generator = train_augmented_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

#validación
val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

#modelo tipo VGG
def build_vgg_like_model(input_shape=(224, 224, 3), num_classes=5):
    model = models.Sequential()

    # Bloque 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bloque 2
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Bloque 3
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Clasificador
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

#construir y compilar el modelo
model = build_vgg_like_model(input_shape=(224, 224, 3), num_classes=train_generator.num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#entrenamiento del modelo con aumento de datos
steps_per_epoch = (train_generator.samples // BATCH_SIZE) * 4

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    epochs=EPOCHS
)

# guardar el modelo
model.save("modelo_vgg_flowers.h5")
print("Modelo guardado como modelo_vgg_flowers.h5")

# evaluación del modelo
class_names = list(train_generator.class_indices.keys())

val_generator.reset()
pred_probs = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = val_generator.classes

print("\n=== Reporte de clasificación ===")
print(classification_report(true_labels, pred_labels, target_names=class_names))

#Matriz de confusión
conf_matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

#Visualización de métricas de entrenamiento

# Precisión
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()

# Pérdida
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.show()
