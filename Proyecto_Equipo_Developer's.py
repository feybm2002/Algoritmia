# Equipo Developer's

!pip install ipython # Instalar python

import os # Libreria para manipulacion de archivos o directorios
import random # Libreria para valores aleatorios
import numpy as np # Libreria para creacion de matrices 
import matplotlib.pyplot as plt # Libreria para graficar 
import tensorflow as tf # Libreria para creacion y entrenamiento de una NN
from tensorflow.keras.preprocessing.image import load_img, img_to_array # Libreria para procesar las imagenes de una CNN
from tensorflow.keras.models import load_model  # Libreria para creacion y entrenamiento de una NN


os.environ["KAGGLE_USERNAME"] = "feybm2"
os.environ["KAGGLE_KEY"] = "e61ed60d5af3d7884c409a4332f9ffe5"
os.environ['KAGGLE_CONFIG_DIR'] = "/content"

# Descarga el conjunto de datos
!kaggle datasets download -d kausthubkannan/5-flower-types-classification-dataset

# Descomprime el conjunto de datos descargado
!unzip 5-flower-types-classification-dataset.zip

# Parámetros del modelo
image_size = (224, 224)
batch_size = 32
num_classes = 5  # Número total de clases

# Generador de datos de entrenamiento y validación
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Directorio de datos
data_dir = "flower_images"

# Lista de nombres de las clases
class_names = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

# Entrenamiento por cada clase
for class_name in class_names:
    print(f"Training model for class: {class_name}")
    
    # Generador de datos de entrenamiento y validación para la clase actual
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        classes=[class_name],
        shuffle=True,
        seed=42
    )

    valid_generator = datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        classes=[class_name]
    )

    # Restablece el modelo para el próximo entrenamiento
    tf.keras.backend.clear_session()

    # Carga del modelo base preentrenado
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    # Congela las capas del modelo base
    base_model.trainable = False

    # Agrega capas adicionales al modelo
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Construye el modelo completo
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    # Compila el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrena el modelo
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs= 20
    )

    # Evaluación del modelo en el conjunto de validación
    loss, accuracy = model.evaluate(valid_generator)
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    # Guarda el modelo entrenado
    model.save(f'model_{class_name}.h5')

# Fin del entrenamiento por cada clase
print("Training completed for all classes.")

# Selecciona una clase al azar
class_name = random.choice(class_names)

# Ruta a la carpeta de la clase seleccionada
class_dir = os.path.join(data_dir, class_name)

# Lista de imágenes en la carpeta de la clase seleccionada
image_files = os.listdir(class_dir)

# Selecciona una imagen al azar
image_file = random.choice(image_files)

# Ruta a la imagen seleccionada
image_path = os.path.join(class_dir, image_file)

# Carga la imagen y la convierte en un arreglo
image = load_img(image_path, target_size=(224, 224))
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)
image_array /= 255.0

# Carga el modelo entrenado correspondiente a la clase seleccionada
model_path = f'model_{class_name}.h5'
model = load_model(model_path)

# Realiza la predicción con la imagen
prediction = model.predict(image_array)

# Obtiene la clase predicha
predicted_class_index = np.argmax(prediction)
predicted_class = class_names[predicted_class_index]

# Imprime los resultados
print("Image Path:", image_path)
print("True Class:", class_name)
print("Predicted Class:", predicted_class)