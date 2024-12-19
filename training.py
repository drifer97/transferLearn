import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np

# Diretório base do dataset
base_dir = "E:/transferlearning/PetImages"  # Altere para o caminho correto do seu dataset

# Função para verificar imagens corrompidas
def filter_corrupted_images(directory):
    num_corrupted = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)  # Tenta abrir a imagem
                img.verify()  # Verifica se está íntegra
            except (IOError, SyntaxError) as e:
                print(f"Imagem corrompida encontrada: {file_path}, removendo...")
                os.remove(file_path)  # Remove o arquivo corrompido
                num_corrupted += 1
    print(f"Total de imagens corrompidas removidas: {num_corrupted}")

# Verificar e filtrar imagens corrompidas
filter_corrupted_images(base_dir)

# Configuração de geradores de dados com aumento de dados
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Modelo pré-treinado
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar pesos do modelo pré-treinado

# Adicionando camadas personalizadas
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Saída binária
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

# Treinamento do modelo
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Salvar o modelo treinado
model.save("E:/transferlearning/modelo_cats_dogs.h5")
print("Modelo salvo com sucesso!")

# Avaliação do modelo
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Gráficos de Acurácia e Perda
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Treinamento')
plt.plot(val_acc, label='Validação')
plt.legend()
plt.title('Acurácia')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Treinamento')
plt.plot(val_loss, label='Validação')
plt.legend()
plt.title('Perda')
plt.show()

# Função para prever a classe de uma nova imagem
def predict_image(img_path, model):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=(150, 150))
    
    # Converter a imagem para array e normalizar os valores
    img_array = image.img_to_array(img) / 255.0  # Normaliza para o mesmo intervalo usado no treinamento
    
    # Expandir a dimensão para se adequar à entrada do modelo
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fazer a previsão
    prediction = model.predict(img_array)
    
    # Interpretar a previsão
    if prediction[0] > 0.5:
        print("Predição: Cachorro")
    else:
        print("Predição: Gato")

# Caminho para a nova imagem que você deseja testar
test_image_path = "E:/transferlearning/teste/gato_ou_cachorro.jpg"  # Altere o caminho da imagem para o seu

# Chame a função para fazer a previsão
predict_image(test_image_path, model)
