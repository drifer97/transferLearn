import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Caminho para carregar o modelo salvo
model_path = "E:\\transferlearning\\modelo_cats_dogs.h5"

# Carregar o modelo salvo
model = tf.keras.models.load_model(model_path)
print("Modelo carregado com sucesso!")

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
test_image_path = "E:\\transferlearning\\4975261_1920.jpg"  # Altere o caminho da imagem para o seu

# Chame a função para fazer a previsão
predict_image(test_image_path, model)
