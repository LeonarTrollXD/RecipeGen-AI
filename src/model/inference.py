import numpy as np
import tensorflow as tf
from src.model.preprocess import cargar_y_preprocesar_datos

def generar_receta_final(ingredientes_input):
    # 1. Cargar modelo y tokenizer
    modelo = tf.keras.models.load_model("models/saved_models/modelo_recetas.keras")
    _, tokenizer, _ = cargar_y_preprocesar_datos()
    
    # 2. Procesar entrada
    secuencia = tokenizer.texts_to_sequences([ingredientes_input])
    entrada = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=15, padding='post')
    
    # 3. Predicción
    prediccion = modelo.predict(entrada)
    
    # 4. Traducción de números a palabras
    # Buscamos el índice con la probabilidad más alta
    index_palabra = np.argmax(prediccion, axis=-1)[0]
    
    # Invertimos el diccionario del tokenizer para encontrar la palabra
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    palabra_resultado = reverse_word_map.get(index_palabra, "<Desconocido>")
    
    return palabra_resultado

if __name__ == "__main__":
    test_ingredientes = "tomates atún"
    resultado = generar_receta_final(test_ingredientes)
    print(f"Ingredientes de entrada: {test_ingredientes}")
    print(f"La IA sugiere incluir: {resultado}")