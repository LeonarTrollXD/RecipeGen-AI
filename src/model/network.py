import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def crear_modelo_recetas(total_palabras, largo_maximo_entrada):
    """
    Define la arquitectura de la red neuronal LSTM para generar recetas.
    """
    model = Sequential([
        # 1. Capa Embedding: Convierte los números en vectores densos con significado conceptual
        Embedding(input_dim=total_palabras, output_dim=64, input_length=largo_maximo_entrada),
        
        # 2. Capa LSTM: La memoria de la red. Analiza la secuencia de los ingredientes
        LSTM(128, return_sequences=False), 
        
        # 3. Capa Dropout: Apaga neuronas al azar en el entrenamiento para evitar que se memorice todo (overfitting)
        Dropout(0.2),
        
        # 4. Capa Densa de Salida: Capa final que predecirá las palabras de la receta resultante
        # Se usa 'softmax' para obtener probabilidades de qué palabra sigue
        Dense(total_palabras, activation='softmax')
    ])
    
    # Compilamos el modelo configurando cómo va a aprender
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    print("¡Arquitectura de la Red Neural creada con éxito!")
    model.summary() # Esto dibuja un mapa de la red en la terminal
    return model



if __name__ == "__main__":
    # Este bloque obliga a Python a ejecutar la función al correr el archivo
    print("Iniciando prueba de la red neuronal...")
    crear_modelo_recetas(total_palabras=100, largo_maximo_entrada=5)