import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def crear_modelo_recetas(vocab_size, num_recetas):
    """
    Define la arquitectura de la red neuronal para clasificar recetas
    según los ingredientes ingresados.
    """
    
    modelo = Sequential([
        
        # 1. Capa de Entrada
        # Recibe el vector binario de ingredientes
        Dense(
            128,
            activation='relu',
            input_shape=(vocab_size,)
        ),
        
        # 2. Dropout
        # Evita overfitting apagando neuronas aleatoriamente
        Dropout(0.3),
        
        # 3. Segunda capa oculta
        Dense(
            64,
            activation='relu'
        ),
        
        # 4. Segundo Dropout
        Dropout(0.2),
        
        # 5. Capa de SALIDA
        # Una neurona por cada receta existente
        # softmax convierte las salidas en probabilidades
        Dense(
            num_recetas,
            activation='softmax'
        )
    ])
    
    # =================================================================
    # COMPILACIÓN DEL MODELO
    # =================================================================
    
    modelo.compile(
        optimizer='adam',
        
        # Ideal cuando Y son IDs enteros
        loss='sparse_categorical_crossentropy',
        
        metrics=['accuracy']
    )
    
    print("¡Arquitectura de clasificación creada con éxito!")
    
    # Mostrar resumen de la red
    modelo.summary()
    
    return modelo


# =====================================================================
# PRUEBA DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    print("Iniciando prueba de la red neuronal...")
    
    # Ejemplo de prueba
    modelo = crear_modelo_recetas(
        vocab_size=500,
        num_recetas=183
    )