import tensorflow as tf

# Importamos nuestras funciones
from src.model.preprocess import cargar_y_preprocesar_datos
from src.model.network import crear_modelo_recetas


def entrenar_modelo():
    """
    Ejecuta el proceso completo de entrenamiento
    del modelo clasificador de recetas.
    """
    
    print("=== INICIANDO PROCESO DE ENTRENAMIENTO ===")
    
    # =================================================================
    # 1. CARGAR Y PREPROCESAR DATOS
    # =================================================================
    
    X, Y, vocab_size, num_recetas, tokenizer, lista_recetas = (
        cargar_y_preprocesar_datos()
    )
    
    # =================================================================
    # 2. CREAR LA RED NEURONAL
    # =================================================================
    
    modelo = crear_modelo_recetas(
        vocab_size,
        num_recetas
    )
    
    # =================================================================
    # 3. CONFIGURAR MODEL CHECKPOINT
    # =================================================================
    
    # Guarda automáticamente el mejor modelo encontrado
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        
        filepath="models/saved_models/modelo_recetas.keras",
        
        # Vigilar la pérdida
        monitor="loss",
        
        verbose=1,
        
        # Solo guardar el mejor modelo
        save_best_only=True,
        
        # Mientras menor sea la pérdida, mejor
        mode="min"
    )
    
    print("\nEntrenando la red clasificadora...")
    
    # =================================================================
    # 4. ENTRENAMIENTO
    # =================================================================
    
    modelo.fit(
        
        # Datos de entrada
        X,
        
        # Etiquetas (IDs de recetas)
        Y,
        
        # Número de épocas
        epochs=200,
        
        # Cantidad de muestras por lote
        batch_size=64,
        
        # Callback para guardar el mejor modelo
        callbacks=[checkpoint]
    )
    
    print("\n¡Entrenamiento finalizado y mejor modelo guardado!")


# =====================================================================
# EJECUCIÓN DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    entrenar_modelo()