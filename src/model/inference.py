import numpy as np
import tensorflow as tf
from src.model.preprocess import cargar_y_preprocesar_datos

def recomendar_receta(ingredientes_input):
    """
    Recomienda la receta más compatible según
    los ingredientes ingresados por el usuario.
    """
    
    # =================================================================
    # 1. CARGAR MODELO ENTRENADO
    # =================================================================
    
    modelo = tf.keras.models.load_model(
        "models/saved_models/modelo_recetas.keras"
    )
    
    # =================================================================
    # 2. CARGAR TOKENIZER Y RECETAS
    # =================================================================
    
    # Ignoramos variables innecesarias usando "_"
    _, _, _, _, tokenizer, lista_recetas = cargar_y_preprocesar_datos()
    
    # =================================================================
    # 3. CONVERTIR TEXTO A VECTOR BINARIO
    # =================================================================
    
    entrada = tokenizer.texts_to_matrix(
        [ingredientes_input],
        mode='binary'
    )
    
    print("\n[IA] Buscando el match perfecto en la base de datos...")
    
    # =================================================================
    # 4. REALIZAR PREDICCIÓN
    # =================================================================
    
    prediccion = modelo.predict(
        entrada,
        verbose=0
    )
    
    # =================================================================
    # 5. OBTENER RECETA GANADORA
    # =================================================================
    
    # Índice de la receta con mayor probabilidad
    id_receta = np.argmax(prediccion[0])
    
    # Porcentaje de confianza
    probabilidad = prediccion[0][id_receta] * 100
    
    # Obtener texto real de la receta
    receta_encontrada = lista_recetas[id_receta]
    
    return receta_encontrada, probabilidad


# =====================================================================
# PRUEBA DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    # Ingredientes de prueba
    ingredientes_prueba = "huevos"
    
    receta, probabilidad = recomendar_receta(
        ingredientes_prueba
    )
    
    print("\n" + "=" * 70)
    print(f"🛒 Ingredientes disponibles : {ingredientes_prueba}")
    print(f"🎯 Certeza de la IA         : {probabilidad:.2f}%")
    print("-" * 70)
    print(f"📖 Receta recomendada       :\n{receta}")
    print("=" * 70 + "\n")