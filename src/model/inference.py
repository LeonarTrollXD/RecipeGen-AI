import numpy as np
from deep_translator import GoogleTranslator

from src.model.preprocess import cargar_y_preprocesar_datos


def recomendar_receta_por_similitud(ingredientes_espanol):
    """
    Traduce ingredientes del español al inglés,
    busca la receta más compatible mediante
    similitud del coseno y traduce el resultado
    nuevamente al español.
    """
    
    # =================================================================
    # 1. INICIALIZAR TRADUCTORES
    # =================================================================
    
    traductor_ingles = GoogleTranslator(
        source='es',
        target='en'
    )
    
    traductor_espanol = GoogleTranslator(
        source='en',
        target='es'
    )
    
    print(f"\n[Backend] Entrada original: {ingredientes_espanol}")
    
    # =================================================================
    # 2. TRADUCIR INGREDIENTES AL INGLÉS
    # =================================================================
    
    ingredientes_ingles = traductor_ingles.translate(
        ingredientes_espanol
    )
    
    ingredientes_limpios = (
        ingredientes_ingles
        .replace(",", " ")
        .lower()
    )
    
    print(f"[Backend] Traducido para el buscador: {ingredientes_limpios}")
    
    # =================================================================
    # 3. CARGAR DATASET PREPROCESADO
    # =================================================================
    
    (
        X,
        _,
        _,
        _,
        tokenizer,
        lista_recetas
    ) = cargar_y_preprocesar_datos()
    
    # =================================================================
    # 4. CONVERTIR ENTRADA DEL USUARIO A VECTOR BINARIO
    # =================================================================
    
    vector_usuario = tokenizer.texts_to_matrix(
        [ingredientes_limpios],
        mode='binary'
    )[0]
    
    # =================================================================
    # 5. CALCULAR SIMILITUD DEL COSENO
    # =================================================================
    
    # Producto punto entre usuario y todas las recetas
    dot_product = np.dot(
        X,
        vector_usuario
    )
    
    # Norma de cada receta
    norm_X = np.linalg.norm(
        X,
        axis=1
    )
    
    # Norma del usuario
    norm_usuario = np.linalg.norm(
        vector_usuario
    )
    
    # Evitar división por cero
    if norm_usuario == 0:
        
        return (
            "No se ingresaron ingredientes válidos.",
            0
        )
    
    # Fórmula de similitud del coseno
    similitudes = dot_product / (
        norm_X * norm_usuario
    )
    
    # =================================================================
    # 6. OBTENER RECETA MÁS COMPATIBLE
    # =================================================================
    
    id_mejor_receta = np.argmax(
        similitudes
    )
    
    porcentaje_coincidencia = (
        similitudes[id_mejor_receta] * 100
    )
    
    receta_ganadora_ingles = (
        lista_recetas[id_mejor_receta]
    )
    
    # =================================================================
    # 7. TRADUCIR RESULTADO AL ESPAÑOL
    # =================================================================
    
    print(
        "[Backend] Traduciendo receta compatible al español..."
    )
    
    receta_espanol = traductor_espanol.translate(
        receta_ganadora_ingles
    )
    
    # =================================================================
    # 8. RETORNAR RESULTADO
    # =================================================================
    
    return (
        receta_espanol,
        porcentaje_coincidencia
    )


# =====================================================================
# PRUEBA DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    ingredientes_prueba = (
        "huevo, queso, cebolla"
    )
    
    receta, porcentaje = (
        recomendar_receta_por_similitud(
            ingredientes_prueba
        )
    )
    
    print("\n" + "=" * 70)
    print(f"🛒 Ingredientes ingresados : {ingredientes_prueba}")
    print(f"🎯 Coincidencia IA         : {porcentaje:.2f}%")
    print("-" * 70)
    print(f"📖 Receta recomendada (ES):\n{receta}")
    print("=" * 70 + "\n")