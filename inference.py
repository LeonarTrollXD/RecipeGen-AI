import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)

from deep_translator import GoogleTranslator


# =====================================================================
# INICIALIZAR MOTOR VECTORIAL
# =====================================================================

def inicializar_motor_vectorial():
    """
    Carga el dataset completo,
    construye la matriz TF-IDF
    y deja el motor listo para buscar recetas.
    """
    
    print(
        "[Motor-IA] Cargando dataset de recetas..."
    )
    
    # ================================================================
    # 1. CARGAR CSV
    # ================================================================
    
    ruta_csv = 'data/raw/recetas_ingles.csv'
    
    df = pd.read_csv(ruta_csv)
    
    # ================================================================
    # 2. LIMPIEZA BÁSICA
    # ================================================================
    
    df = df.dropna(
        subset=[
            'Cleaned_Ingredients',
            'Instructions',
            'Title'
        ]
    )
    
    df = df.drop_duplicates(
        subset=['Cleaned_Ingredients']
    )
    
    df = df.reset_index(drop=True)
    
    # ================================================================
    # 3. CONVERTIR COLUMNAS A LISTAS
    # ================================================================
    
    lista_ingredientes = (
        df['Cleaned_Ingredients']
        .astype(str)
        .str.lower()
        .tolist()
    )
    
    lista_recetas = (
        df['Instructions']
        .astype(str)
        .tolist()
    )
    
    lista_titulos = (
        df['Title']
        .astype(str)
        .tolist()
    )
    
    # ================================================================
    # 4. CONFIGURAR TF-IDF
    # ================================================================
    
    # stop_words='english'
    # elimina palabras inútiles:
    # and, with, the, etc.
    
    vectorizador = TfidfVectorizer(
        stop_words='english'
    )
    
    # ================================================================
    # 5. CREAR MATRIZ MATEMÁTICA
    # ================================================================
    
    matriz_tfidf = vectorizador.fit_transform(
        lista_ingredientes
    )
    
    print(
        "[Motor-IA] Matriz TF-IDF creada correctamente."
    )
    
    print(
        f"[Motor-IA] Total recetas cargadas: {len(lista_recetas)}"
    )
    
    # ================================================================
    # 6. RETORNAR MOTOR COMPLETO
    # ================================================================
    
    return (
        vectorizador,
        matriz_tfidf,
        lista_recetas,
        lista_titulos
    )


# =====================================================================
# BUSCADOR PRINCIPAL
# =====================================================================

def buscar_receta_optima(
    ingredientes_usuario,
    vectorizador,
    matriz_tfidf,
    lista_recetas,
    lista_titulos
):
    """
    Traduce ingredientes del usuario,
    calcula similitud TF-IDF
    y devuelve la receta más compatible.
    """
    
    # ================================================================
    # 1. INICIALIZAR TRADUCTORES
    # ================================================================
    
    traductor_ingles = GoogleTranslator(
        source='es',
        target='en'
    )
    
    traductor_espanol = GoogleTranslator(
        source='en',
        target='es'
    )
    
    print(
        f"\n[Backend] Entrada usuario (ES): {ingredientes_usuario}"
    )
    
    # ================================================================
    # 2. TRADUCIR AL INGLÉS
    # ================================================================
    
    ingredientes_ingles = (
        traductor_ingles
        .translate(ingredientes_usuario)
        .lower()
    )
    
    print(
        f"[Backend] Traducido (EN): {ingredientes_ingles}"
    )
    
    # ================================================================
    # 3. CONVERTIR ENTRADA A VECTOR TF-IDF
    # ================================================================
    
    vector_usuario = vectorizador.transform(
        [ingredientes_ingles]
    )
    
    # ================================================================
    # 4. CALCULAR SIMILITUDES
    # ================================================================
    
    similitudes = cosine_similarity(
        vector_usuario,
        matriz_tfidf
    ).flatten()
    
    # ================================================================
    # 5. OBTENER MEJOR MATCH
    # ================================================================
    
    id_mejor_match = similitudes.argsort()[-1]
    
    porcentaje_coincidencia = (
        similitudes[id_mejor_match] * 100
    )
    
    # ================================================================
    # 6. VALIDAR RESULTADO
    # ================================================================
    
    if porcentaje_coincidencia < 5.0:
        
        return (
            "No encontré recetas compatibles.",
            0,
            ""
        )
    
    # ================================================================
    # 7. EXTRAER RECETA GANADORA
    # ================================================================
    
    titulo_ingles = (
        lista_titulos[id_mejor_match]
    )
    
    receta_ingles = (
        lista_recetas[id_mejor_match]
    )
    
    # ================================================================
    # 8. TRADUCIR RESULTADO AL ESPAÑOL
    # ================================================================
    
    print(
        "[Backend] Traduciendo resultado al español..."
    )
    
    titulo_espanol = traductor_espanol.translate(
        titulo_ingles
    )
    
    receta_espanol = traductor_espanol.translate(
        receta_ingles
    )
    
    # ================================================================
    # 9. RETORNAR RESULTADO FINAL
    # ================================================================
    
    return (
        receta_espanol,
        porcentaje_coincidencia,
        titulo_espanol
    )


# =====================================================================
# PRUEBA DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    # ================================================================
    # INICIAR MOTOR SOLO UNA VEZ
    # ================================================================
    
    (
        vectorizador,
        matriz_tfidf,
        lista_recetas,
        lista_titulos
    ) = inicializar_motor_vectorial()
    
    # ================================================================
    # INGREDIENTES DE PRUEBA
    # ================================================================
    
    ingredientes_test = (
        "pan, carne molida, tomate, cebolla, queso"
    )
    
    # ================================================================
    # BUSCAR RECETA
    # ================================================================
    
    (
        receta,
        certeza,
        titulo
    ) = buscar_receta_optima(
        ingredientes_test,
        vectorizador,
        matriz_tfidf,
        lista_recetas,
        lista_titulos
    )
    
    # ================================================================
    # MOSTRAR RESULTADOS
    # ================================================================
    
    print("\n" + "=" * 70)
    
    print(
        f"🛒 INGREDIENTES : {ingredientes_test}"
    )
    
    print(
        f"🎯 COINCIDENCIA : {certeza:.2f}%"
    )
    
    print(
        f"🍽️ RECETA       : {titulo}"
    )
    
    print("-" * 70)
    
    print(
        f"📖 INSTRUCCIONES:\n{receta}"
    )
    
    print("=" * 70 + "\n")