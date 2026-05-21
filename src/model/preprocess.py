import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


def cargar_y_preprocesar_datos():
    """
    Carga el nuevo dataset de recetas en inglés,
    limpia los datos y convierte los ingredientes
    en vectores binarios para entrenar la IA.
    """
    
    print("Leyendo el nuevo dataset de recetas en inglés...")
    
    # =================================================================
    # 1. CARGAR CSV
    # =================================================================
    
    ruta_csv = 'data/raw/recetas_ingles.csv'
    
    df = pd.read_csv(ruta_csv)
    
    # =================================================================
    # 2. LIMPIEZA BÁSICA
    # =================================================================
    
    # Eliminar filas vacías en columnas importantes
    df = df.dropna(
        subset=[
            'Cleaned_Ingredients',
            'Instructions'
        ]
    )
    
    # Eliminar duplicados
    df = df.drop_duplicates(
        subset=['Cleaned_Ingredients']
    )
    
    # Resetear índices
    df = df.reset_index(drop=True)
    
    # =================================================================
    # 3. CONVERTIR COLUMNAS A LISTAS
    # =================================================================
    
    # Ingredientes para entrenar la IA
    lista_ingredientes = (
        df['Cleaned_Ingredients']
        .astype(str)
        .str.lower()
        .tolist()
    )
    
    # Recetas finales que devolverá la IA
    lista_recetas = (
        df['Instructions']
        .astype(str)
        .tolist()
    )
    
    num_recetas = len(lista_ingredientes)
    
    print(f"¡Carga completa! Total de recetas: {num_recetas}")
    
    # =================================================================
    # 4. TOKENIZACIÓN
    # =================================================================
    
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(lista_ingredientes)
    
    # =================================================================
    # 5. VECTORIZACIÓN BINARIA
    # =================================================================
    
    # Convierte ingredientes en vectores multi-hot
    X = tokenizer.texts_to_matrix(
        lista_ingredientes,
        mode='binary'
    )
    
    # IDs numéricos de recetas
    Y = np.arange(num_recetas)
    
    # Tamaño del vocabulario
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"Vocabulario total de ingredientes: {vocab_size}")
    print("¡Datos convertidos a vectores binarios con éxito!")
    
    # =================================================================
    # 6. RETORNAR DATOS
    # =================================================================
    
    return (
        X,
        Y,
        vocab_size,
        num_recetas,
        tokenizer,
        lista_recetas
    )


# =====================================================================
# PRUEBA DIRECTA DEL SCRIPT
# =====================================================================

if __name__ == "__main__":
    
    (
        X,
        Y,
        vocab_size,
        num_recetas,
        tokenizer,
        lista_recetas
    ) = cargar_y_preprocesar_datos()
    
    print("\n" + "=" * 60)
    print(f"Dimensiones de X : {X.shape}")
    print(f"Dimensiones de Y : {Y.shape}")
    print(f"Total recetas    : {num_recetas}")
    print(f"Vocabulario      : {vocab_size}")
    print("=" * 60)
    
    print("\nEjemplo de ingredientes vectorizados:")
    print(X[0])
    
    print("\nEjemplo de receta:")
    print(lista_recetas[0][:200], "...")