import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


def unificar_datos_relacionales():
    print("Leyendo y unificando bases de datos de recetas...")
    
    # 1. Leer los 3 CSVs
    df_recipes = pd.read_csv('data/raw/recipes.csv')
    df_ingredients = pd.read_csv('data/raw/ingredients.csv')
    df_pivot = pd.read_csv('data/raw/ingredients_recipes.csv')
    
    # 2. Renombrar columnas clave
    df_recipes = df_recipes.rename(columns={
        'id': 'recipe_id',
        'description': 'receta_resultado'
    })

    df_ingredients = df_ingredients.rename(columns={
        'id': 'ingredient_id',
        'name': 'ingredient_name'
    })
    
    # 3. JOINs
    df_merged = pd.merge(
        df_pivot,
        df_recipes[['recipe_id', 'receta_resultado']],
        on='recipe_id'
    )

    df_merged = pd.merge(
        df_merged,
        df_ingredients[['ingredient_id', 'ingredient_name']],
        on='ingredient_id'
    )
    
    # 4. GROUP BY para unir ingredientes
    df_final = df_merged.groupby(
        ['recipe_id', 'receta_resultado']
    )['ingredient_name'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    
    # Renombrar columna final
    df_final = df_final.rename(columns={
        'ingredient_name': 'ingredientes'
    })
    
    print(f"¡Unificación completa! Total de recetas armadas: {len(df_final)}")
    
    return df_final

def cargar_y_preprocesar_datos():
    
    # =====================================================================
    # 1. CARGAR Y UNIFICAR DATOS
    # =====================================================================
    
    df = unificar_datos_relacionales()
    
    # Convertimos columnas a listas
    lista_ingredientes = df['ingredientes'].astype(str).tolist()
    lista_recetas = df['receta_resultado'].astype(str).tolist()
    
    num_recetas = len(lista_ingredientes)
    
    # =====================================================================
    # 2. VECTORIZACIÓN BINARIA
    # =====================================================================
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lista_ingredientes)
    
    # Convierte ingredientes en vectores binarios
    X = tokenizer.texts_to_matrix(lista_ingredientes, mode='binary')
    
    # IDs de recetas
    Y = np.arange(num_recetas)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    print(f"Vocabulario total de ingredientes únicos: {vocab_size}")
    print("¡Datos convertidos a vectores binarios con éxito!")
    
    # Retornamos todo lo necesario
    return X, Y, vocab_size, num_recetas, tokenizer, lista_recetas

# Probar directamente el script
if __name__ == "__main__":
    
    X, Y, vocab_size, num_recetas, tokenizer, lista_recetas = cargar_y_preprocesar_datos()
    
    print("\nEjemplo del primer vector:")
    print(X[0])
    
    print("\nPrimera receta:")
    print(lista_recetas[0][:100], "...")