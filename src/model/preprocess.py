import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def unificar_datos_relacionales():
    print("Leyendo y unificando bases de datos de recetas...")
    
    # 1. Leer los 3 CSVs
    df_recipes = pd.read_csv('data/raw/recipes.csv')
    df_ingredients = pd.read_csv('data/raw/ingredients.csv')
    df_pivot = pd.read_csv('data/raw/ingredients_recipes.csv')
    
    # 2. Renombrar columnas clave para evitar confusiones en los JOINs
    df_recipes = df_recipes.rename(columns={'id': 'recipe_id', 'description': 'receta_resultado'})
    df_ingredients = df_ingredients.rename(columns={'id': 'ingredient_id', 'name': 'ingredient_name'})
    
    # 3. Hacer los JOINs (Inner Merge)
    # Unimos la tabla pivote con las recetas
    df_merged = pd.merge(df_pivot, df_recipes[['recipe_id', 'receta_resultado']], on='recipe_id')
    # Unimos el resultado con los nombres de los ingredientes
    df_merged = pd.merge(df_merged, df_ingredients[['ingredient_id', 'ingredient_name']], on='ingredient_id')
    
    # 4. Agrupar (GROUP BY) por receta para concatenar los ingredientes
    # Esto junta todos los ingredientes de una misma receta en una sola cadena de texto separada por espacios
    df_final = df_merged.groupby(['recipe_id', 'receta_resultado'])['ingredient_name'].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    
    # Renombramos la columna final para mantener la compatibilidad con nuestro código anterior
    df_final = df_final.rename(columns={'ingredient_name': 'ingredientes'})
    
    print(f"¡Unificación completa! Total de recetas armadas: {len(df_final)}")
    return df_final

def cargar_y_preprocesar_datos():
    # 1. Obtenemos el DataFrame ya unificado con los 3 CSVs
    df = unificar_datos_relacionales()
    
    # Separamos las columnas
    textos_ingredientes = df['ingredientes'].astype(str).tolist()
    textos_recetas = df['receta_resultado'].astype(str).tolist()
    
    # 2. Tokenización
    tokenizer_ingredientes = Tokenizer(oov_token="<OOV>")
    tokenizer_ingredientes.fit_on_texts(textos_ingredientes)
    secuencias = tokenizer_ingredientes.texts_to_sequences(textos_ingredientes)
    
    # 3. Padding (Ajustamos el tamaño máximo, por ejemplo a 15 ingredientes por receta)
    datos_entrada = pad_sequences(secuencias, maxlen=15, padding='post')
    
    print("¡Datos tokenizados con éxito!")
    return datos_entrada, tokenizer_ingredientes, textos_recetas
# Esto permite probar el script directamente corriendo: python src/model/preprocess.py
if __name__ == "__main__":
    datos_entrada, tokenizer, textos_recetas = cargar_y_preprocesar_datos()
    print("\nEjemplo de la primera receta armada:")
    print("Ingredientes en texto:", tokenizer.sequences_to_texts([datos_entrada[0]])[0])
    print("Receta original:", textos_recetas[0][:50], "...")