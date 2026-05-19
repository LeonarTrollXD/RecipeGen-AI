import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def cargar_y_preprocesar_datos(ruta_csv):
    # 1. Leer el CSV de prueba
    df = pd.read_csv(ruta_csv)
    
    # Separamos las columnas
    textos_ingredientes = df['ingredientes'].astype(str).tolist()
    textos_recetas = df['receta_resultado'].astype(str).tolist()
    
    # 2. Tokenización (Convertir palabras a números)
    # El Tokenizer de TensorFlow asigna un número único a cada palabra
    tokenizer_ingredientes = Tokenizer(oov_token="<OOV>")
    tokenizer_ingredientes.fit_on_texts(textos_ingredientes)
    
    # Convertimos los textos de ingredientes a secuencias de números
    secuencias = tokenizer_ingredientes.texts_to_sequences(textos_ingredientes)
    
    # 3. Padding (Hacer que todas las secuencias tengan el mismo largo)
    # Las redes neuronales necesitan que las entradas tengan un tamaño fijo
    datos_entrada = pad_sequences(secuencias, padding='post')
    
    print("¡Datos procesados con éxito!")
    print("Ejemplo de ingredientes convertidos a números:", datos_entrada[0])
    
    return datos_entrada, tokenizer_ingredientes

# Esto permite probar el script directamente corriendo: python src/model/preprocess.py
if __name__ == "__main__":
    ruta = "data/raw/recetas_prueba.csv"
    cargar_y_preprocesar_datos(ruta)