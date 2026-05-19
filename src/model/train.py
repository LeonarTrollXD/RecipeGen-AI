import numpy as np
# Importamos tus dos funciones previas
from preprocess import cargar_y_preprocesar_datos
from network import crear_modelo_recetas

def iniciar_entrenamiento():
    print("=== INICIANDO PROCESO DE ENTRENAMIENTO ===")
    
    # 1. Cargamos y preprocesamos los datos de prueba
    ruta_datos = "data/raw/recetas_prueba.csv"
    datos_entrada, tokenizer = cargar_y_preprocesar_datos(ruta_datos)
    
    # En un problema real de generación de texto, intentamos predecir la siguiente palabra.
    # Para esta prueba rápida, crearemos objetivos ficticios emparejados con la entrada.
    total_palabras = len(tokenizer.word_index) + 1
    largo_maximo = datos_entrada.shape[1]
    
    # Creamos etiquetas de salida de prueba (falsas) con el mismo tamaño
    datos_salida = np.random.randint(0, total_palabras, size=(datos_entrada.shape[0],))

    # 2. Construimos la red neuronal
    modelo = crear_modelo_recetas(total_palabras=total_palabras, largo_maximo_entrada=largo_maximo)
    
    # 3. Entrenamos el modelo (Bucle de entrenamiento)
    # Usamos pocas épocas (epochs) solo para verificar que corra sin errores
    print("\nEntrenando la red...")
    modelo.fit(datos_entrada, datos_salida, epochs=5, batch_size=2)
    
    # 4. Guardamos el modelo entrenado para que la Vista/Controlador lo usen después
    ruta_guardado = "models/saved_models/modelo_recetas.keras"
    modelo.save(ruta_guardado)
    print(f"\n¡Cerebro de la IA guardado exitosamente en: {ruta_guardado}!")

if __name__ == "__main__":
    iniciar_entrenamiento()