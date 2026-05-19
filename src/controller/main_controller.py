import tensorflow as tf
import numpy as np
# Importamos las herramientas de tu modelo
from src.model.preprocess import cargar_y_preprocesar_datos

class MainController:
    def __init__(self):
        self.modelo = None
        self.tokenizer = None
        
    def inicializar_sistema(self):
        print("[Controlador] Cargando datos y cerebro de la IA...")
        # Cargamos el tokenizador usando tu función existente
        _, self.tokenizer = cargar_y_preprocesar_datos("data/raw/recetas_prueba.csv")
        
        # Cargamos el archivo del cerebro entrenado que guardaste antes
        # Nota: cambia a .keras si actualizaste la extensión
        self.modelo = tf.keras.models.load_model("models/saved_models/modelo_recetas.h5")
        print("[Controlador] Sistema listo para usar.")

    def generar_receta(self, ingredientes_usuario):
        """
        Recibe un texto (ej: "huevo leche"), lo procesa y predice la receta.
        """
        if not self.modelo or not self.tokenizer:
            return "El sistema no está inicializado."
            
        # 1. Convertir los ingredientes del usuario a números
        secuencia = self.tokenizer.texts_to_sequences([ingredientes_usuario])
        # Ajustamos el tamaño (padding) al largo que espera la red (ej: 5)
        entrada_lista = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=5, padding='post')
        
        # 2. Pedirle a la red neuronal que prediga la salida
        prediccion = self.modelo.predict(entrada_lista)
        palabra_predicha_id = np.argmax(prediccion, axis=-1)[0]
        
        # 3. Buscar la palabra real correspondiente al ID numérico
        receta_final = "Receta generada para: " + ingredientes_usuario
        # (Esto se expandirá más adelante cuando la red genere textos completos)
        
        return receta_final