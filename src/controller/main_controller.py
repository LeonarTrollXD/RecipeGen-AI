"""
Controlador principal - Orquesta la vista con el modelo.
Implementa la lógica de negocio del generador de recetas.
"""

import sys
import os
import numpy as np
import tensorflow as tf

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.preprocess import DataPreprocessor
from src.model.network import RecipeLSTM

class RecipeController:
    """
    Controlador que maneja la lógica de generación de recetas.
    Conecta el modelo entrenado con la interfaz de usuario.
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.tokenizer_data = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
        
        # Cargar modelo y tokenizador al iniciar
        self.load_model()
        self.load_tokenizer()
    
    def load_model(self, model_path='../models/saved_models/recipe_model.h5'):
        """
        Carga el modelo entrenado desde el archivo.
        """
        try:
            # Crear modelo para inferencia (batch_size=1)
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Modelo cargado correctamente")
            return True
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            print("   Asegúrate de haber ejecutado train.py primero")
            return False
    
    def load_tokenizer(self, tokenizer_path='../models/saved_models/tokenizer.pkl'):
        """
        Carga el tokenizador (vocabulario) desde el archivo.
        """
        try:
            self.char_to_idx, self.idx_to_char, self.vocab_size = self.preprocessor.load_tokenizer(tokenizer_path)
            return True
        except Exception as e:
            print(f"❌ Error al cargar el tokenizador: {e}")
            return False
    
    def generate_recipe(self, start_string="[NAME] ", num_generate=1000, temperature=0.7):
        """
        Genera una receta a partir de un texto inicial.
        
        Args:
            start_string: Texto inicial para comenzar la generación
            num_generate: Número de caracteres a generar
            temperature: Controla la creatividad (menos = más predecible, más = más creativo)
        
        Returns:
            str: Receta generada
        """
        if self.model is None:
            return "❌ Error: Modelo no cargado. Entrena primero el modelo."
        
        # Convertir el texto de inicio a índices
        input_eval = [self.char_to_idx.get(s, 0) for s in start_string]  # 0 para caracteres desconocidos
        input_eval = tf.expand_dims(input_eval, 0)
        
        # Lista para guardar los caracteres generados
        text_generated = []
        
        # Resetear el estado del modelo
        self.model.reset_states()
        
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            
            # Aplicar temperatura para controlar la creatividad
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            
            # Pasar el caracter predicho como la siguiente entrada
            input_eval = tf.expand_dims([predicted_id], 0)
            
            # Obtener el caracter del índice
            char_generated = self.idx_to_char.get(predicted_id, '?')
            text_generated.append(char_generated)
        
        return start_string + ''.join(text_generated)
    
    def generate_from_ingredients(self, ingredients_list):
        """
        Genera una receta a partir de una lista de ingredientes.
        
        Args:
            ingredients_list: Lista de ingredientes o string separado por comas
        
        Returns:
            str: Receta generada
        """
        # Procesar entrada
        if isinstance(ingredients_list, list):
            ingredients_str = ' '.join(ingredients_list)
        else:
            ingredients_str = str(ingredients_list)
        
        # Construir el prompt para la red
        prompt = f"[NAME] [INGREDIENTS] {ingredients_str} [INSTRUCTIONS] "
        
        # Generar receta
        recipe = self.generate_recipe(start_string=prompt, num_generate=1500, temperature=0.7)
        
        return recipe
    
    def format_recipe(self, raw_recipe):
        """
        Formatea la receta generada para mejor visualización.
        """
        lines = raw_recipe.split('[INSTRUCTIONS]')
        
        if len(lines) >= 2:
            title_ingredients = lines[0].replace('[NAME]', '📗 NOMBRE:').replace('[INGREDIENTS]', '\n\n🥕 INGREDIENTES:')
            instructions = f"\n\n📝 INSTRUCCIONES:\n{lines[1]}"
            return title_ingredients + instructions
        else:
            return raw_recipe