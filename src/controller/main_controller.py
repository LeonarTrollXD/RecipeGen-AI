"""
Controlador principal - Orquesta la vista con el modelo.
"""

import sys
import os
import numpy as np
import tensorflow as tf
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class RecipeController:
    """
    Controlador que maneja la lógica de generación de recetas.
    """
    
    def __init__(self):
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
        self.model_loaded = False
        
        # Intentar cargar modelo y tokenizador
        self.load_model()
        self.load_tokenizer()
        
        if self.model and self.char_to_idx:
            self.model_loaded = True
    
    def load_model(self, model_path='models/saved_models/recipe_model.h5'):
        """Carga el modelo entrenado."""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("✅ Modelo cargado")
                return True
            else:
                print(f"⚠️ Modelo no encontrado en {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def load_tokenizer(self, tokenizer_path='models/saved_models/tokenizer.pkl'):
        """Carga el tokenizador."""
        try:
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    data = pickle.load(f)
                self.char_to_idx = data['char_to_idx']
                self.idx_to_char = data['idx_to_char']
                self.vocab_size = data['vocab_size']
                print("✅ Tokenizador cargado")
                return True
            else:
                print(f"⚠️ Tokenizador no encontrado en {tokenizer_path}")
                return False
        except Exception as e:
            print(f"❌ Error cargando tokenizador: {e}")
            return False
    
    def generate_from_ingredients(self, ingredients_text, num_generate=800, temperature=0.7):
        """
        Genera una receta a partir de una lista de ingredientes.
        ingredients_text: texto que el usuario escribe (ej: "huevos, queso, jamon")
        """
        if not self.model_loaded:
            return "❌ Modelo no disponible. Entrena primero el modelo."
        
        # Usar LOS INGREDIENTES QUE ESCRIBIÓ EL USUARIO
        prompt = f"[NAME] Receta con {ingredients_text} [INGREDIENTS] {ingredients_text} [INSTRUCTIONS] "
        
        # Convertir prompt a índices
        input_indices = [self.char_to_idx.get(c, 0) for c in prompt]
        input_tensor = tf.expand_dims(input_indices, 0)
        
        # Generar caracteres
        generated = []
        self.model.reset_states()
        
        for _ in range(num_generate):
            predictions = self.model(input_tensor)
            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            
            input_tensor = tf.expand_dims([predicted_id], 0)
            generated.append(self.idx_to_char.get(predicted_id, '?'))
        
        # Unir resultado
        result = prompt + ''.join(generated)
        
        # Limpiar un poco el resultado
        result = result.replace('[NAME]', '📗 NOMBRE:')
        result = result.replace('[INGREDIENTS]', '\n\n🥕 INGREDIENTES:')
        result = result.replace('[INSTRUCTIONS]', '\n\n📝 INSTRUCCIONES:')
        
        return result