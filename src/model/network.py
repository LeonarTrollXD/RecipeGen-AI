"""
Módulo de la arquitectura de la red neuronal LSTM.
Define el modelo, sus capas y la configuración.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import tensorflow as tf

class RecipeLSTM:
    """
    Clase que define la arquitectura LSTM para generación de recetas.
    """
    
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024, batch_size=64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.model = None
    
    def build_model(self):
        """Construye la arquitectura del modelo LSTM."""
        self.model = Sequential([
            Input(shape=(None,)),
            Embedding(self.vocab_size, self.embedding_dim),
            LSTM(self.rnn_units, return_sequences=True, dropout=0.2),
            LSTM(self.rnn_units, return_sequences=True, dropout=0.2),
            Dense(self.vocab_size)
        ])
        
        print("🏗️ Modelo LSTM construido:")
        self.model.summary()
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compila el modelo."""
        self.model.compile(
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        print("✅ Modelo compilado")
        return self.model
    
    def get_checkpoint_callback(self, checkpoint_dir='../models/saved_models/checkpoints'):
        """Crea el callback para guardar checkpoints."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
        
        return ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True,
            save_best_only=True,
            monitor='loss'
        )
    
    def get_early_stopping(self, patience=3):
        """Crea el callback para detener entrenamiento si no mejora."""
        return EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    
    def train(self, dataset, epochs=20, checkpoint_dir='../models/saved_models/checkpoints'):
        """Entrena el modelo."""
        checkpoint_callback = self.get_checkpoint_callback(checkpoint_dir)
        early_stopping = self.get_early_stopping(patience=5)
        
        print(f"🚀 Iniciando entrenamiento por {epochs} épocas...")
        history = self.model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback, early_stopping])
        print("✅ Entrenamiento completado")
        return history
    
    def save_model(self, filepath='../models/saved_models/recipe_model.h5'):
        """Guarda el modelo entrenado."""
        self.model.save(filepath)
        print(f"💾 Modelo guardado en: {filepath}")
    
    def load_model(self, filepath='../models/saved_models/recipe_model.h5'):
        """Carga un modelo guardado."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"📂 Modelo cargado desde: {filepath}")
        return self.model