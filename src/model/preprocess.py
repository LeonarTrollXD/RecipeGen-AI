"""
Módulo de preprocesamiento de datos para el generador de recetas.
"""

import pathlib
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

class DataPreprocessor:
    
    def __init__(self, cache_dir='./tmp', seq_length=100):
        self.cache_dir = cache_dir
        self.seq_length = seq_length
        self.vocab_size = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        
        pathlib.Path(cache_dir).mkdir(exist_ok=True)
        pathlib.Path('../data/processed').mkdir(parents=True, exist_ok=True)
        pathlib.Path('../models/saved_models').mkdir(parents=True, exist_ok=True)
    
    def download_and_load_dataset(self):
        """Descarga dataset de recetas."""
        url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-09-16/all_recipes.csv"
        
        print(f"📥 Descargando dataset...")
        df = pd.read_csv(url)
        print(f"✅ {len(df)} recetas cargadas")
        return df
    
    def load_recipes(self):
        """Carga y procesa las recetas."""
        df = self.download_and_load_dataset()
        recipes_text = []
        
        for idx, row in df.iterrows():
            title = str(row.get('name', ''))
            ingredients = str(row.get('ingredients', ''))
            
            if not title or len(title) < 3:
                continue
            if not ingredients or len(ingredients) < 5:
                continue
            
            ingredients = ingredients.replace('\n', ' ').replace('\r', ' ').strip()
            title = title.replace('\n', ' ').strip()
            
            # El dataset no tiene instrucciones, usamos los ingredientes como instrucciones temporales
            full_recipe = f"[NAME] {title} [INGREDIENTS] {ingredients} [INSTRUCTIONS] {ingredients}"
            recipes_text.append(full_recipe)
            
            if len(recipes_text) % 2000 == 0:
                print(f"   Procesadas: {len(recipes_text)} recetas")
        
        print(f"✅ Recetas procesadas: {len(recipes_text)}")
        
        if len(recipes_text) == 0:
            print("⚠️ Usando recetas de ejemplo...")
            recipes_text = [
                "[NAME] Huevos Revueltos [INGREDIENTS] huevos sal aceite [INSTRUCTIONS] Calentar aceite, agregar huevos batidos, cocinar",
                "[NAME] Pasta con Queso [INGREDIENTS] pasta queso mantequilla [INSTRUCTIONS] Cocer pasta, agregar queso y mantequilla",
                "[NAME] Ensalada [INGREDIENTS] lechuga tomate aceite [INSTRUCTIONS] Lavar vegetales, mezclar con aceite",
            ]
        
        return recipes_text
    
    def process_recipes_to_text(self, dataset_raw):
        return dataset_raw
    
    def create_vocabulary(self, all_text):
        if not all_text:
            raise ValueError("No hay texto para crear vocabulario")
        
        chars = sorted(list(set(all_text)))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"🔤 Vocabulario: {self.vocab_size} caracteres")
        return self.char_to_idx, self.idx_to_char, self.vocab_size
    
    def text_to_sequences(self, all_text):
        return np.array([self.char_to_idx.get(c, 0) for c in all_text])
    
    def create_training_sequences(self, text_as_int):
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(self.seq_length + 1, drop_remainder=True)
        
        def split_input_target(chunk):
            return chunk[:-1], chunk[1:]
        
        dataset = sequences.map(split_input_target)
        BATCH_SIZE = 64
        dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
        
        print(f"✅ Dataset listo: secuencia={self.seq_length}, batch={BATCH_SIZE}")
        return dataset, BATCH_SIZE
    
    def save_tokenizer(self, filepath='../models/saved_models/tokenizer.pkl'):
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Tokenizador guardado")
    
    def load_tokenizer(self, filepath='../models/saved_models/tokenizer.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']
        print(f"📂 Tokenizador cargado")
        return self.char_to_idx, self.idx_to_char, self.vocab_size