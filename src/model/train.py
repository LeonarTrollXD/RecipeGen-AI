"""
Módulo de entrenamiento del modelo.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.preprocess import DataPreprocessor
from src.model.network import RecipeLSTM

def run_training():
    print("="*60)
    print("🍳 RECIPE GEN AI - ENTRENAMIENTO 🍳")
    print("="*60)
    
    print("\n📊 PASO 1: Preprocesamiento")
    preprocessor = DataPreprocessor(seq_length=100)
    
    recipes_text = preprocessor.load_recipes()
    all_text = ' '.join(recipes_text)
    
    char_to_idx, idx_to_char, vocab_size = preprocessor.create_vocabulary(all_text)
    text_as_int = preprocessor.text_to_sequences(all_text)
    dataset, batch_size = preprocessor.create_training_sequences(text_as_int)
    preprocessor.save_tokenizer()
    
    print("\n🧠 PASO 2: Entrenamiento")
    model_builder = RecipeLSTM(
        vocab_size=vocab_size,
        embedding_dim=256,
        rnn_units=1024,
        batch_size=batch_size
    )
    
    model_builder.build_model()
    model_builder.compile_model()
    history = model_builder.train(dataset, epochs=10)
    model_builder.save_model()
    
    print("\n✅ ENTRENAMIENTO COMPLETADO")

if __name__ == "__main__":
    run_training()