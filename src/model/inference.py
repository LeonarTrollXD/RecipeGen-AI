import numpy as np
import tensorflow as tf
from src.model.preprocess import cargar_y_preprocesar_datos

def generar_instruccion_completa(ingredientes_input, max_palabras=20):
    # 1. Cargar modelo y tokenizer
    modelo = tf.keras.models.load_model("models/saved_models/modelo_recetas.keras")
    _, tokenizer, _ = cargar_y_preprocesar_datos()
    
    # Invertimos el diccionario del tokenizer una sola vez antes del bucle (optimiza memoria)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    
    # Esta variable acumulará todo el texto generado paso a paso
    texto_acumulado = ingredientes_input
    
    print("\n[IA] Pensando y redactando la sugerencia...")
    
    for i in range(max_palabras):
        # 2. Procesar el texto acumulado actual
        secuencia = tokenizer.texts_to_sequences([texto_acumulado])
        entrada = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=15, padding='post')
        
        # 3. Predicción (verbose=0 para que no ensucie la pantalla con barras de carga)
        prediccion = modelo.predict(entrada, verbose=0)[0]
        temperatura = 0.7 
        prediccion = np.log(prediccion + 1e-7) / temperatura
        exp_preds = np.exp(prediccion)
        prediccion = exp_preds / np.sum(exp_preds)
        # 4. Encontrar el índice de la palabra más probable
        index_palabra = np.random.choice(len(prediccion), p=prediccion)
        
        # Si el modelo predice 0 (es el token de relleno/padding), terminamos la frase
        if index_palabra == 0:
            break
            
        # Traducir el número a palabra real
        palabra_sugerida = reverse_word_map.get(index_palabra, "")
        
        # Si encuentra un token vacío o desconocido, detenemos la generación
        if not palabra_sugerida or palabra_sugerida == "<OOV>":
            break
            
        # 5. Enganchar la nueva palabra al texto que ya teníamos
        texto_acumulado += " " + palabra_sugerida

    return texto_acumulado

if __name__ == "__main__":
    # Puedes cambiar esta frase por los ingredientes que quieras probar
    ingredientes_prueba = "tomates atún" 
    
    resultado_final = generar_instruccion_completa(ingredientes_prueba, max_palabras=15)
    
    print("\n" + "="*50)
    print(f"Ingredientes iniciales : {ingredientes_prueba}")
    print(f"Receta generada por IA : {resultado_final}")
    print("="*50 + "\n")