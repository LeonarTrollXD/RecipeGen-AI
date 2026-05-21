from flask import Flask, render_template, request, jsonify
import sys
import os

# Agregar el directorio actual al path para importar inference
sys.path.append(os.path.dirname(__file__))

# Importar tus funciones directamente
from inference import inicializar_motor_vectorial, buscar_receta_optima

app = Flask(__name__)

# Variables globales para el motor
vectorizador = None
matriz_tfidf = None
lista_recetas = None
lista_titulos = None

def cargar_motor():
    """Carga el motor vectorial una sola vez al iniciar el servidor"""
    global vectorizador, matriz_tfidf, lista_recetas, lista_titulos
    
    try:
        print("🚀 Iniciando RecipeGen AI Web...")
        (
            vectorizador,
            matriz_tfidf,
            lista_recetas,
            lista_titulos
        ) = inicializar_motor_vectorial()
        
        print("✅ Motor cargado exitosamente")
        print("🌐 Servidor web listo en http://localhost:5000")
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar el motor: {e}")
        return False

@app.route('/')
def home():
    """Página principal"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para buscar recetas"""
    try:
        # Verificar que el motor esté cargado
        if vectorizador is None:
            return jsonify({
                'success': False,
                'error': 'El motor de búsqueda no está inicializado'
            }), 500
        
        # Obtener ingredientes del request
        data = request.get_json()
        ingredientes = data.get('ingredients', '').strip()
        
        if not ingredientes:
            return jsonify({
                'success': False,
                'error': 'Por favor, ingresa algunos ingredientes'
            }), 400
        
        print(f"\n🔍 Buscando receta para: {ingredientes}")
        
        # Usar tu función original de búsqueda
        (
            receta,
            porcentaje,
            titulo
        ) = buscar_receta_optima(
            ingredientes,
            vectorizador,
            matriz_tfidf,
            lista_recetas,
            lista_titulos
        )
        
        # Verificar si encontró algo
        if porcentaje < 5.0:
            return jsonify({
                'success': False,
                'error': 'No se encontraron recetas compatibles con esos ingredientes 😕',
                'match_percentage': 0
            }), 404
        
        # Devolver resultado exitoso
        print(f"✅ Receta encontrada: {titulo} ({porcentaje:.2f}%)")
        
        return jsonify({
            'success': True,
            'recipe': titulo,
            'ingredients': ingredientes,
            'match_percentage': round(porcentaje, 2),
            'instructions': receta
        })
        
    except Exception as e:
        print(f"❌ Error en predict: {e}")
        return jsonify({
            'success': False,
            'error': f'Error al procesar la búsqueda: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Endpoint para verificar que el servidor está funcionando"""
    motor_ok = vectorizador is not None
    return jsonify({
        'status': 'ok' if motor_ok else 'error',
        'motor_cargado': motor_ok,
        'recetas_disponibles': len(lista_titulos) if motor_ok else 0
    })

if __name__ == '__main__':
    # Cargar el motor al iniciar
    if cargar_motor():
        # Iniciar servidor Flask
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000
        )
    else:
        print("❌ No se pudo iniciar el servidor por error en el motor")