"""
Vista de línea de comandos para interactuar con el generador de recetas.
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.main_controller import RecipeController

def clear_screen():
    """Limpia la pantalla de la terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Imprime el banner del programa."""
    print("="*60)
    print("🍳🍳🍳   RECIPE GEN AI - Generador de Recetas   🍳🍳🍳")
    print("="*60)
    print("   Una red neuronal LSTM que inventa recetas")
    print("   a partir de los ingredientes que tengas.")
    print("="*60)
    print()

def run_cli():
    """
    Ejecuta la interfaz de línea de comandos.
    """
    clear_screen()
    print_banner()
    
    # Inicializar controlador
    print("🔄 Cargando el modelo de IA... (esto puede tomar unos segundos)")
    controller = RecipeController()
    
    if controller.model is None:
        print("\n❌ No se pudo cargar el modelo.")
        print("   Asegúrate de haber ejecutado primero el entrenamiento:")
        print("   python -m src.model.train")
        return
    
    print("\n✅ ¡Modelo listo! El asistente culinario está funcionando.\n")
    
    while True:
        print("-"*60)
        print("🍽️ ¿Qué ingredientes tienes?")
        print("   (Ejemplo: huevos, queso, jamón, pan)")
        print("   (Escribe 'salir' para terminar)")
        print("-"*60)
        
        user_input = input("\n🔪 Ingredientes: ").strip()
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego! Buen provecho.")
            break
        
        if not user_input:
            print("⚠️ Por favor, ingresa al menos un ingrediente.")
            continue
        
        print("\n🤖 La IA está pensando en una receta... (puede tomar unos segundos)")
        print("-"*60)
        
        # Generar receta
        recipe = controller.generate_from_ingredients(user_input)
        
        # Formatear y mostrar
        formatted_recipe = controller.format_recipe(recipe)
        print("\n" + formatted_recipe)
        print("\n" + "="*60)
        print("💡 Nota: Las recetas son generadas por IA. ¡Usa tu criterio!")
        print("="*60)
        
        input("\nPresiona Enter para continuar...")
        print("\n"*2)

if __name__ == "__main__":
    run_cli()