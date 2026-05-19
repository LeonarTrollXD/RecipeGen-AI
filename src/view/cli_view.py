"""
Interfaz de línea de comandos para el generador de recetas.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controller.main_controller import RecipeController

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    print("="*60)
    print("🍳🍳🍳   RECIPE GEN AI - Generador de Recetas   🍳🍳🍳")
    print("="*60)
    print("   Una red neuronal LSTM que inventa recetas")
    print("   a partir de los ingredientes que tengas.")
    print("="*60)
    print()

def run_cli():
    clear_screen()
    print_banner()
    
    print("🔄 Cargando el modelo de IA...\n")
    controller = RecipeController()
    
    if not controller.model_loaded:
        print("\n❌ No se pudo cargar el modelo.")
        print("   Asegúrate de haber ejecutado primero el entrenamiento:")
        print("   python -m src.model.train")
        return
    
    print("\n✅ ¡Modelo listo! El asistente culinario está funcionando.\n")
    
    while True:
        print("-"*60)
        print("🍽️ ¿Qué ingredientes tienes?")
        print("   (Ejemplo: huevos, queso, jamon, pan)")
        print("   (Escribe 'salir' para terminar)")
        print("-"*60)
        
        user_input = input("\n🔪 Ingredientes: ").strip()
        
        if user_input.lower() in ['salir', 'exit', 'quit']:
            print("\n👋 ¡Hasta luego! Buen provecho.")
            break
        
        if not user_input:
            print("⚠️ Por favor, ingresa al menos un ingrediente.")
            continue
        
        print("\n🤖 La IA está pensando en una receta...\n")
        print("-"*60)
        
        # Generar receta
        recipe = controller.generate_from_ingredients(user_input)
        
        # Mostrar resultado
        print("\n" + recipe)
        print("\n" + "="*60)
        print("💡 Nota: Las recetas son generadas por IA. ¡Usa tu criterio!")
        print("="*60)
        
        input("\nPresiona Enter para continuar...")
        print("\n"*2)

if __name__ == "__main__":
    run_cli()