from src.controller.main_controller import MainController

def main():
    print("=========================================")
    print("   INICIANDO RECIPEGEN-AI (SISTEMA MVC)  ")
    print("=========================================\n")
    
    # 1. Instanciar el controlador
    controlador = MainController()
    
    # 2. Inicializar el modelo y los datos
    controlador.inicializar_sistema()
    
    # 3. Simular una interacción del usuario (Capa Vista temporal)
    ingredientes_prueba = "huevo leche harina"
    print(f"\n[Vista] Usuario ingresa ingredientes: '{ingredientes_prueba}'")
    
    # 4. El controlador procesa y pide la predicción a TensorFlow
    resultado = controlador.generar_receta(ingredientes_prueba)
    
    print("\n[Vista] Resultado recibido del Controlador:")
    print(f"--> {resultado}")
    print("\n=========================================")

if __name__ == "__main__":
    main()