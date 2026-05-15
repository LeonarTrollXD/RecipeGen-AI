import streamlit as st
import ollama
from dotenv import load_dotenv, find_dotenv
from fpdf import FPDF
from PIL import Image
import os
import time
import requests
import tempfile # Import tempfile for temporary files
from io import BytesIO

load_dotenv(find_dotenv(), override=True)

# Inicializar el historial en la sesión si no existe
if 'historial' not in st.session_state:
    st.session_state.historial = []

# @st.cache_data(show_spinner="Cocinando tu receta con IA...") # Deshabilitado temporalmente para depuración
def generate_recipe(ingredients):
    system_prompt = '''
    Eres un chef de cocina internacional de primera clase. Tu tarea es redactar recetas creativas, deliciosas y detalladas estrictamente en español.
    '''
    user_prompt = f'''
    Crea una receta detallada basada únicamente en los siguientes ingredientes: {', '.join(ingredients)}.
    IMPORTANTE: No añadas ingredientes básicos que no estén en la lista (como huevos, harina o leche). 
    Si no hay huevos, no hagas una tortilla. Si no hay harina, no hagas pan. Crea un plato lógico (guiso, salteado, sopa, etc.) con lo que tienes.
    
    Por favor, formatea la receta de la siguiente manera: 

    Título de la receta:

    Ingredientes de la Receta con tamaño y porcion:

    lista de Instrucciones para esta receta:
    '''

    with st.status("procesando la receta", expanded=True) as status:
        try:
            # Obtenemos la lista de modelos que realmente tienes instalados
            models_info = ollama.list()
            installed_models = []
            
            models_list = getattr(models_info, 'models', models_info.get('models', []))
            for m in models_list:
                name = getattr(m, 'model', m.get('model') or m.get('name'))
                if name:
                    installed_models.append(name)
            
            if not installed_models:
                st.error("❌ No se encontraron modelos en Ollama. Abre una terminal y ejecuta: `ollama pull llama3` o `ollama pull mistral`")
                st.stop()
                
            prioridad = ["llama3:latest", "llama3", "mistral:latest", "mistral", "gemma:latest", "gemma", "phi3:latest", "phi3"]
            modelos_a_probar = [m for m in prioridad if m in installed_models]
            if not modelos_a_probar:
                modelos_a_probar = [installed_models[0]] # Asegurarse de que haya al menos un modelo

            for model_name in modelos_a_probar:
                try:
                    response = ollama.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={"temperature": 0.7}
                    )
                    if response and 'message' in response and response['message']['content']:
                        status.update(label="✅ Receta generada con éxito", state="complete", expanded=False)
                        return response['message']['content'].strip()
                except Exception:
                    continue # Intenta con el siguiente modelo en silencio

        except Exception as e:
            st.error(f"❌ No se pudo conectar con Ollama. ¿Está el programa abierto? Error: {e}")
            st.stop()

    st.error("❌ Todos los modelos instalados fallaron al generar la receta.")
    st.stop()

def obtener_nombre_receta(texto):
    lines = texto.splitlines()
    # 1. Intentar buscar una línea que explícitamente diga Título o Nombre
    for line in lines:
        if ":" in line:
            header, content = line.split(":", 1)
            if any(k in header.lower() for k in ["título", "titulo", "nombre"]):
                res = content.replace('*', '').replace('#', '').strip()
                if res: return res
    
    # 2. Si no hay encabezado, tomar la primera línea que no esté vacía
    for line in lines:
        clean = line.replace('*', '').replace('#', '').strip()
        if clean: return clean
        
    return "Receta_Sin_Nombre"

def generar_imagen(titulo_receta, receta_completa):
    # Extraer ingredientes de la receta completa para un prompt más detallado
    extracted_ingredients_for_image = ""
    ingredients_section_start = receta_completa.find("Ingredientes de la Receta con tamaño y porcion:")
    instructions_section_start = receta_completa.find("lista de Instrucciones para esta receta:")

    if ingredients_section_start != -1 and instructions_section_start != -1:
        # Obtener solo el texto de los ingredientes
        ingredients_text_raw = receta_completa[ingredients_section_start + len("Ingredientes de la Receta con tamaño y porcion:"):instructions_section_start].strip()
        
        # Limpiar y formatear los ingredientes para el prompt de la IA
        # Tomamos las primeras 5 líneas de ingredientes para no sobrecargar el prompt
        cleaned_ingredients = [line.strip().replace('-', '').replace('*', '') for line in ingredients_text_raw.split('\n') if line.strip()]
        extracted_ingredients_for_image = ", ".join(cleaned_ingredients[:5])
        if extracted_ingredients_for_image:
            extracted_ingredients_for_image = f" featuring {extracted_ingredients_for_image}"

    # Usamos Inteligencia Artificial para generar una imagen específica de la receta
    prompt_ia = (
        f"A photorealistic image of the dish '{titulo_receta}'{extracted_ingredients_for_image}. "
        f"The food is beautifully presented on a ceramic plate, close-up focus on textures and vibrant colors. "
        f"Setting: rustic wooden table, natural sunlight, gourmet style. "
        f"Intricate details, looks freshly prepared and delicious, 8k resolution."
    )
    prompt_encoded = requests.utils.quote(prompt_ia)
    url = f"https://image.pollinations.ai/prompt/{prompt_encoded}?width=800&height=450&nologo=true&seed={int(time.time())}"
    
    try:
        response = requests.get(url, timeout=5)
        return Image.open(BytesIO(response.content))
    except:
        return Image.new('RGB', (800, 450), color = (240, 240, 240))

def save_to_pdf(titulo_receta, receta, imagen_pil):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Agregar título
    pdf.set_font("Arial", 'B', 16)
    # Sanitizar el título para evitar errores de caracteres especiales en PDF
    clean_titulo = titulo_receta.encode('latin-1', 'replace').decode('latin-1')
    pdf.cell(0, 10, clean_titulo, ln=True, align='C')

    # Guardar imagen temporalmente para FPDF
    # Usamos tempfile para asegurar que el archivo se limpia
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
        imagen_pil.save(temp_img_file.name, format="JPEG")
        img_path = temp_img_file.name

    pdf.ln(10)  # Agregar espacio después del título
    img_width = 150
    pdf.image(img_path, x=(pdf.w - img_width) / 2, w=img_width, type='JPEG')
    pdf.ln(10)  # Agregar espacio después de la imagen

    # Limpiar el archivo de imagen temporal inmediatamente después de usarlo
    os.remove(img_path)

    # Agregar cuerpo de la receta
    pdf.set_font("Arial", size=12)
    for line in receta.split('\n'):
        # FPDF no soporta caracteres especiales de forma nativa sin fuentes externas
        # Sanitizamos el texto para evitar errores de codificación
        clean_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, clean_line) # Reducido el alto de la celda para mejor formato

    # Retornar el contenido del PDF como bytes
    return pdf.output(dest='S').encode('latin-1', 'replace')

# Callback para eliminar una receta del historial
def delete_recipe_from_history(index):
    if 'historial' in st.session_state and 0 <= index < len(st.session_state.historial):
        # Si la receta eliminada es la que se está mostrando, limpiar la visualización
        if 'titulo_receta' in st.session_state and st.session_state.historial[index]['titulo'] == st.session_state.titulo_receta:
            del st.session_state.receta
            del st.session_state.titulo_receta
            del st.session_state.imagen_receta
            if 'pdf_bytes' in st.session_state: del st.session_state.pdf_bytes
        
        del st.session_state.historial[index]
        st.rerun() # Recargar para actualizar la interfaz

st.title("Generador de Recetas")

# --- Barra Lateral para Historial ---
st.sidebar.title("📚 Historial de Recetas")
if not st.session_state.historial:
    st.sidebar.write("Aún no has generado recetas.")
else:
    # Mostrar las recetas en orden inverso (las más nuevas primero)
    for i in range(len(st.session_state.historial) - 1, -1, -1):
        item = st.session_state.historial[i]
        col1, col2 = st.sidebar.columns([4, 1]) # Dividir el espacio para el botón de título y el de eliminar
        with col1:
            if st.button(f"🍴 {item['titulo']}", key=f"hist_select_{i}"):
                st.session_state.receta = item['texto']
                st.session_state.titulo_receta = item['titulo']
                st.session_state.imagen_receta = item['imagen']
                if 'pdf_bytes' in st.session_state: del st.session_state.pdf_bytes
        with col2:
            if st.button("🗑️", key=f"hist_delete_{i}", help="Eliminar esta receta"):
                delete_recipe_from_history(i)

# --- Interfaz Principal ---
st.write("Escribe los ingredientes que tienes a mano para crear una receta única:")
ingredientes = st.text_input("Ingredientes (separados por comas)")

if st.button("Generar Receta"):
    if ingredientes:
        ingredients_list = [ing.strip() for ing in ingredientes.split(',')]
        receta = generate_recipe(ingredients_list)
        titulo_receta = obtener_nombre_receta(receta)
        imagen_receta = generar_imagen(titulo_receta, receta) # Pasar la receta completa aquí
        
        # Guardar en el estado actual
        st.session_state.receta = receta
        st.session_state.titulo_receta = titulo_receta
        st.session_state.imagen_receta = imagen_receta
        if 'pdf_bytes' in st.session_state: del st.session_state.pdf_bytes
        
        # Guardar en el historial
        nueva_entrada = {
            "titulo": titulo_receta,
            "texto": receta,
            "imagen": imagen_receta
        }
        st.session_state.historial.append(nueva_entrada)
        st.rerun() # Recargar para que aparezca en el sidebar inmediatamente
    else:
        st.warning("Por favor, introduce algunos ingredientes.")

# Lógica de visualización fuera del botón para evitar que desaparezca al interactuar
if 'receta' in st.session_state:
    st.divider()
    st.subheader(st.session_state.titulo_receta)
    
    # Opción para "Compartir" (copiar al portapapeles o mostrar link)
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔗 Copiar Receta"):
            st.toast("Receta lista para compartir (puedes copiar el texto de abajo)")

    st.markdown(st.session_state.receta) 
    st.image(st.session_state.imagen_receta, caption=st.session_state.titulo_receta)
    
    # Generar PDF automáticamente para el botón de descarga
    if 'pdf_bytes' not in st.session_state:
        st.session_state.pdf_bytes = save_to_pdf(st.session_state.titulo_receta, st.session_state.receta, st.session_state.imagen_receta)

    # Limpiar el nombre de archivo de caracteres prohibidos y espacios
    nombre_archivo = "".join([c for c in st.session_state.titulo_receta if c.isalnum() or c == ' ']).strip().replace(' ', '_')

    st.download_button(
        label="📥 Descargar Receta en PDF",
        data=st.session_state.pdf_bytes,
        file_name=f"{nombre_archivo}.pdf",
        mime="application/pdf"
    )