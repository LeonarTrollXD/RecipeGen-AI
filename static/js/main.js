// Agregar ingredientes rápidos de ejemplo
const ejemplosIngredientes = [
    "pasta, tomate, albahaca, ajo",
    "pollo, cebolla, ajo, pimentón",
    "arroz, huevo, zanahoria, cebolla",
    "harina, huevo, leche, mantequilla"
];

// Crear botones de ingredientes rápidos
function crearBotonesRapidos() {
    const searchBox = document.querySelector('.search-box');
    const quickDiv = document.createElement('div');
    quickDiv.className = 'quick-ingredients';
    quickDiv.innerHTML = '<small style="width:100%;color:#999;">Prueba con:</small>';
    
    ejemplosIngredientes.forEach(ing => {
        const btn = document.createElement('button');
        btn.className = 'quick-ingredient-btn';
        btn.textContent = ing;
        btn.onclick = () => {
            document.getElementById('ingredients').value = ing;
            getRecipe();
        };
        quickDiv.appendChild(btn);
    });
    
    searchBox.appendChild(quickDiv);
}

async function getRecipe() {
    const ingredients = document.getElementById('ingredients').value.trim();
    
    if (!ingredients) {
        showError('Por favor, ingresa algunos ingredientes 🥘');
        return;
    }

    // Mostrar loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('search-btn').disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ingredients: ingredients })
        });

        const data = await response.json();

        // Ocultar loading
        document.getElementById('loading').style.display = 'none';
        document.getElementById('search-btn').disabled = false;

        if (!response.ok) {
            throw new Error(data.error || 'Error al buscar la receta');
        }

        if (data.success) {
            mostrarResultado(data);
        } else {
            showError(data.error || 'No se encontraron recetas');
        }
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('search-btn').disabled = false;
        showError(error.message || 'Error de conexión');
        console.error('Error:', error);
    }
}

function mostrarResultado(data) {
    // Actualizar contenido
    document.getElementById('recipe-name').textContent = data.recipe;
    
    // Formatear porcentaje con color
    const matchBadge = document.getElementById('match-percentage');
    matchBadge.textContent = `🎯 ${data.match_percentage}% coincidencia`;
    
    // Cambiar color según porcentaje
    matchBadge.className = 'match-badge';
    if (data.match_percentage > 70) {
        matchBadge.classList.add('high');
    } else if (data.match_percentage > 30) {
        matchBadge.classList.add('medium');
    } else {
        matchBadge.classList.add('low');
    }
    
    document.getElementById('ingredients-list').textContent = data.ingredients;
    
    // Formatear instrucciones con números
    const instructions = data.instructions
        .split('.')
        .filter(i => i.trim())
        .map((step, index) => `<p><strong>${index + 1}.</strong> ${step.trim()}.</p>`)
        .join('');
    
    document.getElementById('instructions-text').innerHTML = instructions;
    
    // Mostrar resultado con animación
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.style.animation = 'none';
    resultDiv.offsetHeight; // Trigger reflow
    resultDiv.style.animation = 'slideIn 0.5s ease-out';
    
    // Scroll suave
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.innerHTML = `⚠️ ${message}`;
    errorDiv.style.display = 'block';
    errorDiv.style.animation = 'none';
    errorDiv.offsetHeight;
    errorDiv.style.animation = 'slideIn 0.3s ease-out';
    
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 6000);
}

// Función para copiar receta
function copiarReceta() {
    const receta = document.getElementById('instructions-text').innerText;
    navigator.clipboard.writeText(receta).then(() => {
        alert('✅ Receta copiada al portapapeles');
    });
}

// Inicializar al cargar la página
document.addEventListener('DOMContentLoaded', () => {
    crearBotonesRapidos();
    
    // Permitir búsqueda con Enter
    document.getElementById('ingredients').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getRecipe();
        }
    });
});