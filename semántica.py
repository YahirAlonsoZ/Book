import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import spacy

# 1. Cargar el modelo de spaCy
nlp = spacy.load("es_core_news_sm")

# 2. Tu texto lematizado
texto_procesado = "pequeño zorro vivir cerca bosque mañana salir caminar árbol saludar animal encontrar viejo mapa señalar camino lago escondido curiosidad decidir seguir pista caminar hora lago agua transparente poder pez nadar piedra zorro comprender tesoro mapa experiencia descubrir lugar aprender naturaleza regresar casa contar aventura amigo decidir explorar bosque respeto cuidar árbol río animal encontrar camino"

# 3. Obtener palabras únicas y sus vectores con spaCy
palabras_unicas = list(set(texto_procesado.split()))
vectores = [nlp(p).vector for p in palabras_unicas]

# 4. Reducir a 2D con PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(vectores)

# 5. Graficar y guardar como .png
plt.figure(figsize=(10, 7))
plt.scatter(coords[:, 0], coords[:, 1], c="purple", alpha=0.7)

for i, palabra in enumerate(palabras_unicas):
  plt.annotate(palabra, xy=(coords[i, 0], coords[i, 1]), fontsize=9)

plt.title("Espacio Vectorial Semántica Distribucional (spaCy Embeddings)")
plt.grid(True)

# Guardar la imagen para el repositorio de GitHub
plt.savefig("espacio_distribucional.png", dpi=300, bbox_inches="tight")
plt.show()

print("¡Imagen 'espacio_distribucional.png' guardada con éxito!")