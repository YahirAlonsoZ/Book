import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
import pandas as pd

# 1. Tu texto procesado (lemas)
texto_procesado = "pequeño zorro vivir cerca bosque mañana salir caminar árbol saludar animal encontrar viejo mapa señalar camino lago escondido curiosidad decidir seguir pista caminar hora lago agua transparente poder pez nadar piedra zorro comprender tesoro mapa experiencia descubrir lugar aprender naturaleza regresar casa contar aventura amigo decidir explorar bosque respeto cuidar árbol río animal encontrar camino"

# Dividimos el texto en "oraciones" o bloques para crear el corpus (por ejemplo, dividiendo por la palabra 'bosque' o creando frases)
# Si tu libro procesado es un solo string, creamos oraciones/frases separándolo para poder comparar bloques dentro del texto:
corpus_lematizado = [
    "pequeño zorro vivir cerca bosque mañana salir caminar árbol saludar animal",
    "encontrar viejo mapa señalar camino lago escondido curiosidad decidir seguir pista",
    "caminar hora lago agua transparente poder pez nadar piedra zorro comprender tesoro mapa",
    "experiencia descubrir lugar aprender naturaleza regresar casa contar aventura amigo",
    "decidir explorar bosque respeto cuidar árbol río animal encontrar camino"
]

# --- A. BAG OF WORDS (BoW) ---
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
vocab_bow = bow_vectorizer.get_feature_names_out()

# --- B. TF-IDF ---
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)
vocab_tfidf = tfidf_vectorizer.get_feature_names_out()

# Mostrar las primeras filas de la matriz TF-IDF en un DataFrame
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vocab_tfidf)
print("Vista previa de la Matriz TF-IDF:")
print(df_tfidf.iloc[:, :8])  # Muestra las primeras 8 palabras


# ---------------------------------------------------------
# FUNCIÓN PARA GRAFICAR PALABRAS EN 3D (PCA)
# ---------------------------------------------------------
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos):
    # Transponer: Filas = Palabras, Columnas = Documentos/Oraciones
    matriz_palabras = matriz.T 
    
    # Reducir a 3 dimensiones con PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matriz_palabras.toarray())
    
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    
    # Graficar los puntos en 3D
    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors='k', alpha=0.8, depthshade=True)
    
    # Etiquetar cada palabra en el espacio
    for i, palabra in enumerate(vocabulario):
        ax.text(x[i], y[i], z[i] + 0.02, palabra, fontsize=8)
        
    ax.set_title(titulo)
    ax.set_xlabel('Comp. Principal 1')
    ax.set_ylabel('Comp. Principal 2')
    ax.set_zlabel('Comp. Principal 3')


# ---------------------------------------------------------
# RENDERIZADO DE LOS GRÁFICOS
# ---------------------------------------------------------
fig = plt.figure(figsize=(16, 7))

# Gráfico 1: Bag of Words
ax1 = fig.add_subplot(121, projection='3d')
graficar_palabras_3d(ax1, X_bow, vocab_bow, "Espacio BoW 3D (Conteos)", "orange")

# Gráfico 2: TF-IDF
ax2 = fig.add_subplot(122, projection='3d')
graficar_palabras_3d(ax2, X_tfidf, vocab_tfidf, "Espacio TF-IDF 3D (Importancia)", "teal")

plt.tight_layout()
plt.show()