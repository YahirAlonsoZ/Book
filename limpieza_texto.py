import re
import spacy

# Cargar el modelo de español
nlp = spacy.load("es_core_news_sm")

# Leer el archivo
with open("libro.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# Normalización
texto = texto.lower()                     # Minúsculas
texto = re.sub(r"\d+", "", texto)         # Eliminar números
texto = re.sub(r"[^\w\s]", "", texto)     # Eliminar signos de puntuación
texto = re.sub(r"\s+", " ", texto).strip()

# Lematización
doc = nlp(texto)

lemas = []

for token in doc:
    if not token.is_stop and not token.is_punct and not token.is_space:
        lemas.append(token.lemma_)

texto_procesado = " ".join(lemas)

# Guardar resultado
with open("libro_procesado.txt", "w", encoding="utf-8") as archivo:
    archivo.write(texto_procesado)

print("Proceso completado.")
print("Texto original:")
print(texto[:200])

print("\nTexto procesado:")
print(texto_procesado[:300])