# PSUIA_Q2_Practica1_FAISS.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Cargar modelo paraphrase-multilingual-MiniLM-L12-v2, que transforma frases en vectores
# de 384 dimensiones.
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Definimos nuestras frases de ejemplo en una lista y creamos los embeddings para cada frase:
documents = [
    "El amanecer en la montaña revela colores que no existen en la ciudad.",
    "Los algoritmos cuánticos prometen cambiar la forma en que resolvemos problemas complejos.",
    "Un perro corre feliz por la playa mientras las olas rompen suavemente.",
    "La historia del arte está llena de secretos que aún no hemos descubierto.",
    "El café colombiano tiene un aroma que despierta recuerdos de infancia."
]

# Embeddings
doc_vectors = model.encode(documents)
doc_vectors = np.array(doc_vectors, dtype='float32')

# Crear índice FAISS
# IndexFlatL2 crea un índice que calcula distancias Euclídeas (L2) directamente sobre los vectores "en crudo".
# Es simple pero para conjuntos pequeños va perfecto
dimension = doc_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_vectors)

# Consulta en lenguaje natural, convertimos la consulta a vector usando el mismo modelo
query = "¿Qué bebida tiene un olor que me recuerda a cuando era niño?"
query_vector = model.encode([query])
query_vector = np.array(query_vector, dtype='float32')

# Búsqueda en el índice FAISS del vector más similar (k=1)
k = 1
distances, indices = index.search(query_vector, k)

# Mostrar el resultado
best_idx = indices[0][0]
print("Consulta:", query)
print("Respuesta:", documents[best_idx])
print("Distancia L2:", distances[0][0])
print("Forma de la matriz:", doc_vectors.shape)   # (5, 384)
print("Tipo de dato:", doc_vectors.dtype)          # float32
print("vectores frases",doc_vectors)                                 # NumPy la muestra recortada con "..."
print("vector query",query_vector)
