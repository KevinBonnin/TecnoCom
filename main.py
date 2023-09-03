import pandas as pd
import numpy as np
from fastapi import FastAPI
import uvicorn
import ast
import joblib
from enum import Enum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
#from fastapi.middleware.cors import CORSMiddleware

inf = pd.read_csv('csv_final.csv')

inf['Year'] = inf['Year'].astype(str)

app = FastAPI()

def recomendacion( titulo ):
    # Reemplazar valores NaN en la columna de sinopsis por una cadena vacía
    inf['genres'] = inf['genres'].fillna('')

    # Reemplazar valores NaN en la columna de belongs_to_collection por una cadena vacía
    inf['specs'] = inf['specs'].fillna('')
    
    # Reemplazar valores NaN en la columna de crew por una cadena vacía
    inf['tags'] = inf['tags'].fillna('')

    # Obtener las características de las películas
    generos = inf['genres'].tolist()
    aspectos = inf['specs'].tolist()
    etiquetas = inf['tags'].tolist()
    #collection = juegos['belongs_to_collection'].tolist()

    n_components = min(50, len(etiquetas[0].split('|')))

    # Vectorizar los castings de las películas utilizando TF-IDF
    vectorizer_etiquetas = TfidfVectorizer()
    etiquetas_vectors = vectorizer_etiquetas.fit_transform(etiquetas)
    # Reducción de dimensionalidad con LSA
    lsa_model_casting = TruncatedSVD(n_components=n_components)
    etiquetas_vectors_reduced = lsa_model_casting.fit_transform(etiquetas_vectors)

    vectorizer_collection = TfidfVectorizer()
    aspectos_vectors = vectorizer_collection.fit_transform(aspectos)
    lsa_model_collection = TruncatedSVD(n_components=n_components)
    aspectos_vectors_reduced = lsa_model_collection.fit_transform(aspectos_vectors)

    vectorizer_genres = TfidfVectorizer()
    genres_vectors = vectorizer_genres.fit_transform(generos)
    lsa_model_genres = TruncatedSVD(n_components=n_components)
    genres_vectors_reduced = lsa_model_genres.fit_transform(genres_vectors)
    
    feature_vectors = np.concatenate((etiquetas_vectors_reduced, aspectos_vectors_reduced, genres_vectors_reduced), axis=1)
    column_names = ['Feature_' + str(i+1) for i in range(3 * n_components)]
    df_feature_vectors = pd.DataFrame(data=feature_vectors, columns=column_names)
    feature_vectors = df_feature_vectors.values

    n_neighbors = 6
    metric = 'cosine'
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(feature_vectors)

    movie_index = inf[inf['title'] == titulo].index[0]

    s, indices = model.kneighbors(feature_vectors[movie_index].reshape(1, -1))

    recommended_movies = inf.loc[indices.flatten()].copy()

    return recommended_movies[['title']].head(5)

@app.get("/genero/{Year}/Los 5 géneros más vendidos.")
def genero( Year: str ):
    juegos = (inf[inf['Year'] == Year])
    juegos['genres'] = juegos['genres'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
    generos_desglosados = juegos['genres'].explode()
    generos_mas_vendidos = generos_desglosados.value_counts().head(5).index.tolist()
    return {'Los generos mas vendidos en el año': Year, 'son': generos_mas_vendidos}

@app.get("/precio/{juego}/Precio del juego solicitado")
def precio(juego: str):
    jueg = (inf[inf['title'] == juego])
    valor = jueg['price'].values[0]
    return {'El valor del juego': juego, 'es': valor}

@app.get("/recomendacion/{juego}/Recomienda 4 juegos similares")
def juegos_recomendados(juego: str):
    recom = recomendacion(juego)
    return {recom}