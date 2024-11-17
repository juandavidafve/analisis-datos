import os
from dotenv import load_dotenv
import pandas as pd
import stanza
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai

load_dotenv()
pd.set_option('display.max_columns', None)

# Crear cache
os.makedirs("cache", exist_ok=True)

# Leer datos
if os.path.exists('cache/comentario.csv'):
    df_comentario = pd.read_csv('cache/comentario.csv')
else:
    df_comentario = pd.read_csv('dataset/comentario.csv')

if os.path.exists('cache/respuesta.csv'):
    df_respuesta = pd.read_csv('cache/respuesta.csv')
else:
    df_respuesta = pd.read_csv('dataset/respuesta.csv')

if os.path.exists('cache/resena.csv'):
    df_resena = pd.read_csv('cache/resena.csv')
else:
    df_resena = pd.read_csv('dataset/resena.csv')

if os.path.exists('cache/registro.csv'):
    df_registro = pd.read_csv('cache/registro.csv')
else:
    df_registro = pd.read_csv('dataset/registro.csv').merge(
        pd.read_csv('dataset/resultado.csv'),
        how='left',
        left_on='id_resultado',
        right_on='id'
    ).drop(
        columns=['id_resultado', 'id_y']
    ).rename(
        columns={
            'descripcion': 'resultado',
            'id_x': 'id'
        }
    ).merge(
        pd.read_csv('dataset/proceso.csv'),
        how='left',
        left_on='id_proceso',
        right_on='id'
    ).drop(
        columns=['id_proceso', 'id_y']
    ).rename(
        columns={
            'nombre': 'proceso',
            'id_x': 'id'
        }
    ).merge(
        pd.read_csv('dataset/producto.csv'),
        how='left',
        left_on='id_producto',
        right_on='id'
    ).drop(
        columns=['id_producto', 'id_y', "precio"]
    ).rename(
        columns={
            'nombre': 'producto',
            'id_x': 'id'
        }
    )

if os.path.exists('cache/pqrs.csv'):
    df_pqrs = pd.read_csv('cache/pqrs.csv')
else:
    df_pqrs = pd.read_csv('dataset/pqrs.csv').merge(
        pd.read_csv('dataset/tipo_pqrs.csv'),
        how='left',
        left_on='id_tipo_pqrs',
        right_on='id'
    ).drop(
        columns=['id_tipo_pqrs', 'id_y']
    ).rename(
        columns={
            'descripcion': 'tipo_pqrs',
            'id_x': 'id'
        }
    )

# Tarea 0 - Preprocesamiento

nlp = stanza.Pipeline(
    lang='es',
    processors='tokenize,mwt,pos,lemma,sentiment',
    tokenize_no_ssplit=True
)


def preprocess_df(df, col_name):
    sentiments = []
    lematized = []

    for text in df[col_name]:
        lematized_words = []
        sentiment = 0

        for sentence in nlp(text).sentences:
            for word in sentence.words:
                stop_pos_tags = {'DET', 'PRON', 'ADP', 'AUX',
                                 'SCONJ', 'CCONJ', 'PART', 'INTJ', 'PUNCT'}
                if (word.upos not in stop_pos_tags):
                    lematized_words.append(word.lemma)

            if (sentence.sentiment == 0):
                sentiment = sentiment - 1

            if (sentence.sentiment == 2):
                sentiment = sentiment + 1

        sentiments.append(sentiment)
        lematized.append(" ".join(str(element) for element in lematized_words))

    df["lematized"] = lematized
    df['sentiments'] = sentiments


if ('lematized' not in df_comentario.columns):
    preprocess_df(df_comentario, "contenido")

if ('lematized' not in df_respuesta.columns):
    preprocess_df(df_respuesta, "contenido")

if ('lematized' not in df_resena.columns):
    preprocess_df(df_resena, "contenido")

if ('lematized' not in df_pqrs.columns):
    preprocess_df(df_pqrs, "contenido")

if ('lematized' not in df_registro.columns):
    preprocess_df(df_registro, "anotaciones")

# Filtrar los comentarios negativos
df_comentario = df_comentario[df_comentario['sentiments'] == -1]
df_respuesta = df_respuesta[df_respuesta['sentiments'] == -1]
df_resena = df_resena[df_resena['sentiments'] == -1]
df_pqrs = df_pqrs[df_pqrs['sentiments'] == -1]
df_registro = df_registro[
    (df_registro['resultado'] == 'Mejorable') |
    (df_registro['resultado'] == 'Fracaso')
]

# Guardar en cache, para no estar recalculando
df_comentario.to_csv("cache/comentario.csv", index=False)
df_respuesta.to_csv("cache/respuesta.csv", index=False)
df_resena.to_csv("cache/resena.csv", index=False)
df_pqrs.to_csv("cache/pqrs.csv", index=False)
df_registro.to_csv("cache/registro.csv", index=False)

# Tarea 1 - Hallar problema inicial
# Combinar comentarios, encuestas, reseñas y pqrs en un solo dataset
df = pd.DataFrame()
tipos = []
observaciones = []

for row in df_comentario.itertuples():
    tipos.append("comentario")
    observaciones.append(row.lematized)

for row in df_respuesta.itertuples():
    tipos.append("encuesta")
    observaciones.append(row.lematized)

for row in df_resena.itertuples():
    tipos.append("reseña")
    observaciones.append(row.lematized)

for row in df_pqrs.itertuples():
    tipos.append(row.tipo_pqrs)
    observaciones.append(row.lematized)

df["tipo"] = tipos
df["observacion"] = observaciones

print('Dataset:')
print(df.head())

# Asignar un número para cada posible valor de las variables cualitativas
encoder = LabelEncoder()
df_encoded = df[['tipo']].copy()
for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
    df_encoded[column] = encoder.fit_transform(df_encoded[column])

# Aplicar TF-IDF
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['observacion'])

# Juntar el calculo de TF-IDF y el dataset con las variables codificadas en uno solo
data = hstack([tfidf, df_encoded[['tipo']].values])

# Hallar cantidad optima de clusters
wcss = []  # Within-Cluster Sum of Squares: Variabilidad
ss = []  # Puntuacion de Silueta: Separación de los clusteres

# Calcular PCA (Principal Component Analysis)
pca = PCA(2)
pca_encoded = pca.fit_transform(data)

# Buscar el valor optimo para k-means
for i in range(2, 12):
    kmean = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmean.fit(data)
    wcss.append(kmean.inertia_)

    y = kmean.predict(data)
    s = silhouette_score(pca_encoded, y, random_state=0)
    ss.append(round(s, 5))

# Crear la grafica para el numero de k-means
fig, axs = plt.subplots(1, 2, figsize=(15, 4))
sns.pointplot(x=list(range(2, 12)), y=wcss, color='Green', ax=axs[0])
axs[0].set_title("Elbow Plot", size=15)
axs[0].set_xlabel("Number of Clusters", size=12)

sns.pointplot(x=list(range(2, 12)), y=ss, ax=axs[1], color='Teal')
axs[1].set_title("Silhouette Plot", size=15)
axs[1].set_xlabel("Number of Clusters", size=12)

max_ss = max(ss)  # Maximum silhouette score
optimal_k = ss.index(max_ss) + 2  # Adding 2 because range starts at 2

print(f"Maximum Silhouette Score: {max_ss}")
print(f"Optimal Number of Clusters (K): {optimal_k}")

plt.show()

# Aplicar k-means pero esta vez con el numero optimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(data)
df['cluster'] = kmeans.labels_

# Crear grafica en 3d de los clusters
pca_3d = PCA(n_components=3)
pca_df_3d = pca_3d.fit_transform(data)
centers_3d = pca_3d.transform(kmeans.cluster_centers_)

fig = px.scatter_3d(
    x=pca_df_3d[:, 0],
    y=pca_df_3d[:, 1],
    z=pca_df_3d[:, 2],
    color=kmeans.labels_,
    title="KMeans Clustering (3D)",
    labels={
        'x': 'PCA Component 1',
        'y': 'PCA Component 2',
        'z': 'PCA Component 3'
    }
)

# Add cluster centroids to the 3D plot
fig.add_scatter3d(
    x=centers_3d[:, 0],
    y=centers_3d[:, 1],
    z=centers_3d[:, 2],
    mode='markers',
    marker=dict(size=5, color='red', symbol='x'),
    name='Centroids'
)

fig.show()

print('Dataset con clusters:')
print(df.head())

# Guardar el dataframe clusterizado en cache
df.to_csv("cache/df.csv")

# Inicializar AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

response = model.generate_content(
    '''Given the dataset:
    {df}
    For each cluster, resume it into one problem statement.
    Use only one sentence.
    Only give the problem statement.
    Give the results as break line separated list, ready to be used with string.split(sep="\\n").
    Use spanish language.'''
    .format(df=df.to_string())).text

problems = [[x] for x in response.split(sep="\n") if x]

print("Problemas encontrados:")
print("\n".join(map(str, [x[0] for x in problems])))

# Tarea 2 - Hallar causa raiz
i = 0
for problem_arr in problems:
    while len(problem_arr) < 6:
        i += 1

        cause = model.generate_content(
            '''Given the dataset:
            {df}
            Which anotacion can be a cause for the problem {problem}.
            If no anotacion satisfy this, give a possible cause that could be inferred from the dataset.
            Give the results in spanish.
            Give the results as a single phrase.
            Only return the main cause detected, dont include another extra information.
            The cause detected must not be in the array {previous_answers} or be similar to one of its elements.
            If there are no possible answers that satisfy these conditions, return None'''
            .format(df=df_registro, previous_answers=problem_arr, problem=problem_arr[-1])
        ).text.replace("\n", "")

        if (cause != 'None'):
            problem_arr.append(cause)
        else:
            break

# Tarea 3 - Definicion del problema
answers = ''
for problem_arr in problems:
    problem_text = f'¿Por qué {problem_arr[0]}?\n'
    for problem in problem_arr[1:-1]:
        problem_text += f'- Porque {problem}\n¿Por qué {problem}?\n'
    problem_text += f'- Porque {problem_arr[-1]}\n'
    answers += f'{problem_text}\n'

# Usar AI para ajustar la redacción, y generar un resumen.
problem_statement = model.generate_content(
    '''Refine the wording of the text in Spanish to ensure it is syntactically correct, while maintaining the same style, language and text structure.
    Once the adjustments are made, add a summary at the end.
    {answers}'''
    .format(answers=answers)
).text

print("Resultados:")
print(problem_statement)
