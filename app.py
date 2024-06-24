import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import streamlit as st

# Descargar y procesar textos
links = [
    "https://www.argentina.gob.ar/normativa/nacional/ley-20744-25552/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-24013-412/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-24557-27971/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-11544-63368/actualizacion"
]

def download_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_function(embedding_name):
  # Create a dictionary with model configuration options, specifying to use the CPU for computations
  #model_kwargs = {'device':'cuda'}

  # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
  encode_kwargs = {'normalize_embeddings': True}

  # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
  embeddings = HuggingFaceEmbeddings(
      model_name=embedding_name,     # Provide the pre-trained model's path
      #model_kwargs=model_kwargs, # Pass the model configuration options
      encode_kwargs=encode_kwargs # Pass the encoding options
  )
  return embeddings

documents = [download_text(url) for url in links]

# Crear embeddings y base de datos vectorial

model_name = "hiiamsid/sentence_similarity_spanish_es"
embedding_model = get_embedding_function(model_name)

# Convertir documentos a embeddings
#document_embeddings = embedding_model.encode(documents)

# Crear FAISS index
faiss_index = FAISS.from_texts(texts=documents, embedding=embedding_model)


# Configurar el modelo LLM
#"meta-llama/Llama-2-13b-hf"
llm_model_name = "timpal0l/mdeberta-v3-base-squad2" # "meta-llama/Llama-2-13b-hf" #"nvidia/Llama3-ChatQA-1.5-8B"
llm_pipeline = pipeline("question-answering", model=llm_model_name, token= "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" ) # st.secrets["HF_TOKEN"])

# Crear la aplicación Streamlit
st.title("Aplicación RAG con LLM en Español")

query = st.text_input("Ingrese su consulta:")

if query:
    relevant_docs = faiss_index.similarity_search(query, k=3)
    context = ' '
    for doc in relevant_docs:
       context = context + doc.page_content

    result = llm_pipeline(question=query, context=context)
    
    st.write("Respuesta del modelo:")
    st.write(result['answer'])
