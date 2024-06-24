import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFace
import streamlit as st

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

documents = [download_text(url) for url in links]

model_name = "hiiamsid/sentence_similarity_spanish_es"
embedding_model = SentenceTransformer(model_name)
embeddings = HuggingFaceEmbeddings(model=embedding_model)

faiss_index = FAISS.from_documents(documents, embeddings)

llm_model_name = "llama3-chatqa"
llm = HuggingFace(model_name=llm_model_name)

st.title("Aplicación RAG con LLM en Español")

query = st.text_input("Ingrese su consulta:")

if query:
    relevant_docs = faiss_index.similarity_search(query)
    context = ' '.join(relevant_docs)
    response = llm(context=context, question=query)
    
    st.write("Respuesta del modelo:")
    st.write(response)
