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

documents = [download_text(url) for url in links]

# Crear embeddings y base de datos vectorial

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


# Convertir documentos a embeddings
#document_embeddings = embedding_model.encode(documents)

# Crear FAISS index
faiss_index = FAISS.from_texts(texts=documents, embedding=embedding_model)


# Configurar el modelo LLM
from huggingface_hub.hf_api import HfFolder

HfFolder.save_token('') #INTRODUCIR TOKEN DE HUGGINGFACE. DEBE ESTAR APROBADA LA SOLICITUD DE USO DE MODELO LLAMA-3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

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
