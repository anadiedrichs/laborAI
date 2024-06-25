
#from bs4 import BeautifulSoup
#from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import streamlit as st
from huggingface_hub.hf_api import HfFolder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# Descargar y procesar textos
links = [
    "https://www.argentina.gob.ar/normativa/nacional/ley-20744-25552/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-24013-412/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-24557-27971/actualizacion",
    "https://www.argentina.gob.ar/normativa/nacional/ley-11544-63368/actualizacion"
]

titles = [
"REGIMEN DE CONTRATO DE TRABAJO  LEY N° 20.744",
"EMPLEO Ley Nº 24.013",
"RIESGOS DEL TRABAJO Ley N° 24.557",
"JORNADA DE TRABAJOLey 11.544"
]


loader = AsyncHtmlLoader(links)
docs = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["article"])

documents = []
for i in range(len(docs_transformed)):
  metadata = {"title": titles[i], "source": links[i]}
  d = Document(metadata=metadata, page_content = docs_transformed[i].page_content)
  documents.append(d)

chunks = split_documents(documents)


# Crear embeddings y base de datos vectorial

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")


# Convertir documentos a embeddings
#document_embeddings = embedding_model.encode(documents)

# Crear FAISS index
faiss_index = FAISS.from_documents(documents=chunks, embedding=embedding_model)


# Configurar el modelo LLM

# Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct to ask for access.

#INTRODUCIR TOKEN DE HUGGINGFACE. 
# DEBE ESTAR APROBADA LA SOLICITUD DE USO DE MODELO LLAMA-3
HfFolder.save_token('YOUR_HF_TOKEN') 

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
