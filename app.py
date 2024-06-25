import streamlit as st
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import Chroma

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain.schema.output_parser import StrOutputParser
import os

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL="llama3-chaqa"

PROMPT_TEMPLATE = """
Responda en español a la siguiente pregunta:

{question}

, basándose en el siguiente contexto: {context}
"""

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


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def download_documents():
   
    loader = AsyncHtmlLoader(links)
    docs = loader.load()
    return docs

# Regresa arreglo de Document, los textos en pedacitos
def get_dataset():

    docs = download_documents()
    print("Texto legal descargado.")
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["article"])

    documents = []
    for i in range(len(docs_transformed)):
        metadata = {"title": titles[i], "source": links[i]}
        d = Document(metadata=metadata, page_content = docs_transformed[i].page_content)
        documents.append(d)

    chunks = split_documents(documents)
    print("Documentos creados.")
    return chunks


# Base de datos 
def create_or_load_database():

    collection_name = "laborAI_db"
    db_directory = f"./{collection_name}"


    # create the open-source embedding function
    embedding = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Verificar si el directorio de la base de datos existe
    if not os.path.exists(db_directory):

        
        print("Inicio de proceso de creación de base de datos.")

        os.makedirs(db_directory)
        
        # descarga y armado del dataset 
        chunks = get_dataset()

        database = Chroma.from_documents(chunks,
                                        embedding = embedding, 
                                        collection_name=collection_name,
                                        persist_directory=db_directory)
        # save to disk

        print("Base de datos CREADA y cargada exitosamente.")
        
    else:
        # cargar la base de datos a memoria or load the data base
        database = Chroma(collection_name=collection_name, 
                          embedding_function=embedding, 
                          persist_directory=db_directory)
        
        print("Base de datos cargada exitosamente.")


   
    return database

def get_rag_chain(retriever):

    # Configurar el modelo LLM
    llm = Ollama(model = MODEL)
    
    retrieval = RunnableParallel(
        {"context": retriever,  "question": RunnablePassthrough()}
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = retrieval | prompt | llm | StrOutputParser()

    return chain





database = create_or_load_database()

print("Hay ", database._collection.count(), " elementos en la base de datos ")


# query it
#query = "Despidos"
#docs = database.similarity_search(query)
#print("Test de base de datos.")
# print results
#print(docs[0].page_content)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
retriever =  database.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)


chain = get_rag_chain(retriever)

# Crear la aplicación Streamlit
st.title("LaborAI: aplicación RAG + LLM en Español para consultas de derecho laboral")

question = st.text_input("Ingrese su pregunta:")

if question:

    result = chain.invoke(question)
    st.write("Respuesta:")
    st.write(result)
    contexts = []
    contexts.append([docs.page_content for docs in retriever.get_relevant_documents(question)])
    st.write("Contexto:")
    st.write(contexts)

