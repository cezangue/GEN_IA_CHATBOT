import streamlit as st
import os
import requests
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Récupérer le jeton Hugging Face
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez vos secrets.")
    st.stop()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

# Fonction pour scraper des articles
def scrape_articles(url):
    # (Votre code de scraping ici)

# Prétraiter les articles
def preprocess_articles(documents):
    # (Votre code de prétraitement ici)

# Chargement des articles
base_url = "https://www.agenceecofin.com/"
documents = scrape_articles(base_url)
processed_articles = preprocess_articles(documents)

# Créer des embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(processed_articles, embeddings)

# Vérifiez le nombre de documents
st.write(f"Documents dans le vecteur store : {len(processed_articles)}")

# Chaîne de questions-réponses avec récupération
try:
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
except Exception as e:
    st.error(f"Erreur lors de la création de la chaîne QA : {e}")
    st.stop()

# Fonction de réponse
def repondre(question):
    # (Votre fonction de réponse ici)

# Interface Streamlit
st.title("Chatbot RAG")
st.write("Posez vos questions, et le chatbot y répondra en s'appuyant sur des articles scrappés.")
question = st.text_input("Votre question :")
if question:
    reponse = repondre(question)
    st.write(f"📝 Réponse : {reponse}")import streamlit as st
import os
import requests
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Récupérer le jeton Hugging Face
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez vos secrets.")
    st.stop()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

# Fonction pour scraper des articles
def scrape_articles(url):
    # (Votre code de scraping ici)

# Prétraiter les articles
def preprocess_articles(documents):
    # (Votre code de prétraitement ici)

# Chargement des articles
base_url = "https://www.agenceecofin.com/"
documents = scrape_articles(base_url)
processed_articles = preprocess_articles(documents)

# Créer des embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(processed_articles, embeddings)

# Vérifiez le nombre de documents
st.write(f"Documents dans le vecteur store : {len(processed_articles)}")

# Chaîne de questions-réponses avec récupération
try:
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
except Exception as e:
    st.error(f"Erreur lors de la création de la chaîne QA : {e}")
    st.stop()

# Fonction de réponse
def repondre(question):
    # (Votre fonction de réponse ici)

# Interface Streamlit
st.title("Chatbot RAG")
st.write("Posez vos questions, et le chatbot y répondra en s'appuyant sur des articles scrappés.")
question = st.text_input("Votre question :")
if question:
    reponse = repondre(question)
    st.write(f"📝 Réponse : {reponse}")
