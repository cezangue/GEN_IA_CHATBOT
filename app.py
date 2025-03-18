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

# Définition du token et du modèle
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez votre environnement.")
    st.stop()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Chargement du modèle
llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

# Fonction de scraping (simplifiée)
def scrape_articles(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Erreur lors de l'accès à {url} : {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    articles = []
    for p in soup.find_all('p'):
        articles.append(Document(page_content=p.get_text(strip=True), metadata={"url": url}))
    
    return articles

# URL de la source
base_url = "https://www.agenceecofin.com/"
documents = scrape_articles(base_url)

if not documents:
    st.warning("⚠️ Aucun article récupéré. Vérifiez l'URL ou le site source.")
else:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Utilisation de RetrievalQA au lieu de ConversationalRetrievalChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    # Interface Streamlit
    st.title("🤖 Agent Conversationnel")
    st.write("Bienvenue sur votre agent conversationnel, veuillez me poser votre question.")

    question = st.text_input("Votre question :")
    if st.button("Poser la question"):
        if question:
            reponse = qa_chain.run(question)
            st.write("📝 Réponse :", reponse)
        else:
            st.warning("Veuillez entrer une question.")
