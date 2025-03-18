import os
import requests
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

# 📌 Définir votre jeton Hugging Face à partir des variables d'environnement
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez vos variables d'environnement.")
    st.stop()

# 📌 Définition du modèle IA
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# 📌 Chargement du modèle Hugging Face
llm = HuggingFaceEndpoint(
    repo_id=MODEL_ID,
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

# 📌 Fonction pour scraper les articles
def scrape_articles(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Vérifie les erreurs HTTP
    except requests.RequestException as e:
        st.error(f"🚨 Erreur lors de l'accès à {url} : {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = {a['href'] for a in soup.find_all('a', href=True)}

    articles = []
    for link in links:
        if any(social in link for social in ["facebook", "twitter", "linkedin", "instagram", "mailto"]):
            continue

        full_link = urljoin(url, link)

        try:
            article_response = requests.get(full_link, timeout=10)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.content, 'html.parser')

            title = article_soup.find('h1')
            title = title.get_text(strip=True) if title else "Titre non trouvé"

            content = " ".join([p.get_text(strip=True) for p in article_soup.find_all('p')])
            if content:
                articles.append(Document(page_content=f"{title}\n\n{content}", metadata={"url": full_link}))
        except requests.RequestException as e:
            st.error(f"❌ Impossible de récupérer {full_link} : {e}")

    return articles

# 📌 Nettoyage du texte
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Supprime les balises HTML
    text = re.sub(r'\s+', ' ', text).strip()  # Supprime les espaces inutiles
    return text

# 📌 Prétraitement des articles
def preprocess_articles(documents):
    return [Document(page_content=clean_text(doc.page_content), metadata=doc.metadata) for doc in documents]

# 📌 URL de la page à scraper
base_url = "https://www.agenceecofin.com/"
documents = scrape_articles(base_url)

# 📌 Vérification et traitement des articles
if not documents:
    st.warning("⚠️ Aucun article n'a été récupéré. Vérifiez l'URL ou le site source.")
else:
    processed_articles = preprocess_articles(documents)

    # 📌 Création des embeddings pour le RAG
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 📌 Indexation des documents dans FAISS
    vectorstore = FAISS.from_documents(processed_articles, embeddings)

    # 📌 Chaîne de questions-réponses avec récupération
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # 📌 Interface Streamlit
    st.title("Chatbot RAG")
    st.write("Posez vos questions, et le chatbot y répondra en s'appuyant sur des articles scrappés.")
    question = st.text_input("Votre question :")
    
    if question:
        reponse = qa_chain.run(question)
        st.write(f"📝 Réponse : {reponse}")
