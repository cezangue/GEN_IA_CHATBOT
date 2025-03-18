import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# Récupérer le jeton Hugging Face
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez vos secrets sur Streamlit Cloud.")
    st.stop()

# Charger le modèle localement
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
pipe = pipeline(
    "text-generation",
    model=model_name,
    token=HF_TOKEN,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.95,
    top_p=0.95
)

# Configuration du modèle LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Fonction pour scraper des articles
def scrape_articles(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        st.error(f"🚨 Erreur lors de la récupération de la page : {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    links = {a['href'] for a in soup.find_all('a', href=True)}
    articles = []
    for link in links:
        if any(social in link for social in ["facebook", "twitter", "linkedin", "instagram", "mailto"]):
            continue
        full_link = link if link.startswith("http") else url + link
        try:
            article_response = requests.get(full_link, timeout=10)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            title = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else "Titre non trouvé"
            content = " ".join([p.get_text(strip=True) for p in article_soup.find_all('p')])
            if content:
                articles.append(Document(page_content=f"{title}\n\n{content}", metadata={"url": full_link}))
        except requests.RequestException as e:
            st.warning(f"❌ Impossible de récupérer {full_link} : {e}")
    return articles

# Prétraiter les articles
def preprocess_articles(documents):
    return [Document(page_content=re.sub(r'<.*?>', '', doc.page_content), metadata=doc.metadata) for doc in documents]

# Chargement des articles
base_url = "https://www.agenceecofin.com/"
documents = scrape_articles(base_url)
if not documents:
    st.error("Aucun article n'a été récupéré. Vérifiez l'URL ou la connexion Internet.")
    st.stop()

processed_articles = preprocess_articles(documents)

# Créer des embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(processed_articles, embeddings)

# Chaîne de questions-réponses avec récupération
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Fonction de réponse
def repondre(question):
    try:
        reponse = qa_chain.run(question)
        sources = [doc.metadata["url"] for doc in qa_chain.retriever.get_relevant_documents(question)]
        return f"{reponse}\n\nSources:\n" + "\n".join(sources)
    except Exception as e:
        return f"❌ Erreur lors de la génération de la réponse : {e}"

# Interface Streamlit
st.title("Chatbot RAG")
st.write("Posez vos questions, et le chatbot y répondra en s'appuyant sur des articles scrappés.")
question = st.text_input("Votre question :")
if question:
    reponse = repondre(question)
    st.write(f"📝 Réponse : {reponse}")
