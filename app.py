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

# 📌 Récupérer le jeton Hugging Face depuis les secrets Streamlit
HF_TOKEN = st.secrets.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("⚠️ Hugging Face token non défini ! Vérifiez vos secrets sur Streamlit Cloud.")
    st.stop()

# 📌 Charger le modèle localement
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

# 📌 Configuration du modèle LangChain
llm = HuggingFacePipeline(pipeline=pipe)

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
        # Filtrer les liens inutiles (réseaux sociaux, fichiers, etc.)
        if any(social in link for social in ["facebook", "twitter", "linkedin", "instagram", "mailto"]):
            continue

        # Ajouter le domaine si le lien est relatif
        full_link = link if link.startswith("http") else url + link

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
            st.warning(f"❌ Impossible de récupérer {full_link} : {e}")

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

# 📌 Chargement des articles
documents = scrape_articles(base_url)
if not documents:
    st.error("⚠️ Aucun article n'a été récupéré. Vérifiez l'URL ou la connexion Internet.")
    st.stop()

processed_articles = preprocess_articles(documents)

# 📌 Création des embeddings pour le RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 📌 Indexation des documents dans FAISS
vectorstore = FAISS.from_documents(processed_articles, embeddings)

# 📌 Chaîne de questions-réponses avec récupération
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 📌 Fonction pour répondre aux questions
def repondre(question):
    try:
        reponse = qa_chain.run(question)
        sources = [doc.metadata["url"] for doc in qa_chain.retriever.get_relevant_documents(question)]
        return f"{reponse}\n\nSources:\n" + "\n".join(sources)
    except Exception as e:
        return f"❌ Erreur lors de la génération de la réponse : {e}"

# 📌 Interface Streamlit
st.title("Chatbot RAG")
st.write("Posez vos questions, et le chatbot y répondra en s'appuyant sur des articles scrappés.")

# 📌 Champ de saisie pour la question
question = st.text_input("Votre question :")

# 📌 Bouton pour soumettre la question
if st.button("Soumettre"):
    if question:
        reponse = repondre(question)
        st.write(f"📝 Réponse : {reponse}")
    else:
        st.warning("Veuillez entrer une question.")
