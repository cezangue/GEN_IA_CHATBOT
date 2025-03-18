# GEN_IA_CHATBOT-Chatbot RAG
il s'agit du montage d'un chatbot basé sur l'architecture RAG du site ECOFIN

Ce projet utilise un modèle de langage basé sur Retrieval-Augmented Generation (RAG) pour répondre aux questions des utilisateurs en scrappant des articles en ligne. Le chatbot utilise Hugging Face pour le modèle IA et Streamlit pour l'interface web.

## Fonctionnement
1. Scraping des articles à partir d'une URL donnée.
2. Utilisation de LangChain pour le modèle RAG avec Hugging Face.
3. Interface web construite avec Streamlit pour interagir avec le chatbot.

## Déploiement
1. Clonez ce dépôt.
2. Installez les dépendances avec `pip install -r requirements.txt`.
3. Lancez l'application avec `streamlit run app.py`.

## Technologies utilisées
- LangChain
- Hugging Face
- FAISS
- Streamlit
