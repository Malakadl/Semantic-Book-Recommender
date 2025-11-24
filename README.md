📚 Semantic Book Recommender

Un système intelligent de recommandation de livres basé sur la similarité sémantique.
Le projet utilise Ollama Embeddings pour transformer les descriptions de livres en vecteurs, puis recherche les ouvrages les plus proches selon leur sens grâce à ChromaDB.

🚀 Fonctionnalités

- Extraction des embeddings via Ollama (modèles locaux, gratuits, rapides).

- Indexation vectorielle avec ChromaDB.

- Recommandation basée sur la similarité cosinus.

- Tableau de bord interactif créé avec Gradio.

- Chargement de datasets personnalisés (CSV, textes…).

- Pipeline simple et reproductible.

🧠 Comment ça fonctionne

1. Chargement du dataset de livres.

2. Nettoyage et segmentation du texte.

3. Génération d’embeddings avec Ollama (nomic-embed-text ou autre modèle).

4. Stockage des vecteurs dans ChromaDB.

5. Lors d’une requête utilisateur, le système trouve les livres les plus similaires sémantiquement.

🛠️ Technologies utilisées

- Python

- Ollama (embeddings)

- LangChain

- ChromaDB

- Gradio (interface utilisateur)

- Pandas / NumPy

📦 Installation

`pip install -r requirements.txt`


Assure-toi d’avoir Ollama installé :

👉 https://ollama.com/download

Puis télécharge un modèle d’embedding :

`ollama pull nomic-embed-text`

▶️ Lancer l'application

`python gradio-dashboard.py`


Une interface web s’ouvrira automatiquement dans ton navigateur.

📁 Structure du projet
```python
📦 book-recommender
 ┣ 📄 gradio-dashboard.py
 ┣ 📄 books_with_emotions.csv
 ┣ 📄 tagged_description.txt
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

🤝 Contributions

Les contributions et suggestions sont les bienvenues !
N’hésite pas à proposer une issue ou un pull request.

📜 Licence

Ce projet est publié sous licence MIT.
