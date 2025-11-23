import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 🔥 Embeddings locaux avec Ollama
from langchain_ollama import OllamaEmbeddings

# Base vectorielle
from langchain_chroma import Chroma

import gradio as gr

# Charge le .env (facultatif)
load_dotenv()

# ------------------ LOAD BOOKS ------------------

books = pd.read_csv('books_with_emotions.csv')

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover_not_found.png",
    books["large_thumbnail"],
)

# ------------------ LOAD & SPLIT TEXT ------------------

raw_document = TextLoader("tagged_description.txt", encoding="utf-8").load()

text_splitter = CharacterTextSplitter(
    separator="\n", 
    chunk_size=1000, 
    chunk_overlap=0
)

documents = text_splitter.split_documents(raw_document)

# ------------------ OLLAMA EMBEDDINGS ------------------

# ⚠️ Assure-toi d'avoir exécuté : ollama pull nomic-embed-text
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Création de la base vectorielle Chroma
db_books = Chroma.from_documents(documents, embedding_model)

# ------------------ SEMANTIC RETRIEVAL ------------------

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Recherche vectorielle avec embeddings Ollama + Chroma
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extraire ISBN
    books_list = [
        int(rec.page_content.strip('"').split()[0])
        for rec in recs
    ]

    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Filtre catégorie
    if category != "All" and category is not None:
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

    # Tri émotionnel
    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)

# ------------------ RECOMMENDER FUNCTION ------------------

def recommend_books(query: str, category: str, tone: str):

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors = row["authors"].split(";")
        if len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            author_str = f"{', '.join(authors[:-1])} and {authors[-1]}"
        else:
            author_str = authors[0]

        caption = f"{row['title']} by {author_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# ------------------ GRADIO DASHBOARD ------------------

tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
categories = ["All"] + sorted(books["simple_categories"].unique())

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender (Ollama-powered)")

    with gr.Row():
        user_query = gr.Textbox(
            label="Describe a book you want:",
            placeholder="e.g., A story about forgiveness",
        )
        category_dropdown = gr.Dropdown(choices=categories, label="Category", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional tone", value="All")
        submit_button = gr.Button(value="Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()
