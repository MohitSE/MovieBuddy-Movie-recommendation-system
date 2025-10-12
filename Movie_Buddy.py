# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 14:23:59 2025

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
AI-powered Movie Buddy Chatbot
"""

import datetime
import time
import sys
import pandas as pd
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util

# -------- Typing + Delay Helpers --------
def get_greeting():
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "🌞 Good Morning!"
    elif 12 <= hour < 17:
        return "☀️ Good Afternoon!"
    elif 17 <= hour < 21:
        return "🌆 Good Evening!"
    else:
        return "🌙 Good Night!"

def type_text(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

# -------- NLP Helpers --------
def normalize_text_column(col):
    """Convert list-like strings to lowercase text"""
    return col.fillna("").astype(str).str.replace(r"[\[\]']", "", regex=True).str.lower()

# -------- ML-powered Movie Chatbot --------
def movie_buddy():
    # Load dataset
    df = pd.read_csv("cleaned_movies_metadata.csv")

    # Normalize relevant columns
    df['genres_list'] = normalize_text_column(df['genres_list'])
    df['cast_list'] = normalize_text_column(df['cast_list'])
    df['director'] = normalize_text_column(df['director'])
    df['title'] = df['title'].fillna("N/A")
    df['tagline'] = df['tagline'].fillna("")
    df['overview'] = df['overview'].fillna("")
    df['release_date'] = df['release_date'].fillna("N/A")

    # Combine metadata for semantic search
    df['combined_features'] = (
        df['title'] + ' ' +
        df['genres_list'] + ' ' +
        df['cast_list'] + ' ' +
        df['director'] + ' ' +
        df['tagline'] + ' ' +
        df['overview']
    )

    embeddings_file = 'movie_embeddings.pt'

    # Load model (automatically selects GPU if available)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load or compute embeddings in batches
    if os.path.exists(embeddings_file):
        print("✅ Loading precomputed embeddings...")
        movie_embeddings = torch.load(embeddings_file)
    else:
        print("⏳ Computing embeddings in batches (this may take some time)...")
        batch_size = 64
        embeddings_list = []
        for i in range(0, len(df), batch_size):
            start = time.time()
            batch = df['combined_features'].tolist()[i:i+batch_size]
            batch_emb = model.encode(batch, convert_to_tensor=True)
            embeddings_list.append(batch_emb)
            print(f"Batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} done in {time.time()-start:.2f}s")
        movie_embeddings = torch.cat(embeddings_list, dim=0)
        torch.save(movie_embeddings, embeddings_file)
        print("✅ Embeddings saved to movie_embeddings.pt")

    # Welcome
    type_text(get_greeting())
    time.sleep(1)
    type_text("🤖 Hi there! I’m your AI-powered Movie Buddy 🎬")
    time.sleep(1)
    type_text("You can ask me things like:")
    type_text("   - 'I want a comedy movie'")
    type_text("   - 'Show me movies with Tom Hanks'")
    type_text("   - 'Any films by Christopher Nolan?'")
    type_text("Type 'exit' anytime to leave the chat.")

    # Chat loop
    while True:
        user_input = input("\nYou: ").strip()
        if re.search(r"\b(bye|exit|quit|stop)\b", user_input.lower()):
            type_text("👋 Bye! Enjoy your movies! 🍿")
            break

        # Encode user query
        query_embedding = model.encode(user_input, convert_to_tensor=True)

        # Compute cosine similarity
        similarity_scores = util.cos_sim(query_embedding, movie_embeddings)[0]

        # --- KEYWORD BOOSTING ---
        query_lower = user_input.lower()
        boost = torch.zeros_like(similarity_scores)

        # Director boost
        boost_mask = df['director'].str.contains(query_lower, na=False)
        boost[boost_mask] += 2.0

        # Cast boost
        boost_mask = df['cast_list'].str.contains(query_lower, na=False)
        boost[boost_mask] += 1.0

        # Genre boost
        boost_mask = df['genres_list'].str.contains(query_lower, na=False)
        boost[boost_mask] += 0.5

        # Apply boost
        similarity_scores += boost
        # --- END BOOSTING ---

        # Get top 5 movies
        top_indices = torch.topk(similarity_scores, k=5).indices.tolist()
        recommended_movies = df.iloc[top_indices]

        # Display results
        type_text("✅ Here are some movies you might enjoy:\n")
        for _, row in recommended_movies.iterrows():
            title = row["title"]
            release_year = row["release_year"]
            rating = row["vote_average"] if pd.notna(row["vote_average"]) else "N/A"
            tagline = row["tagline"].strip()
            overview = row["overview"].strip()
            cast = row["cast_list"]
            director = row["director"]
            genre = row["genres_list"]

            type_text(f"🎬 {title} ({release_year}) ⭐ {rating}")
            type_text(f"Genre: {genre}")
            type_text(f"   🎭 Cast: {cast}")
            type_text(f"   🎥 Director: {director}")
            if tagline:
                type_text(f"   📝 {tagline}")
            if overview:
                type_text(f"   📖 {overview[:200]}...")  # truncate long plot
            type_text("")

# Run chatbot
if __name__ == "__main__":
    movie_buddy()
