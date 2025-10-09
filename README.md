
# 🎬 Movie Buddy – AI Movie Recommendation Chatbot

**Movie Buddy** is an intelligent conversational chatbot that recommends movies based on what you like using **semantic similarity** and **natural language understanding**.  
Built with Python and Sentence Transformers, it allows users to chat naturally and discover similar movies through AI-powered text analysis.

---

## 📘 Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Model & Dataset](#model--dataset)
- [Screenshots / Demo](#screenshots--demo)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## 🧩 About the Project

**Movie Buddy** is designed to make movie discovery simple and conversational.  
Instead of browsing endless lists, users can just chat with Movie Buddy — describing movies they like, genres they prefer, or moods they’re in — and get tailored recommendations instantly.

The chatbot leverages **Sentence Transformers** to convert movie descriptions and user queries into embeddings, then finds the most semantically similar movies using **cosine similarity**.

---

## ✨ Features
- 🗣 Conversational interface for natural chatting  
- 🎥 Recommends movies based on meaning, not just keywords  
- ⚙️ Uses Sentence Transformers or TF-IDF embeddings  
- 🔊 Optional voice responses using `pyttsx3`  
- 📊 Dataset-based search by title, genre, cast, or overview  
- ⏳ Simulated typing delays for a realistic chat experience  

---

## 🛠 Tech Stack

**Languages & Frameworks**
- Python 3.9+

**Core Libraries**
- `pandas`
- `numpy`
- `sentence-transformers`
- `torch`
- `scikit-learn`
- `pyttsx3`
- `re`, `os`, `datetime`

---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/movie-buddy.git
   cd movie-buddy
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**
   Place your movie dataset (e.g., `movies.csv`) in the project folder.
   The dataset should include columns like:

   ```
   id, title, genres_list, cast_list, overview, release_year, vote_average
   ```

4. **Run Movie Buddy**

   ```bash
   python movie_buddy.py
   ```

---

## 💡 Usage

Once started, **Movie Buddy** will greet you and ask about your movie preferences.
You can type things like:

```
User: I liked Interstellar.
Movie Buddy: You might also enjoy Inception, The Martian, and Gravity!
```

Or describe a mood:

```
User: I want a funny superhero movie.
Movie Buddy: You might like Deadpool, Guardians of the Galaxy, or Thor: Ragnarok!
```

---

## 🧠 Model & Dataset

* **Model Used:** `sentence-transformers/all-MiniLM-L6-v2`
* **Embedding Method:** Sentence embeddings for semantic similarity
* **Similarity Metric:** Cosine similarity (`sklearn.metrics.pairwise.cosine_similarity`)
* **Dataset:** Custom CSV file with movie metadata (title, genres, cast, overview, etc.)

---

## 🖼️ Screenshots / Demo

*(Add terminal screenshots or interface previews here)*

Example:

```text
🎬 Hello! I’m Movie Buddy — your personal movie assistant.
What kind of movie are you in the mood for today?
```

---

## 🚀 Future Improvements

* Add a **web-based UI** using Streamlit or Gradio
* Include **voice input** and speech-to-text functionality
* Connect with live movie APIs (e.g., TMDB API)
* Implement **user-based collaborative filtering**
* Enable **personalized user profiles**

---

## 👤 Author

**Mohit Kumar**
Email : mohit260raj@gmail.com
LinkedIn : https://www.linkedin.com/in/mohit-kumar-iitp/ 

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

⭐ **If you like this project, don’t forget to give it a star on GitHub!**


