import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📚 Movie dataset: Bollywood titles with short descriptions
titles = [
    "3 Idiots", "Taare Zameen Par", "PK", "Dangal", "Chak De! India",
    "Swades", "Bhaag Milkha Bhaag", "Barfi!", "Drishyam", "Kahaani"
]

descriptions = [
    "Three engineering students challenge the education system and redefine success.",
    "A dyslexic child struggles in school until a teacher discovers his hidden talent.",
    "An alien lands on Earth and questions religious dogmas through satire.",
    "A former wrestler trains his daughters to become wrestling champions.",
    "A disgraced hockey player coaches an Indian women’s team to world victory.",
    "An NRI engineer returns to India and finds his purpose in rural development.",
    "The story of Indian athlete Milkha Singh's journey through hardship and glory.",
    "A mute and autistic man falls in love while uncovering a murder mystery.",
    "A man goes to great lengths to protect his family after a crime is committed.",
    "A pregnant woman searches for her missing husband in the streets of Kolkata."
]

# 🧱 Create DataFrame
df = pd.DataFrame({'title': titles, 'description': descriptions})

# 🔍 Convert descriptions to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# 🔗 Compute cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 🎯 Map movie titles to their index
title_to_index = {t.lower(): i for i, t in enumerate(df['title'])}

# 🤖 Recommendation function
def recommend(title):
    title = title.lower()
    if title not in title_to_index:
        return "Movie not found."
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    indices = [i[0] for i in sim_scores]
    return df['title'].iloc[indices]

# 🧪 Test example
print("🎬 Recommended for '3 Idiots':")
print(recommend("3 Idiots"))
