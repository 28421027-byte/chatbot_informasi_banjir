import json
import pickle
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# 1. Load Dataset
with open('intents.json') as file:
    data = json.load(file)

# Mengubah JSON ke format tabel (DataFrame) agar mudah diolah
all_patterns = []
all_tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(intent['tag'])

df = pd.DataFrame({'text': all_patterns, 'intent': all_tags})

# 2. Preprocessing (Bahasa Indonesia)
print("Sedang melakukan preprocessing dengan Sastrawi...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    # Case folding (kecilkan huruf) & Stemming (kata dasar)
    text = text.lower()
    text = stemmer.stem(text)
    return text

df['text_clean'] = df['text'].apply(clean_text)

# 3. Feature Extraction & Modeling (Pipeline)
# Kita gunakan TF-IDF Vectorizer untuk mengubah teks ke angka
# Dan MultinomialNB sebagai algoritmanya
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 4. Training Model
print("Memulai training model Naive Bayes...")
model.fit(df['text_clean'], df['intent'])

# 5. Evaluasi Sederhana
y_pred = model.predict(df['text_clean'])
print(f"Training Accuracy: {accuracy_score(df['intent'], y_pred) * 100:.2f}%")

# 6. Simpan Model dan Data Pendukung
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model berhasil disimpan dengan nama 'chatbot_model.pkl'!")