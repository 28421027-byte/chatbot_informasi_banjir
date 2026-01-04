import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("Memuat dataset intents.json...")
with open('intents.json') as file:
    data = json.load(file)

all_patterns = []
all_tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        all_patterns.append(pattern)
        all_tags.append(intent['tag'])

df = pd.DataFrame({'text': all_patterns, 'intent': all_tags})

# ==========================================
# 2. PREPROCESSING (BAHASA INDONESIA)
# ==========================================
print("Sedang melakukan preprocessing dengan Sastrawi (Stemming)...")
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    # Case folding (kecilkan huruf)
    text = text.lower()
    # Stemming (mengubah ke kata dasar)
    text = stemmer.stem(text)
    return text

df['text_clean'] = df['text'].apply(clean_text)

# ==========================================
# 3. MODELING (PIPELINE)
# ==========================================
# Pipeline menggabungkan TF-IDF dan Naive Bayes menjadi satu kesatuan
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# ==========================================
# 4. TRAINING MODEL
# ==========================================
print("Memulai training model Multinomial Naive Bayes...")
model.fit(df['text_clean'], df['intent'])

# ==========================================
# 5. EVALUASI MODEL
# ==========================================
y_true = df['intent']
y_pred = model.predict(df['text_clean'])
acc = accuracy_score(y_true, y_pred)

print("\n" + "="*45)
print(f"HASIL EVALUASI MODEL CHATBOT")
print("="*45)
print(f"OVERALL ACCURACY: {acc * 100:.2f}%")
print("-"*45)
print("\nLAPORAN KLASIFIKASI PER KATEGORI:\n")
print(classification_report(y_true, y_pred))

# ==========================================
# 6. VISUALISASI UNTUK LAPORAN (Confusion Matrix)
# ==========================================
print("Menampilkan visualisasi Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=model.classes_, 
            yticklabels=model.classes_, 
            cmap='Greens')

plt.title('Confusion Matrix: Prediksi Intent Chatbot Banjir')
plt.ylabel('Tag Sebenarnya (Actual)')
plt.xlabel('Tag Prediksi Model')
plt.tight_layout()

# Simpan gambar untuk laporan
plt.savefig('confusion_matrix_hasil.png')
print("Gambar grafik disimpan sebagai 'confusion_matrix_hasil.png'")

# Tampilkan grafik di layar
plt.show()

# ==========================================
# 7. SIMPAN MODEL (PICKLE)
# ==========================================
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nSUKSES: Model 'chatbot_model.pkl' telah diperbarui!")
print("Silakan upload file .pkl terbaru ini ke GitHub untuk update di Streamlit Cloud.")