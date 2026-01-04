import streamlit as st
import json
import pickle
import random
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Chatbot Info Banjir", page_icon="🌊")

# 2. Load Data dan Model
@st.cache_resource # Agar model tidak di-load ulang setiap kali chat
def load_assets():
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return intents, model

intents, model = load_assets()

# 3. Inisialisasi Stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 4. Fungsi Prediksi
def get_response(user_input):
    # Preprocessing input user (harus sama dengan saat training)
    clean_input = stemmer.stem(user_input.lower())
    
    # Prediksi Tag menggunakan Model Naive Bayes secara langsung
    tag = model.predict([clean_input])[0]
    
    # Cari jawaban yang sesuai di intents.json
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
            
    return "Maaf, saya kurang mengerti. Bisa diulangi?"
# 5. Tampilan UI Streamlit
st.title("🌊 Chatbot Informasi Banjir")
st.markdown("Tanyakan hal seputar lokasi rawan, pencegahan, hingga kontak darurat banjir.")

# Inisialisasi history chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan chat lama
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat dari User
if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Ambil respon dari model
    response = get_response(prompt)

    # Tampilkan respon bot
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Gunakan path relatif agar terbaca di server
with open('chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)


