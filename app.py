import streamlit as st
import pandas as pd
import requests
import pdfplumber
import matplotlib.pyplot as plt
import re
from io import BytesIO
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. Пряме посилання на файл у репозиторії (Raw-посилання)
FILE_URL = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/document.txt"

st.set_page_config(page_title="Текстова аналітика", layout="wide")
st.title("📄 Автоматична текстова аналітика")

# Функція завантаження
@st.cache_data
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Помилка завантаження: {e}")
        return None

# Функція очищення тексту
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# --- Процес завантаження ---
file_data = fetch_data(FILE_URL)

if file_data:
    # Витягування тексту (PDF або TXT)
    if FILE_URL.lower().endswith(".pdf"):
        with pdfplumber.open(BytesIO(file_data)) as pdf:
            raw_text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        raw_text = file_data.decode("utf-8")

    if raw_text.strip():
        cleaned_text = clean_text(raw_text)
        words = cleaned_text.split()

        # --- 1. Частота слів та Barplot ---
        st.header("📊 Частота слів")
        word_counts = pd.Series(words).value_counts().head(15)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Топ-15 слів:**")
            st.dataframe(word_counts, use_container_width=True)
        with col2:
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            word_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
            ax1.set_title("Найчастіші слова")
            plt.xticks(rotation=45)
            st.pyplot(fig1)

        # --- 2. WordCloud ---
        st.divider()
        st.header("☁️ Хмара слів (WordCloud)")
        wc = WordCloud(width=1000, height=500, background_color='white').generate(cleaned_text)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wc, interpolation='bilinear')
        ax2.axis('off')
        st.pyplot(fig2)

        # --- 3. Тематичне моделювання (LDA) ---
        st.divider()
        st.header("🤖 Тематичне моделювання (LDA)")
        
        n_topics = st.slider("Кількість тем", 2, 5, 3)
        
        # Використовуємо CountVectorizer для підготовки даних
        vectorizer = CountVectorizer(stop_words='english', max_features=500)
        # Для LDA потрібен список документів, тому кладемо текст у список
        data_vectorized = vectorizer.fit_transform([cleaned_text])
        
        if data_vectorized.shape[1] > 5:
            lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda_model.fit(data_vectorized)
            
            # Вивід тем
            feature_names = vectorizer.get_feature_names_out()
            cols = st.columns(n_topics)
            
            for i, topic in enumerate(lda_model.components_):
                with cols[i]:
                    st.success(f"**Тема №{i+1}**")
                    top_idx = topic.argsort()[-7:][::-1]
                    top_words = [feature_names[idx] for idx in top_idx]
                    st.write(" • " + "\n • ".join(top_words))
        else:
            st.info("Текст занадто короткий для якісного виділення тем.")

    else:
        st.error("Файл порожній.")
else:
    st.warning("Чекаю на завантаження файлу з GitHub...")
