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

# 1. Пряме посилання на файл (Raw URL)
FILE_URL = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/document.txt"

st.set_page_config(page_title="Текстовий аналізатор", layout="wide")
st.title("📄 Текстова аналітика документів")

@st.cache_data
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if url.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        else:
            return response.content.decode("utf-8")
    except Exception as e:
        st.error(f"Помилка завантаження: {e}")
        return None

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

# --- Виконання ---
raw_text = fetch_text_from_url(FILE_URL)

if raw_text:
    cleaned_text = clean_text(raw_text)
    words = cleaned_text.split()
    
    # --- 1. Частота слів та BARPLOT ---
    st.header("📊 Частота вживання слів")
    word_counts = pd.Series(words).value_counts().head(15)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Топ-15 слів (Таблиця):**")
        st.dataframe(word_counts, use_container_width=True)
    
    with col2:
        st.write("**Графік частоти (Barplot):**")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        word_counts.plot(kind='bar', ax=ax1, color='teal', edgecolor='black')
        ax1.set_ylabel("Кількість повторень")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    # --- 2. Wordcloud ---
    st.divider()
    st.header("☁️ Хмара слів (Wordcloud)")
    wc = WordCloud(width=1000, height=500, background_color='white').generate(cleaned_text)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.imshow(wc, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

    # --- 3. Тематичне моделювання (LDA) ---
    st.divider()
    st.header("🤖 Тематичне моделювання (LDA)")
    
    n_topics = st.slider("Оберіть кількість тем", 2, 5, 2)
    
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    # LDA потребує список текстів, тому створюємо список з одного документа
    dtm = vectorizer.fit_transform([cleaned_text])
    
    if dtm.shape[1] > 10:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        
        words_list = vectorizer.get_feature_names_out()
        cols = st.columns(n_topics)
        
        for i, topic in enumerate(lda.components_):
            with cols[i]:
                st.info(f"**Тема №{i+1}**")
                top_words_idx = topic.argsort()[-8:][::-1]
                top_words = [words_list[idx] for idx in top_words_idx]
                st.write(" • " + "\n • ".join(top_words))
    else:
        st.warning("Текст занадто малий для LDA-аналізу.")
else:
    st.info("Перевірте посилання на файл у коді.")
