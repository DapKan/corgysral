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

# 1. Пряме посилання на Raw-файл (GitHub)
FILE_URL = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/document.txt"

st.set_page_config(page_title="Текстова аналітика", layout="wide")
st.title("📄 Текстова аналітика документа")

@st.cache_data
def fetch_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if url.lower().endswith(".pdf"):
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        return response.content.decode("utf-8")
    except:
        return None

def clean_text(text):
    text = text.lower()
    # Видаляємо все, крім літер та пробілів
    text = re.sub(r'[^a-zа-яіїєґ\s]', '', text)
    return text

# Завантаження
raw_text = fetch_text(FILE_URL)

if raw_text:
    cleaned_text = clean_text(raw_text)
    words = cleaned_text.split()
    
    # --- 1. Частота слів та BARPLOT ---
    st.header("📊 Аналіз частоти слів")
    word_counts = pd.Series(words).value_counts().head(15)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Топ-15 слів:**")
        st.dataframe(word_counts, use_container_width=True)
    with col2:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        word_counts.plot(kind='bar', ax=ax1, color='teal', edgecolor='black')
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    # --- 2. WordCloud (Хмара слів) ---
    st.divider()
    st.header("☁️ Візуалізація: Хмара слів")
    wc = WordCloud(width=1000, height=500, background_color='white', colormap='viridis').generate(cleaned_text)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.imshow(wc, interpolation='bilinear')
    ax2.axis('off')
    st.pyplot(fig2)

    # --- 3. Тематичне моделювання (LDA) ---
    st.divider()
    st.header("🤖 Розподіл за темами (LDA)")
    
    n_topics = st.slider("Оберіть кількість тем", 2, 5, 2)
    
    # Щоб LDA краще працював на одному документі, ми розіб'ємо його на речення
    sentences = raw_text.split('.') 
    sentences = [clean_text(s) for s in sentences if len(s) > 10]

    if len(sentences) > 2:
        vectorizer = CountVectorizer(stop_words='english', max_features=500)
        dtm = vectorizer.fit_transform(sentences)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        
        words_list = vectorizer.get_feature_names_out()
        cols = st.columns(n_topics)
        
        for i, topic in enumerate(lda.components_):
            with cols[i]:
                st.success(f"**Тема №{i+1}**")
                # Беремо топ-7 слів для кожної теми
                top_idx = topic.argsort()[-7:][::-1]
                for idx in top_idx:
                    st.write(f"- {words_list[idx]}")
    else:
        st.info("Додайте більше речень у файл для якісного розподілу тем.")
else:
    st.error("Файл не знайдено за посиланням.")
