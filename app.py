import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pdfplumber
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Налаштування сторінки
st.set_page_config(page_title="Текстова аналітика", layout="wide")
st.title("📄 Текстова аналітика документів")

# Функція для очищення тексту
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Видаляємо пунктуацію
    text = re.sub(r'\d+', '', text)      # Видаляємо цифри
    return text

# 1. Завантаження файлу
uploaded_file = st.file_uploader("Завантажте документ (PDF або TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    raw_text = ""
    
    # Витягування тексту
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                raw_text += page.extract_text() + " "
    else:
        raw_text = str(uploaded_file.read(), "utf-8")

    if raw_text.strip():
        cleaned_text = clean_text(raw_text)
        words = cleaned_text.split()
        
        # 2. Частота слів
        st.header("📊 Частота слів")
        word_counts = pd.Series(words).value_counts().head(20)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Топ-20 слів:")
            st.dataframe(word_counts)
        
        with col2:
            fig, ax = plt.subplots()
            word_counts.plot(kind='barh', ax=ax, color='teal')
            ax.invert_yaxis()
            st.pyplot(fig)

        # 3. WordCloud і Barplot
        st.divider()
        st.header("☁️ Хмара слів")
        
        if len(words) > 0:
            wc = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

        # 4. Тематичне моделювання (LDA)
        st.divider()
        st.header("🤖 Тематичне моделювання (LDA)")
        
        n_topics = st.slider("Кількість тем для виділення", 2, 5, 3)
        
        if len(words) > 10:
            # Створюємо матрицю частот слів
            vectorizer = CountVectorizer(stop_words='english', max_features=1000)
            data_vectorized = vectorizer.fit_transform([cleaned_text])
            
            # LDA модель
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
            lda.fit(data_vectorized)
            
            # Вивід результатів
            words_list = vectorizer.get_feature_names_out()
            
            cols = st.columns(n_topics)
            for i, topic in enumerate(lda.components_):
                with cols[i]:
                    st.subheader(f"Тема №{i+1}")
                    # Отримуємо топ-10 слів для кожної теми
                    top_words_idx = topic.argsort()[-10:][::-1]
                    top_words = [words_list[idx] for idx in top_words_idx]
                    st.write(", ".join(top_words))
        else:
            st.warning("Тексту замало для моделювання тем.")
            
    else:
        st.error("Файл порожній або не вдалося зчитати текст.")
else:
    st.info("Будь ласка, завантажте файл для початку аналізу.")
