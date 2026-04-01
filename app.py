import streamlit as st
import pandas as pd
import requests
import pdfplumber
from io import BytesIO, StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. Пряме посилання на ваш файл у репозиторії
# ЗАМІНІТЬ ЦЕ ПОСИЛАННЯ НА ВЛАСНЕ
FILE_URL = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/document.txt"

st.set_page_config(page_title="Авто-завантаження з GitHub", layout="wide")
st.title("📄 Аналітика файлу з репозиторію")

@st.cache_data # Кешуємо, щоб не завантажувати файл при кожному натисканні кнопки
def load_file_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        return None

# Завантаження
file_content = load_file_from_github(FILE_URL)

if file_content:
    # Визначаємо тип файлу за розширенням посилання
    if FILE_URL.endswith(".pdf"):
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages])
    else:
        # Для TXT файлів
        text = file_content.decode("utf-8")

    st.success(f"Файл успішно завантажено з GitHub!")
    
    # Вивід тексту (перші 500 символів)
    with st.expander("Переглянути текст файлу"):
        st.write(text[:1000] + "...")

    # Далі йде ваш код аналітики (WordCloud, частота і т.д.)
    st.header("☁️ Хмара слів")
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis('off')
    st.pyplot(fig)

else:
    st.error("Не вдалося завантажити файл. Перевірте посилання FILE_URL.")
