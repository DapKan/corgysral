import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from datetime import datetime

# Налаштування сторінки
st.set_page_config(page_title="Twitter/X Analyzer", layout="wide")

st.title("📊 Веб-додаток «Аналіз Twitter / X-аккаунтів»")

# --- 1. ОТРИМАННЯ ДАНИХ ---
st.sidebar.header("Завантаження даних")
uploaded_file = st.sidebar.file_uploader("Завантажте CSV-файл із твітами", type=["csv"])

# Приклад структури CSV, якщо файл не завантажено
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Будь ласка, завантажте CSV. Очікувані колонки: 'text', 'created_at', 'retweets', 'likes'.")
    # Демо-дані для демонстрації роботи
    data = {
        'text': [
            "I love this new update! So helpful.", 
            "This is the worst service ever.", 
            "Just okay, nothing special.", 
            "Absolutely amazing experience!",
            "I'm so frustrated with the bugs."
        ],
        'created_at': ['2023-10-01', '2023-10-02', '2023-10-02', '2023-10-03', '2023-10-04'],
        'retweets': [10, 5, 2, 20, 1],
        'likes': [50, 10, 5, 100, 2]
    }
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'])

# --- 2. ОБЧИСЛЕННЯ АКТИВНОСТІ ТА ENGAGEMENT ---
st.header("📈 Метрики залученості (Engagement)")

# Engagement = (Likes + Retweets)
df['engagement'] = df['likes'] + df['retweets']
avg_engagement = df['engagement'].mean()
total_tweets = len(df)

col1, col2 = st.columns(2)
col1.metric("Загальна кількість твітів", total_tweets)
col2.metric("Середній Engagement", round(avg_engagement, 2))

# --- 3. ВІЗУАЛІЗАЦІЯ ДИНАМІКИ ---
st.header("📅 Динаміка публікацій")

# Групування за датами
df['date'] = pd.to_datetime(df['created_at']).dt.date
daily_counts = df.groupby('date').size().reset_index(name='tweet_count')

fig_line = px.line(daily_counts, x='date', y='tweet_count', 
                  title="Кількість твітів по днях",
                  labels={'tweet_count': 'Кількість твітів', 'date': 'Дата'})
st.plotly_chart(fig_line, use_container_width=True)

# --- 4. SENTIMENT ANALYSIS (Аналіз тональності) ---
st.header("🧠 Sentiment Analysis")

def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['text'].apply(get_sentiment)
sentiment_counts = df['sentiment'].value_counts().reset_index()

fig_pie = px.pie(sentiment_counts, values='count', names='sentiment', 
                 title="Розподіл настроїв аудиторії",
                 color='sentiment',
                 color_discrete_map={'Positive':'green', 'Neutral':'gray', 'Negative':'red'})
st.plotly_chart(fig_pie)

# Відображення таблиці даних
if st.checkbox("Показати сирі дані"):
    st.write(df)
