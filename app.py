import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob

# Налаштування сторінки
st.set_page_config(page_title="Twitter/X Auto-Analyzer", layout="wide")

st.title("📊 Аналіз Twitter-акаунтів (Автозавантаження з GitHub)")

# --- 1. АВТОМАТИЧНЕ ОТРИМАННЯ ДАНИХ ---
# ЗАМІНИ ЦЕ ПОСИЛАННЯ НА СВОЄ (натисни 'Raw' на файлі в GitHub і скопіюй URL)
repo_url = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/tweets_data.csv"

@st.cache_data # Кешування, щоб не качати файл при кожному кліку
def load_data(url):
    try:
        data = pd.read_csv(url)
        # Перетворення дати
        data['created_at'] = pd.to_datetime(data['created_at'])
        return data
    except Exception as e:
        st.error(f"Не вдалося завантажити файл з репозиторію: {e}")
        return None

df = load_data(repo_url)

if df is not None:
    # --- 2. ОБЧИСЛЕННЯ АКТИВНОСТІ ---
    st.sidebar.success("Дані успішно завантажені з GitHub!")
    
    df['engagement'] = df['likes'] + df['retweets']
    avg_engagement = df['engagement'].mean()
    
    col1, col2 = st.columns(2)
    col1.metric("Всього твітів", len(df))
    col2.metric("Середній Engagement", round(avg_engagement, 2))

    # --- 3. ВІЗУАЛІЗАЦІЯ ---
    st.header("📅 Динаміка публікацій")
    df['date'] = df['created_at'].dt.date
    daily_counts = df.groupby('date').size().reset_index(name='tweet_count')
    
    fig_line = px.line(daily_counts, x='date', y='tweet_count', markers=True,
                      title="Активність по днях")
    st.plotly_chart(fig_line, use_container_width=True)

    # --- 4. SENTIMENT ANALYSIS ---
    st.header("🧠 Sentiment Analysis")
    
    def get_sentiment(text):
        analysis = TextBlob(str(text))
        return "Positive" if analysis.sentiment.polarity > 0 else \
               "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

    df['sentiment'] = df['text'].apply(get_sentiment)
    sentiment_counts = df['sentiment'].value_counts().reset_index()

    fig_pie = px.pie(sentiment_counts, values='count', names='sentiment', 
                     color='sentiment',
                     color_discrete_map={'Positive':'#00CC96', 'Neutral':'#636EFA', 'Negative':'#EF553B'})
    st.plotly_chart(fig_pie)

    if st.checkbox("Показати таблицю даних"):
        st.dataframe(df)
else:
    st.warning("Перевір правильність посилання на Raw CSV файл у коді.")
