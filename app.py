import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# 1. Налаштування сторінки
st.set_page_config(page_title="Аналіз кіберінцидентів", layout="wide")
st.title("🛡️ Аналіз кіберінцидентів")

# 2. Завантаження або створення тестових даних
# Оскільки в завданні вказано CSV, додаємо можливість завантаження
uploaded_file = st.sidebar.file_uploader("Завантажте CSV файл", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    # Створюємо демонстраційні дані, якщо файл не завантажено
    data = {
        'дата': pd.date_range(start='2020-01-01', periods=100, freq='M'),
        'тип атаки': ['Phishing', 'DDoS', 'Malware', 'SQL Injection'] * 25,
        'сектор': ['Government', 'Finance', 'Healthcare', 'Tech', 'Energy'] * 20,
        'втрати': [1000, 5000, 15000, 2000, 8000, 45000, 3000, 12000] * 12 + [5000, 1000, 2000, 3000]
    }
    df = pd.DataFrame(data)
    st.info("Використовуються демонстраційні дані. Завантажте свій CSV у бічній панелі.")

# Перетворення дати
df['дата'] = pd.to_datetime(df['дата'])
df['рік'] = df['дата'].dt.year

# 3. Фільтри (Бічна панель)
st.sidebar.header("Фільтрація")
selected_year = st.sidebar.multiselect("Оберіть рік", options=df['рік'].unique(), default=df['рік'].unique())
selected_type = st.sidebar.multiselect("Оберіть тип атаки", options=df['тип атаки'].unique(), default=df['тип атаки'].unique())

filtered_df = df[(df['рік'].isin(selected_year)) & (df['тип атаки'].isin(selected_type))]

# 4. Статистика атак за секторами
st.header("📊 Статистика атак за секторами")
if not filtered_df.empty:
    sector_stats = filtered_df['сектор'].value_range().value_counts() if 'сектор' in filtered_df else filtered_df.groupby('сектор').size()
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(filtered_df)
    with col2:
        fig, ax = plt.subplots()
        filtered_df['сектор'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel("Кількість інцидентів")
        ax.set_xlabel("Сектор")
        st.pyplot(fig)
else:
    st.warning("Немає даних для відображення за обраними фільтрами.")

# 5. Кластеризація (K-Means)
st.header("🤖 Кластеризація інцидентів (K-Means)")

if len(filtered_df) >= 3:
    # Підготовка даних для кластеризації
    # Оскільки K-Means працює з числами, кодуємо категоріальні ознаки
    le_type = LabelEncoder()
    le_sector = LabelEncoder()
    
    cluster_df = filtered_df.copy()
    cluster_df['тип_encoded'] = le_type.fit_transform(cluster_df['тип атаки'])
    cluster_df['сектор_encoded'] = le_sector.fit_transform(cluster_df['сектор'])
    
    # Вибираємо ознаки для кластеризації: Тип, Сектор та Втрати
    features = cluster_df[['тип_encoded', 'сектор_encoded', 'втрати']]
    
    n_clusters = st.slider("Кількість кластерів", 2, 5, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_df['кластер'] = kmeans.fit_predict(features)
    
    # Візуалізація кластерів
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(cluster_df['сектор_encoded'], cluster_df['втрати'], 
                          c=cluster_df['кластер'], cmap='viridis', s=100)
    ax2.set_xlabel("Сектор (encoded)")
    ax2.set_ylabel("Фінансові втрати")
    plt.colorbar(scatter, label='Номер кластера')
    st.pyplot(fig2)
    
    st.write("Результати кластеризації (перші 10 рядків):")
    st.table(cluster_df[['дата', 'тип атаки', 'сектор', 'втрати', 'кластер']].head(10))
else:
    st.error("Недостатньо даних для проведення кластеризації (мінімум 3 інциденти).")
