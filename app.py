import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

FILE_URL = "https://raw.githubusercontent.com/DapKan/corgysral/refs/heads/main/cyber_attacks.csv"

st.set_page_config(page_title="Аналіз кіберінцидентів", layout="wide")
st.title("🛡️ Аналіз кіберінцидентів (Auto-load)")

# Функція для завантаження даних з GitHub
@st.cache_data
def load_data_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Перевірка на помилки (напр. 404)
        return pd.read_csv(BytesIO(response.content))
    except Exception as e:
        st.error(f"Не вдалося завантажити файл з репозиторію: {e}")
        return None

# --- Завантаження даних ---
df = load_data_from_url(FILE_URL)

if df is not None:
    # Базова підготовка даних
    df['дата'] = pd.to_datetime(df['дата'])
    df['рік'] = df['дата'].dt.year

    # --- Фільтри у бічній панелі ---
    st.sidebar.header("⚙️ Фільтрація")
    
    all_years = sorted(df['рік'].unique())
    selected_years = st.sidebar.multiselect("Оберіть роки", options=all_years, default=all_years)
    
    all_types = sorted(df['тип атаки'].unique())
    selected_types = st.sidebar.multiselect("Оберіть типи атак", options=all_types, default=all_types)

    # Фільтрація датафрейму
    filtered_df = df[(df['рік'].isin(selected_years)) & (df['тип атаки'].isin(selected_types))]

    # --- Побудова статистики атак за секторами ---
    st.header("📊 Статистика атак за секторами")
    
    if not filtered_df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Відфільтровані дані:**")
            st.dataframe(filtered_df, use_container_width=True)
            
        with col2:
            st.write("**Розподіл за секторами:**")
            fig, ax = plt.subplots()
            sector_counts = filtered_df['сектор'].value_counts()
            sector_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
            ax.set_ylabel("Кількість інцидентів")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.warning("Дані відсутні для обраних фільтрів.")

    # --- Кластеризація інцидентів (K-Means) ---
    st.divider()
    st.header("🤖 Кластеризація інцидентів (K-Means)")
    
    if len(filtered_df) >= 3:
        # Підготовка ознак для кластеризації
        le_type = LabelEncoder()
        le_sector = LabelEncoder()
        
        cluster_df = filtered_df.copy()
        cluster_df['тип_n'] = le_type.fit_transform(cluster_df['тип атаки'])
        cluster_df['сектор_n'] = le_sector.fit_transform(cluster_df['сектор'])
        
        # Кластеризуємо за сектором та сумою втрат
        X = cluster_df[['сектор_n', 'втрати']]
        
        k = st.slider("Кількість кластерів (K)", 2, 5, 3)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_df['кластер'] = kmeans.fit_predict(X)
        
        # Візуалізація кластерів
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        scatter = ax2.scatter(
            cluster_df['сектор'], 
            cluster_df['втрати'], 
            c=cluster_df['кластер'], 
            cmap='viridis', 
            s=150, 
            edgecolors='white',
            alpha=0.8
        )
        ax2.set_title(f"Розподіл на {k} кластери")
        ax2.set_xlabel("Сектор")
        ax2.set_ylabel("Фінансові втрати")
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        
        st.write("**Таблиця з результатами кластеризації:**")
        st.dataframe(cluster_df[['дата', 'тип атаки', 'сектор', 'втрати', 'кластер']].sort_values('кластер'))
    else:
        st.info("Для кластеризації потрібно мінімум 3 записи.")

else:
    st.info("Будь ласка, перевірте посилання FILE_URL у коді.")
