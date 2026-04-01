import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# 1. Налаштування сторінки
st.set_page_config(page_title="Аналіз кіберінцидентів", layout="wide")
st.title("🛡️ Аналіз кіберінцидентів")

# 2. Завантаження або створення даних
uploaded_file = st.sidebar.file_uploader("Завантажте CSV файл", type="csv")

if uploaded_file is not None:
    try:
        # Спроба прочитати файл з різними кодуваннями
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Помилка при читанні файлу: {e}")
        st.stop()
else:
    # Виправлено: freq='ME' для сумісності з новими версіями Pandas
    data = {
        'дата': pd.date_range(start='2020-01-01', periods=100, freq='ME'),
        'тип атаки': (['Phishing', 'DDoS', 'Malware', 'SQL Injection'] * 25),
        'сектор': (['Government', 'Finance', 'Healthcare', 'Tech', 'Energy'] * 20),
        'втрати': ([1000, 5000, 15000, 2000, 8000, 45000, 3000, 12000] * 12 + [5000, 1000, 2000, 3000])
    }
    df = pd.DataFrame(data)
    st.info("Використовуються демонстраційні дані. Завантажте свій CSV у бічній панелі.")

# Перетворення дати та створення колонки "рік"
df['дата'] = pd.to_datetime(df['дата'])
df['рік'] = df['дата'].dt.year

# 3. Фільтри (Бічна панель)
st.sidebar.header("⚙️ Фільтрація")
years = sorted(df['рік'].unique())
selected_year = st.sidebar.multiselect("Оберіть рік", options=years, default=years)

attack_types = sorted(df['тип атаки'].unique())
selected_type = st.sidebar.multiselect("Оберіть тип атаки", options=attack_types, default=attack_types)

# Застосування фільтрів
filtered_df = df[(df['рік'].isin(selected_year)) & (df['тип атаки'].isin(selected_type))]

# 4. Статистика атак за секторами
st.header("📊 Статистика атак за секторами")

if not filtered_df.empty:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Дані")
        st.dataframe(filtered_df, use_container_width=True)
    
    with col2:
        st.subheader("Графік")
        fig, ax = plt.subplots()
        # Рахуємо кількість інцидентів на сектор
        sector_counts = filtered_df['сектор'].value_counts()
        sector_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_ylabel("Кількість інцидентів")
        ax.set_xlabel("Сектор")
        plt.xticks(rotation=45)
        st.pyplot(fig)
else:
    st.warning("Дані відсутні для обраних фільтрів.")

# 5. Кластеризація (K-Means)
st.divider()
st.header("🤖 Кластеризація інцидентів (K-Means)")

if len(filtered_df) >= 5:
    # Підготовка даних: кодуємо текст у числа для алгоритму
    le_type = LabelEncoder()
    le_sector = LabelEncoder()
    
    cluster_prep = filtered_df.copy()
    cluster_prep['тип_n'] = le_type.fit_transform(cluster_prep['тип атаки'])
    cluster_prep['сектор_n'] = le_sector.fit_transform(cluster_prep['сектор'])
    
    # Використовуємо Сектор та Втрати для кластеризації
    X = cluster_prep[['сектор_n', 'втрати']]
    
    n_clusters = st.select_slider("Оберіть кількість кластерів", options=[2, 3, 4, 5], value=3)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    filtered_df['кластер'] = kmeans.fit_predict(X)
    
    # Візуалізація кластерів
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        filtered_df['сектор'], 
        filtered_df['втрати'], 
        c=filtered_df['кластер'], 
        cmap='viridis', 
        s=100, 
        edgecolors='black'
    )
    ax2.set_xlabel("Сектор")
    ax2.set_ylabel("Фінансові втрати")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.write("**Результат з розподілом по кластерах:**")
    st.dataframe(filtered_df.sort_values('кластер'), use_container_width=True)
else:
    st.info("Додайте більше даних через фільтри для проведення кластеризації.")
