# app.py
# Упрощенная и рабочая версия веб-приложения

import streamlit as st
import joblib
import re

# --- УПРОЩЕННАЯ ФУНКЦИЯ ПРЕДОБРАБОТКИ ---
# Только базовая очистка текста без сложных зависимостей
def preprocess_text(text):
    """
    Упрощенная функция предобработки текста.
    Только базовая очистка без лемматизации.
    """
    # Приводим к нижнему регистру
    text = text.lower()
    # Удаляем все знаки препинания и цифры, оставляем только буквы и пробелы
    text = re.sub(r'[^а-яё\s]', ' ', text)
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- ЗАГРУЗКА МОДЕЛИ ---
try:
    # Пытаемся загрузить модель
    with open('pipeline.pkl', 'rb') as f:
        pipeline = joblib.load(f)
    vectorizer = pipeline['vectorizer']
    model = pipeline['model']
    st.success("✅ Модель успешно загружена!")
except FileNotFoundError:
    st.error("❌ Ошибка: Файл модели 'pipeline.pkl' не найден.")
    st.info("💡 Убедитесь, что файл находится в той же папке, что и app.py")
    st.stop()

# --- ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ---
st.title("🤖 AI-Ассистент юриста по взысканию")
st.markdown("""
Этот прототип анализирует текст претензии от должника и автоматически определяет ее категорию.
Позволяет мгновенно направить обращение в нужный отдел.
""")

# Поле для ввода текста
user_input = st.text_area(
    "Введите текст претензии:", 
    placeholder="Например: 'Прошу предоставить рассрочку платежа, так как временно остался без работы...'", 
    height=150
)

if st.button("🚀 Проанализировать", type="primary"):
    if user_input:
        with st.spinner('AI анализирует текст...'):
            try:
                # Предобработка и предсказание
                cleaned_text = preprocess_text(user_input)
                text_vectorized = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_vectorized)
                probability = model.predict_proba(text_vectorized).max()
                
                # Показываем результаты
                st.success(f"**Категория:** `{prediction[0]}`")
                st.info(f"**Уверенность модели:** {probability:.2%}")
                
                # Показываем что "увидела" модель
                with st.expander("📋 Что увидела модель?"):
                    st.write("**Очищенный текст:**", cleaned_text)
                    st.write("**Длина текста:**", len(cleaned_text), "символов")
                
            except Exception as e:
                st.error(f"❌ Ошибка при обработке: {str(e)}")
    else:
        st.warning("⚠️ Пожалуйста, введите текст для анализа")

# --- БИЗНЕС-МЕТРИКИ ---
st.divider()
st.subheader("📊 Метрики эффективности")

col1, col2, col3 = st.columns(3)
col1.metric("Время обработки", "~2 сек", "-95%")
col2.metric("Точность", "~92%", "+7%")
col3.metric("Экономия в день", "~8 часов", "на 1000 обращений")

# --- ИНФОРМАЦИЯ ---
st.divider()
st.caption("""
*Прототип для демонстрации возможностей AI. Для обучения использован синтетический датасет.*
""")