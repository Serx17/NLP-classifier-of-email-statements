# train_model.py
# Этот скрипт обучает модель машинного обучения для классификации текстов.

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Для сохранения и загрузки модели
import nltk
from nltk.corpus import stopwords
import pymorphy3

# --- 1. ЗАГРУЗКА ДАННЫХ ---
print("1. Загружаю датасет...")
# Читаем наш сгенерированный CSV-файл
df = pd.read_csv('data/synthetic_claims_dataset.csv')
# Показываем первые 3 строки, чтобы убедиться, что всё загрузилось правильно
print("   Первые 3 строки датасета:")
print(df.head(3))
print(f"   Размер датасета: {df.shape}")

# --- 2. ПОДГОТОВКА И ОЧИСТКА ТЕКСТА ---
print("\n2. Начинаю очистку и подготовку текста...")

# Скачиваем стоп-слова для русского языка (слова-мусор: "и", "в", "на", "и т.д.")
nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')

# Инициализируем анализатор pymorphy3 для лемматизации
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    """
    Функция принимает сырой текст и возвращает его очищенную и лемматизированную версию.
    """
    # 2.1. Приводим к нижнему регистру и удаляем всё, кроме букв и пробелов
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text, flags=re.IGNORECASE)

    # 2.2. Разбиваем текст на отдельные слова (токенизация)
    words = text.split()

    # 2.3. Удаляем стоп-слова и приводим каждое слово к его нормальной форме (лемме)
    clean_words = []
    for word in words:
        if word not in russian_stopwords and len(word) > 2:  # Игнорируем короткие слова
            # Получаем нормальную форму слова (например, "бежал" -> "бежать")
            lemma = morph.parse(word)[0].normal_form
            clean_words.append(lemma)

    # 2.4. Собираем очищенные слова обратно в одну строку
    return " ".join(clean_words)

# Применяем функцию очистки ко всему столбцу с текстом
# Это может занять несколько секунд из-за сложной лемматизации
print("   Идет лемматизация... это может занять минуту.")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Покажем пример "до" и "после"
print("\n   Пример очистки текста:")
print("   Оригинал:    ", df['text'].iloc[0])
print("   После очистки:", df['cleaned_text'].iloc[0])

# --- 3. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ ---
print("\n3. Подготавливаю данные для обучения модели...")
# X - это наши признаки (очищенные тексты)
X = df['cleaned_text']
# y - это наша цель (категории, которые мы хотим предсказывать)
y = df['category']

# Делим данные на обучающую и тестовую выборку.
# 80% данных пойдет на обучение, 20% - на проверку точности.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Размер обучающей выборки: {X_train.shape[0]} примеров")
print(f"   Размер тестовой выборки:   {X_test.shape[0]} примеров")

# --- 4. ВЕКТОРИЗАЦИЯ (ПРЕВРАЩЕНИЕ ТЕКСТА В ЧИСЛА) ---
print("\n4. Превращаю текст в числа (векторизация)...")
# Создаем объект TfidfVectorizer.
# Он учитывает не только частоту слова, но и его важность в документе.
vectorizer = TfidfVectorizer(max_features=5000)  # Возьмем 5000 самых частых слов

# Обучаем векторизатор на обучающих данных и сразу преобразуем их в числа
X_train_vectorized = vectorizer.fit_transform(X_train)
# Преобразуем тестовые данные, используя уже обученный векторизатор (важно!)
X_test_vectorized = vectorizer.transform(X_test)

print(f"   Размерность признаков после векторизации: {X_train_vectorized.shape}")

# --- 5. ОБУЧЕНИЕ МОДЕЛИ ---
print("\n5. Обучаю модель LogisticRegression...")
# Создаем и обучаем модель на преобразованных данных
model = LogisticRegression(random_state=42, max_iter=1000)  # max_iter - чтобы модель гарантированно обучилась
model.fit(X_train_vectorized, y_train)

# --- 6. ПРОВЕРКА ТОЧНОСТИ МОДЕЛИ ---
print("\n6. Проверяю точность модели на тестовых данных...")
# Делаем прогнозы для тестовой выборки
y_pred = model.predict(X_test_vectorized)

# Считаем и выводим метрики точности
accuracy = accuracy_score(y_test, y_pred)
print(f"   Точность (Accuracy) модели: {accuracy:.2%}")
print("\n   Подробный отчет по классификации:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 7. СОХРАНЕНИЕ МОДЕЛИ ---
print("\n7. Сохраняю модель и векторизатор в файл...")
# Важно сохранить и модель, и векторизатор, чтобы использовать их в приложении.
# Мы создадим "pipeline", завернув их в один объект.
# В нашем случае мы просто сохраним их вместе в один файл.

# Создаем словарь со всеми необходимыми объектами
pipeline = {
    'vectorizer': vectorizer,
    'model': model
}

# Сохраняем весь pipeline в файл с помощью joblib
joblib.dump(pipeline, 'pipeline.pkl')
print("   Готово! Модель сохранена в файл 'pipeline.pkl'")

print("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ---")
print("Модель готова к использованию в Streamlit-приложении!") 
