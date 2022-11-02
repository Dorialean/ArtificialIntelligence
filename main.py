from sklearn.datasets import make_regression, make_classification, make_blobs
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def naive_bayess():
    """Наивный Байесовскй классификатор"""
    import nltk
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    # Чтение датасета
    df = pd.read_table('datasets/SMSSpamCollection',
                       sep='\t',
                       header=None,
                       names=['label', 'message'])
    """
        Если открыть датасет, то можно увидеть, что там всего
         2 типа меток в начале строки [ham и spam], обозначим
         ham как 0 и spam как 1
    """
    df['label'] = df.label.map({'ham': 0, 'spam': 1})
    # Избавление от шума, приведение к нижнему регистру + удаление лишних символов при помощи Regex
    df['message'] = df.message.map(lambda x: x.lower())
    df['message'] = df.message.str.replace('[^\w\s]', '')
    # Для того чтобы привести строки в понятные компу цифры используется
    # токенизация, которую предоставляет NLP библиотека NLTK
    # Stemmer из NLTK тут нормализует слова, сложна и непонятно, но вот как он работает "http://snowball.tartarus.org/algorithms/porter/stemmer.html"
    df['message'] = df['message'].apply(nltk.word_tokenize)
    stemmer = PorterStemmer()
    df['message'] = df['message'].apply(lambda x: [stemmer.stem(y) for y in x])
    # Добавление шаблонности, чтобы можно было спарсить эти строки
    df['message'] = df['message'].apply(lambda x: ' '.join(x))

    """CountVectorizer преобразовывает входной текст в матрицу,
     значениями которой, являются количества вхождения данного ключа(слова)
     в текст"""
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(df['message'])

    # tf-idf - считает насколько важны слова по формуле:
    # TF[Term Frequency](t): сколько_раз_t_в_тексте / всего_слов , где t - слово
    # IDF[Inverse Document Frequency](t): log_e(всего_слов / сколько_раз_t_в_тексте)
    """
    ПРИМЕР: 
    Рассмотрим документ, содержащий 100 слов, в котором слово кошка встречается 3 раза. Тогда термин 
    частота (т. е. tf) для кошки равен (3/100) = 0,03. Теперь предположим, что у нас есть 10 миллионов документов, 
    и слово «кошка» встречается в тысяче из них. Затем обратная частота документа (т.е. idf) рассчитывается как log(
    10 000 000 / 1000) = 4. Таким образом, вес Tf-idf является произведением этих величин: 0,03 * 4 = 0,12. 
    """
    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)

    # Склёрн сам создаёт трениров
    X_train, X_test, y_train, y_test = train_test_split(counts, df['label'], test_size=0.1, random_state=69)

    # Обучение модели на x_train - сообщения и y_train - их метка
    model = MultinomialNB().fit(X_train, y_train)

    # Предсказывание метки на x_test - тестовой выборке сгенерированной выше
    predicted = model.predict(X_test)

    # Среднее арифметическое - показывает насколько % модель "угадала" значения
    print(np.mean(predicted == y_test))


def main():
    naive_bayess()


if __name__ == '__main__':
    main()
