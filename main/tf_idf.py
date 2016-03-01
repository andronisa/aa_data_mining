from sklearn.feature_extraction.text import TfidfVectorizer
from bag_of_words import get_cleaned_books


def perform_tf_idf():
    books = get_cleaned_books()

    # define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=5000,
                                       min_df=0.2, stop_words=None,
                                       use_idf=True, tokenizer=None, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(books)  # fit the vectorizer to synopses

    # print(tfidf_matrix.shape)
    # print(tfidf_matrix)

    return tfidf_matrix


if __name__ == '__main__':
    tfidf_matr = perform_tf_idf()
    print(tfidf_matr)
