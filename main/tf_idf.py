from sklearn.feature_extraction.text import TfidfVectorizer
from bag_of_words import get_cleaned_books
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


def perform_tf_idf():
    print("Getting cleaned books...")
    books = get_cleaned_books()
    max_features = 200000

    # define vectorizer parameters
    print("Setup TF-IDF Vectorizer")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=max_features,
                                       min_df=0.2, stop_words=None,
                                       use_idf=True, tokenizer=None, ngram_range=(1, 3))

    print("Perform TF-IDF on the books -- Max features = " + str(max_features))

    tfidf_matrix = tfidf_vectorizer.fit_transform(books)  # fit the vectorizer to books

    return tfidf_matrix, tfidf_vectorizer


def get_terms_from_tf_idf(tfidf_vectorizer):
    terms = tfidf_vectorizer.get_feature_names()

    return terms


def get_cosine_similarity(tfidf_matr):
    dist = 1 - cosine_similarity(tfidf_matr, tfidf_matr)

    return dist


if __name__ == '__main__':
    tfidf_matr, tfidf_vectorizer = perform_tf_idf()
    # print(tfidf_matr.shape)
    # print(tfidf_matr)
    #
    # tfidf_terms = get_terms_from_tf_idf(tfidf_vectorizer)
    # print(tfidf_terms)

    cos_sim = get_cosine_similarity(tfidf_matr)
    print(type(cos_sim))
    print(cos_sim)
