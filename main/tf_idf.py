from sklearn.feature_extraction.text import TfidfVectorizer
from bag_of_words import get_cleaned_books
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel
from nlp import NLPHandler


def get_kernel_types():
    return [
        'cosine_similarity',
        'linear_kernel',
        'polynomial_kernel',
        'sigmoid_kernel',
        'rbf_kernel',
        'laplacian_kernel',
    ]


def perform_tf_idf(use_nlp=False):
    print("Getting cleaned books...")

    gram_range = None

    if use_nlp:
        handler = NLPHandler()
        books = handler.parse_nlp_results()
    else:
        books = get_cleaned_books()

    max_features = 50000

    # define vectorizer parameters
    print("Setup TF-IDF Vectorizer")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=max_features,
                                       min_df=0.2, stop_words=None,
                                       use_idf=True, tokenizer=None, ngram_range=gram_range)

    print("Perform TF-IDF on the books -- Max features = " + str(max_features) + " - n-grams: " + str(gram_range))

    tfidf_matrix = tfidf_vectorizer.fit_transform(books)  # fit the vectorizer to books
    print(tfidf_matrix.shape)

    return tfidf_matrix, tfidf_vectorizer


def get_terms_from_tf_idf(tfidf_vectorizer):
    terms = tfidf_vectorizer.get_feature_names()

    return terms


def get_kernel(kernel_name):
    options = {
        'cosine_similarity': cosine_similarity,
        'linear_kernel': linear_kernel,
        'polynomial_kernel': polynomial_kernel,
        'sigmoid_kernel': sigmoid_kernel,
        'rbf_kernel': rbf_kernel,
        'laplacian_kernel': laplacian_kernel
    }

    return options[kernel_name]


def get_similarity_matrix(tfidf_matr, kernel):
    dist = 1 - get_kernel(kernel)(tfidf_matr)

    return dist


if __name__ == '__main__':
    tfidf_matr, tfidf_vectorizer = perform_tf_idf()
    # print(tfidf_matr.shape)
    # print(tfidf_matr)
    #
    # tfidf_terms = get_terms_from_tf_idf(tfidf_vectorizer)
    # print(tfidf_terms)

    cos_sim = get_similarity_matrix(tfidf_matr)
    print(type(cos_sim))
    print(cos_sim)
