from __future__ import print_function

import os

import pandas as pd
from sklearn.cluster import KMeans
from bag_of_words import get_cleaned_books, get_book_names, FILE_FOLDER
from preprocessing import get_vocab_frame
from tf_idf import perform_tf_idf, get_terms_from_tf_idf
from sklearn.externals import joblib


def perform_kmeans():
    print("Start performing K-means clustering")

    tfidf_matr, tfidf_vect = perform_tf_idf()

    if os.path.isfile(os.path.join(FILE_FOLDER, 'book_cluster.pkl')):
        print("Detected existing model. Loading ... ")
        km = joblib.load(os.path.join(FILE_FOLDER, 'book_cluster.pkl'))
    else:
        num_clusters = get_num_clusters()

        km = KMeans(n_clusters=num_clusters)
        km.fit(tfidf_matr)

        joblib.dump(km, os.path.join(FILE_FOLDER, 'book_cluster.pkl'))

    return km, tfidf_vect


def get_num_clusters():
    num_clusters = 6
    return num_clusters


def get_clusters(km_model):
    return km_model.labels_.tolist()


def get_clusters_frame(km_model):
    clusters = get_clusters(km_model)

    book_titles = get_book_names()
    book_data = get_cleaned_books()

    books_for_clustering = {'title': book_titles, 'book_data': book_data, 'cluster': clusters}

    # print(books_for_clustering['title'])
    # print(len(books_for_clustering['title']))

    # print(type(clusters))
    # print(len(clusters))
    # print(clusters)

    # print(type(books_for_clustering['book_data']))
    # print(len(books_for_clustering['book_data']))

    frame = pd.DataFrame(books_for_clustering, index=[clusters], columns=['title', 'cluster'])

    print(frame['cluster'].value_counts())

    return frame


def get_centroids(km_model):
    return km_model.cluster_centers_.argsort()[:, ::-1]


def fancy_print(km_model, terms, frame):
    print("Top terms per cluster:")
    print()
    # sort cluster centers by proximity to centroid
    order_centroids = get_centroids(km_model)
    vocab_frame = get_vocab_frame()

    for i in range(get_num_clusters()):
        print("Cluster %d top 20 words:" % i, end='')

        for ind in order_centroids[i, :20]:  # replace 6 with n words per cluster
            print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'),
                  end=',')
        print()  # add whitespace

        print("Cluster %d titles:" % i, end='')
        if type(frame.ix[i]['title']) == str:
            print(' %s,' % frame.ix[i]['title'], end='')
        else:
            for title in frame.ix[i]['title'].values.tolist():
                print(' %s,' % title, end='')
        print()  # add whitespace
        print()  # add whitespace

    print()
    print()


if __name__ == '__main__':
    kmeans_model, tfidf_vectorizer = perform_kmeans()
    frame = get_clusters_frame(kmeans_model)
    terms = get_terms_from_tf_idf(tfidf_vectorizer)

    fancy_print(kmeans_model, terms, frame)
