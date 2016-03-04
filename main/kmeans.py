from __future__ import print_function

import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.externals import joblib

from bag_of_words import get_cleaned_books, get_book_names, FILE_FOLDER
from preprocessing import get_vocab_frame, STATIC_FOLDER
from tf_idf import perform_tf_idf, get_terms_from_tf_idf, get_cosine_similarity, get_kernel_types


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

    return km, tfidf_matr, tfidf_vect


def get_num_clusters():
    num_clusters = 8
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


def mds(cos_simil_mtr):
    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(cos_simil_mtr)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]
    print()

    return xs, ys


def plot_clusters(clusters, book_titles, xs, ys):
    # set up cluster names using a dict
    cluster_names = {0: 'First Cluster',
                     1: 'Second Cluster',
                     2: 'Third Cluster',
                     3: 'Fourth Cluster',
                     4: 'Fifth Cluster',
                     5: 'Sixth Cluster',
                     6: 'Eleventh Cluster',
                     7: 'Eighth Cluster'}

    # create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=book_titles))

    # group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    i = 0
    for name, group in groups:
        i += 1
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name], color=cm.jet(1. * i / get_num_clusters()),
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='off',  # ticks along the bottom edge are off
            top='off',  # ticks along the top edge are off
            labelleft='off')

    # ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

    plt.show()  # show the plot

    plt.savefig(os.path.join(STATIC_FOLDER, '2d_clusters.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    kmeans_model, tfidf_matrix, tfidf_vectorizer = perform_kmeans()

    frame = get_clusters_frame(kmeans_model)
    terms = get_terms_from_tf_idf(tfidf_vectorizer)
    fancy_print(kmeans_model, terms, frame)

    for kernel in get_kernel_types():
        print("HC with: " + kernel)

        cos_similarity_mtr = get_cosine_similarity(tfidf_matrix, kernel)
        mds_xs, mds_ys = mds(cos_similarity_mtr)

        clusters = get_clusters(kmeans_model)
        titles = get_book_names()

        plot_clusters(clusters, titles, mds_xs, mds_ys)
