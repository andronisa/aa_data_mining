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


def perform_hierarchical_clustering():
    print("Start performing Hierarchical clustering")
    tfidf_matrix, tfidf_vectorizer = perform_tf_idf()

    titles = get_book_names()

    for kernel in get_kernel_types():
        print("HC with: " + kernel)

        cos_similarity_mtr = get_cosine_similarity(tfidf_matrix, kernel)
        plot_hierarchical_clustering(cos_similarity_mtr, titles, kernel_type=kernel)


def plot_hierarchical_clustering(cos_simil_matr, book_titles, kernel_type):
    from scipy.cluster.hierarchy import ward, dendrogram

    # define the linkage_matrix using ward clustering pre-computed distances
    linkage_matrix = ward(cos_simil_matr)

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=book_titles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout
    plt.show()

    plt.savefig(os.path.join(STATIC_FOLDER, 'hc_dendrogram' + '_' + kernel_type + '.png'), dpi=200)
    # save figure as ward_clusters

    plt.close()


if __name__ == '__main__':
    perform_hierarchical_clustering()
