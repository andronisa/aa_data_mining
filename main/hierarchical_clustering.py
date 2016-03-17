'''
*    Title: Document Clustering with Python
*    Author: Rose, B
*    Date: N/A
*    Code version: N/A
*    Availability: http://brandonrose.org/clustering#Tf-idf-and-document-similarity
*
'''
from __future__ import print_function

import os

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

from bag_of_words import get_book_names
from preprocessing import STATIC_FOLDER
from tf_idf import perform_tf_idf, get_similarity_matrix, get_kernel_types
from nlp import NLPHandler
from semantic import run_semantic_analysis


def get_hc_methods():
    return [
        'single',
        'complete',
        'average',
        'weighted',
        'centroid',
        'median',
        'ward',
    ]


def perform_hierarchical_clustering(use_nlp=False, use_nlp_sparse_matrix=False):
    print("Start performing Hierarchical clustering")

    if use_nlp_sparse_matrix:
        handler = NLPHandler()
        tfidf_matrix = handler.create_sparse_matrix()
    else:
        tfidf_matrix, tfidf_vectorizer = perform_tf_idf(use_nlp)

    titles = get_book_names()

    for kernel in get_kernel_types():
        for method in get_hc_methods():

            if use_nlp:
                nlp_tag = '_with_nlp_sparse_matr' if use_nlp_sparse_matrix else '_with_nlp'
            else:
                nlp_tag = '_without_nlp_sparse_matr' if use_nlp_sparse_matrix else '_without_nlp'

            title = "HC with: " + kernel + " _ " + method + nlp_tag
            print(title)

            cos_similarity_mtr = get_similarity_matrix(tfidf_matrix, kernel)
            plot_hierarchical_clustering(
                cos_similarity_mtr,
                titles,
                kernel_type=kernel,
                method_type=method,
                title=title,
                nlp_tag=nlp_tag
            )


def perform_semantic_hc():
    print("Start performing Hierarchical clustering with semantic analysis")
    titles = get_book_names()

    similarity_matrix = run_semantic_analysis()

    for method in get_hc_methods():
        title = "HC with semantic analysis and " + method
        print(title)

        plot_hierarchical_clustering(
            similarity_matrix,
            titles,
            kernel_type="semantic",
            method_type=method,
            title=title,
            nlp_tag=""
        )


def plot_hierarchical_clustering(cos_simil_matr, book_titles, kernel_type, method_type, title, nlp_tag):
    from scipy.cluster.hierarchy import ward, dendrogram, linkage

    plt.close('all')

    # define the linkage_matrix using ward clustering pre-computed distances
    # linkage_matrix = ward(cos_simil_matr)

    linkage_matrix = linkage(cos_simil_matr, method=method_type)

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    fig.canvas.set_window_title(title)

    ax = dendrogram(linkage_matrix, orientation="right", labels=book_titles)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout
    plt.show()

    plt.savefig(os.path.join(STATIC_FOLDER,
                             'hc_dendrogram' + '_' + kernel_type + '_' + method_type + nlp_tag + '.png'), dpi=200)
    plt.close()


if __name__ == '__main__':
    # perform_hierarchical_clustering(use_nlp=False, use_nlp_sparse_matrix=False)
    perform_semantic_hc()
