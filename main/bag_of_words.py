'''
*    Title: Part 1: For Beginners - Bag of Words
*    Author: Kaggle
*    Date: Tue 9 Dec 2014
*    Code version: N/A
*    Availability: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
*
'''

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html', 'cleaned_books'))


def get_cleaned_books(type_of_book='stemmed'):
    books = []

    for root, dirs, clean_books in sorted(os.walk(FILE_FOLDER)):
        for book in sorted(clean_books):
            # print(book) # print book names
            if type_of_book in book:
                with open(os.path.join(root, book)) as f:
                    data = f.read()
                    books.append(data)
    return books


def get_book_names(type_of_book='stemmed'):
    book_names = []

    for root, dirs, book_files in sorted(os.walk(FILE_FOLDER)):
        for book in sorted(book_files):
            if type_of_book in book:
                book_name = book.replace("_".join(["", type_of_book]), "")
                book_names.append(book_name)

    return book_names


def create_bag_of_words(books=list()):
    print "Creating the bag of words...\n"

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.

    train_data_features = vectorizer.fit_transform(books)

    # Numpy arrays are easy to work with, so convert the result to an array
    train_data_features = train_data_features.toarray()

    # print type of result
    # print(type(train_data_features))

    # print all vectors of words
    # print(train_data_features)

    # print size of our vector
    # print train_data_features.shape

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()

    # print(str(len(vocab)) + " unique words in all the books")
    # print(vocab)

    # write all the vocabulary without using stemming in a file
    # with open(os.path.join(FILE_FOLDER, 'vocab.txt'), 'w+') as vocab_file:
    #     vocab_file.write(" ".join(vocab))

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    dist = sorted(dist, reverse=True)

    # For each, print the vocabulary word and the number of times it appears in the training set
    for tag, count in zip(vocab, dist):
        print count, tag


if __name__ == '__main__':
    cleaned_books = get_cleaned_books()
    create_bag_of_words(books=cleaned_books)
