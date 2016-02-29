import os
from sklearn.feature_extraction.text import CountVectorizer

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html', 'cleaned_books'))


def get_cleaned_books():
    books = []

    for root, dirs, files in os.walk(FILE_FOLDER):
        for file in files:
            # print book names
            print(file)

            with open(os.path.join(root, file)) as f:
                data = f.read()
                books.append(data)

    return books


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

    # print(type(train_data_features))
    print(train_data_features)


if __name__ == '__main__':
    cleaned_books = get_cleaned_books()
    create_bag_of_words(books=cleaned_books)
