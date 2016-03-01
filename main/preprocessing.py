from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.stem.snowball import SnowballStemmer

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html'))


def page_to_words(raw_page):
    # Swap <br> tags with newline character
    raw_page = re.sub('<br\s*?>', '\n', raw_page)

    page_text = BeautifulSoup(raw_page).get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in page_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", text, 0, re.UNICODE)

    # 3. Remove phrase 'OCR Output'
    cleaner_text = re.sub('OCR Output', " ", letters_only)

    # 4. Convert to lower case, split into individual words
    words = cleaner_text.lower().split()

    # 5. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(nltk.corpus.stopwords.words("english"))

    # 6. Remove stop words
    meaningful_words = [w for w in words if w not in stops]

    # 7. Join the words back into one string separated by space and tokenize/stem
    cleaned_text = " ".join(meaningful_words)
    final_book_text = tokenize_and_stem(cleaned_text)

    return final_book_text


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")

    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]

    return " ".join(stems)


# def tokenize_only(text):
#     # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
#     tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
#     filtered_tokens = []
#     # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
#     for token in tokens:
#         if re.search('[a-zA-Z]', token):
#             filtered_tokens.append(token)
#     return filtered_tokens


def get_books_structure(root_directory):
    dir = {}
    rootdir = root_directory.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1

    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict.fromkeys(files)
        parent = reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir


def recreate_books():
    for parent_dir, books in get_books_structure(FILE_FOLDER).iteritems():
        for book, pages in books.iteritems():
            if book == 'cleaned_books':
                continue

            print("Processing Book: " + book)
            print("")

            book_text = []
            counter = 0
            for page in sorted(pages):
                counter += 1
                if counter % 100 == 0:
                    print("Processed " + str(counter) + " pages")

                with open(os.path.join(FILE_FOLDER, book, page), 'r') as f:
                    data = f.read()
                    book_text.append(page_to_words(data))

            print("")

            with open(os.path.join(FILE_FOLDER, 'cleaned_books', book), 'w+') as new_book_name:
                new_book_name.write(" ".join(book_text))


if __name__ == '__main__':
    recreate_books()
