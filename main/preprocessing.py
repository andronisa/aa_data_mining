from bs4 import BeautifulSoup
import os
import re
import nltk

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html'))


def page_to_words(raw_page):
    page_text = BeautifulSoup(raw_page).get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", page_text)

    # 3. Remove phrase 'OCR Output'
    cleaner_text = re.sub('OCR Output', '', letters_only)

    # 4. Convert to lower case, split into individual words
    words = cleaner_text.lower().split()

    # 5. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(nltk.corpus.stopwords.words("english"))

    # 6. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 7. Join the words back into one string separated by space and return the result.
    return " ".join(meaningful_words)


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

            with open(os.path.join(FILE_FOLDER, 'cleaned_books', book), 'w') as new_book_name:
                new_book_name.write(" ".join(book_text))
