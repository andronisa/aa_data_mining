from bs4 import BeautifulSoup
import os
import re
import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from bag_of_words import get_cleaned_books

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html'))
STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))


def map_book_name(book):
    books = {
        'gap_2X5KAAAAYAAJ': 'WorksOfCorneliusTacitus-Book-1',
        'gap_9ksIAAAAQAAJ': 'PeloponnesianWar-Book-1',
        'gap_aLcWAAAAQAAJ': 'DeclineAndFallOfRomanEmpire-Book-1',
        'gap_Bdw_AAAAYAAJ': 'HistoryOfRome-Titus-Livius-Book-1',
        'gap_-C0BAAAAQAAJ': 'DictionaryOfGreekAndRomanGeography',
        'gap_CnnUAAAAMAAJ': 'JewishAntiquities-Book-1',
        'gap_CSEUAAAAYAAJ': 'DeclineAndFallOfRomanEmpire-Book-2',
        'gap_DhULAAAAYAAJ': 'TheDescriptionOfGreece-Book-9',
        'gap_dIkBAAAAQAAJ': 'HistoryOfRome-Book-1',
        'gap_DqQNAAAAYAAJ': 'HistoryOfRome-Book-2',
        'gap_fnAMAAAAYAAJ': 'PeloponnesianWar-Book-2',
        'gap_GIt0HMhqjRgC': 'DeclineAndFallOfRomanEmpire-Book-3',
        'gap_IlUMAQAAMAAJ': 'DeclineAndFallOfRomanEmpire-Book-4',
        'gap_m_6B1DkImIoC': 'HistoryOfRome-Titus-Livius-Book-2',
        'gap_MEoWAAAAYAAJ': 'WorksOfCorneliusTacitus-Book-2',
        'gap_ogsNAAAAIAAJ': 'JewishAntiquities-Book-2',
        'gap_pX5KAAAAYAAJ': 'WorksOfCorneliusTacitus-Book-3',
        'gap_RqMNAAAAYAAJ': 'HistoryOfRome-Book-3',
        'gap_TgpMAAAAYAAJ': 'JewishAntiquities-Book-3',
        'gap_udEIAAAAQAAJ': 'NaturalHistoryOfPliny-Book-1',
        'gap_VPENAAAAQAAJ': 'HistoryOfRome-Book-4',
        'gap_WORMAAAAYAAJ': 'WorksOfCorneliusTacitus-Book-4',
        'gap_XmqHlMECi6kC': 'DeclineAndFallOfRomanEmpire-Book-4',
        'gap_y-AvAAAAYAAJ': 'JewishAntiquities-Book-3',
    }

    return books[book]


def simple_page(raw_page):
    raw_page = re.sub('<br\s*?>', '\n', raw_page)
    page_text = BeautifulSoup(raw_page).get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in page_text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    letters_only = re.sub("[^a-zA-Z]", " ", text, 0, re.UNICODE)
    # 3. Remove phrase 'OCR Output'
    cleaner_text = re.sub('OCR Output', " ", letters_only)
    words = cleaner_text.lower().split()
    final_book_text = " ".join(words)

    return final_book_text


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
    final_book_text = " ".join(meaningful_words)

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

    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


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


def get_vocab_frame():
    totalvocab_stemmed = []
    totalvocab_tokenized = []

    for book_text in get_cleaned_books():
        stemmed_words = book_text.split()
        totalvocab_stemmed.extend(stemmed_words)
    for book_text in get_cleaned_books('tokenized'):
        tokenized_words = book_text.split()
        totalvocab_tokenized.extend(tokenized_words)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    return vocab_frame


def get_first_20_pages():
    for parent_dir, books in get_books_structure(FILE_FOLDER).iteritems():
        for book, pages in books.iteritems():
            if book == 'cleaned_books':
                continue

            print("Processing Book: " + book)
            print("")

            first_fifty_pages = []
            counter = 0
            for page in sorted(pages):
                counter += 1
                if counter == 20:
                    break

                with open(os.path.join(FILE_FOLDER, book, page), 'r') as f:
                    data = f.read()
                    page_text = simple_page(data)
                    first_fifty_pages.append(page_text)

            book_name = map_book_name(book)

            print("First Pages Only")
            with open(os.path.join(FILE_FOLDER, 'cleaned_books', book_name + "_first_pages"), 'w+') as new_book_name:
                new_book_name.write(" ".join(first_fifty_pages))


def recreate_books():
    for parent_dir, books in get_books_structure(FILE_FOLDER).iteritems():
        for book, pages in books.iteritems():
            if book == 'cleaned_books':
                continue

            print("Processing Book: " + book)
            print("")

            stemmed_book_pages = []
            tokenized_book_pages = []
            counter = 0
            for page in sorted(pages):
                counter += 1
                if counter % 100 == 0:
                    print("Processed " + str(counter) + " pages")

                with open(os.path.join(FILE_FOLDER, book, page), 'r') as f:
                    data = f.read()
                    page_text = page_to_words(data)
                    stemmed_book_pages.append(" ".join(tokenize_and_stem(page_text)))

                    tokenized_book_pages.append(" ".join(tokenize_only(page_text)))

            book_name = map_book_name(book)

            print("Stemmed only")
            with open(os.path.join(FILE_FOLDER, 'cleaned_books', book_name + "_stemmed"), 'w+') as new_book_name:
                new_book_name.write(" ".join(stemmed_book_pages))

            print("Tokenized only")
            with open(os.path.join(FILE_FOLDER, 'cleaned_books', book_name + "_tokenized"), 'w+') as new_book_name:
                new_book_name.write(" ".join(tokenized_book_pages))

                print("")


if __name__ == '__main__':
    recreate_books()
    get_first_20_pages()
