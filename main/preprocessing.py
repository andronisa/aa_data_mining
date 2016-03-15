from __future__ import division
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys
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

    # whole_book_tokens = []
    # for(each) page:
    # get the tokens
    # whole_book_tokens.append(tokens)
    # whole_book(str, txt) = " ".join(whole_book_tokens)
    # with open("book.txt", w+) as file:
    # file.write(whole_book)

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

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

# ######################### word similarity ##########################

# def get_best_synset_pair(word_1, word_2):
#     """ 
#     Choose the pair with highest path similarity among all pairs. 
#     Mimics pattern-seeking behavior of humans.
#     """
#     max_sim = -1.0
#     synsets_1 = wn.synsets(word_1)
#     synsets_2 = wn.synsets(word_2)
#     if len(synsets_1) == 0 or len(synsets_2) == 0:
#         return None, None
#     else:
#         max_sim = -1.0
#         best_pair = None, None
#         for synset_1 in synsets_1:
#             for synset_2 in synsets_2:
#                sim = wn.path_similarity(synset_1, synset_2)
#                if sim > max_sim:
#                    max_sim = sim
#                    best_pair = synset_1, synset_2
#         return best_pair

# def length_dist(synset_1, synset_2):
#     """
#     Return a measure of the length of the shortest path in the semantic 
#     ontology (Wordnet in our case as well as the paper's) between two 
#     synsets.
#     """
#     l_dist = sys.maxint
#     if synset_1 is None or synset_2 is None: 
#         return 0.0
#     if synset_1 == synset_2:
#         # if synset_1 and synset_2 are the same synset return 0
#         l_dist = 0.0
#     else:
#         wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
#         wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
#         if len(wset_1.intersection(wset_2)) > 0:
#             # if synset_1 != synset_2 but there is word overlap, return 1.0
#             l_dist = 1.0
#         else:
#             # just compute the shortest path between the two
#             l_dist = synset_1.shortest_path_distance(synset_2)
#             if l_dist is None:
#                 l_dist = 0.0
#     # normalize path length to the range [0,1]
#     return math.exp(-ALPHA * l_dist)

# def hierarchy_dist(synset_1, synset_2):
#     """
#     Return a measure of depth in the ontology to model the fact that 
#     nodes closer to the root are broader and have less semantic similarity
#     than nodes further away from the root.
#     """
#     h_dist = sys.maxint
#     if synset_1 is None or synset_2 is None: 
#         return h_dist
#     if synset_1 == synset_2:
#         # return the depth of one of synset_1 or synset_2
#         h_dist = max([x[1] for x in synset_1.hypernym_distances()])
#     else:
#         # find the max depth of least common subsumer
#         hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
#         hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
#         lcs_candidates = set(hypernyms_1.keys()).intersection(
#             set(hypernyms_2.keys()))
#         if len(lcs_candidates) > 0:
#             lcs_dists = []
#             for lcs_candidate in lcs_candidates:
#                 lcs_d1 = 0
#                 if hypernyms_1.has_key(lcs_candidate):
#                     lcs_d1 = hypernyms_1[lcs_candidate]
#                 lcs_d2 = 0
#                 if hypernyms_2.has_key(lcs_candidate):
#                     lcs_d2 = hypernyms_2[lcs_candidate]
#                 lcs_dists.append(max([lcs_d1, lcs_d2]))
#             h_dist = max(lcs_dists)
#         else:
#             h_dist = 0
#     return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
#         (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
# def word_similarity(word_1, word_2):
#     synset_pair = get_best_synset_pair(word_1, word_2)
#     return (length_dist(synset_pair[0], synset_pair[1]) * 
#         hierarchy_dist(synset_pair[0], synset_pair[1]))

# ######################### sentence similarity ##########################

# def most_similar_word(word, word_set):
#     """
#     Find the word in the joint word set that is most similar to the word
#     passed in. We use the algorithm above to compute word similarity between
#     the word and each word in the joint word set, and return the most similar
#     word and the actual similarity value.
#     """
#     max_sim = -1.0
#     sim_word = ""
#     for ref_word in word_set:
#       sim = word_similarity(word, ref_word)
#       if sim > max_sim:
#           max_sim = sim
#           sim_word = ref_word
#     return sim_word, max_sim
    
# def info_content(lookup_word):
#     """
#     Uses the Brown corpus available in NLTK to calculate a Laplace
#     smoothed frequency distribution of words, then uses this information
#     to compute the information content of the lookup_word.
#     """
#     global N
#     if N == 0:
#         # poor man's lazy evaluation
#         for sent in brown.sents():
#             for word in sent:
#                 word = word.lower()
#                 if not brown_freqs.has_key(word):
#                     brown_freqs[word] = 0
#                 brown_freqs[word] = brown_freqs[word] + 1
#                 N = N + 1
#     lookup_word = lookup_word.lower()
#     n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
#     return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
# def semantic_vector(words, joint_words, info_content_norm):
#     """
#     Computes the semantic vector of a sentence. The sentence is passed in as
#     a collection of words. The size of the semantic vector is the same as the
#     size of the joint word set. The elements are 1 if a word in the sentence
#     already exists in the joint word set, or the similarity of the word to the
#     most similar word in the joint word set if it doesn't. Both values are 
#     further normalized by the word's (and similar word's) information content
#     if info_content_norm is True.
#     """
#     sent_set = set(words)
#     semvec = np.zeros(len(joint_words))
#     i = 0
#     for joint_word in joint_words:
#         if joint_word in sent_set:
#             # if word in union exists in the sentence, s(i) = 1 (unnormalized)
#             semvec[i] = 1.0
#             if info_content_norm:
#                 semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
#         else:
#             # find the most similar word in the joint set and set the sim value
#             sim_word, max_sim = most_similar_word(joint_word, sent_set)
#             semvec[i] = PHI if max_sim > PHI else 0.0
#             if info_content_norm:
#                 semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
#         i = i + 1
#     return semvec                
            
# def semantic_similarity(sentence_1, sentence_2, info_content_norm):
#     """
#     Computes the semantic similarity between two sentences as the cosine
#     similarity between the semantic vectors computed for each sentence.
#     """
#     words_1 = nltk.word_tokenize(sentence_1)
#     words_2 = nltk.word_tokenize(sentence_2)
#     joint_words = set(words_1).union(set(words_2))
#     vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
#     vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
#     return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

# def word_order_vector(words, joint_words, windex):
#     """
#     Computes the word order vector for a sentence. The sentence is passed
#     in as a collection of words. The size of the word order vector is the
#     same as the size of the joint word set. The elements of the word order
#     vector are the position mapping (from the windex dictionary) of the 
#     word in the joint set if the word exists in the sentence. If the word
#     does not exist in the sentence, then the value of the element is the 
#     position of the most similar word in the sentence as long as the similarity
#     is above the threshold ETA.
#     """
#     wovec = np.zeros(len(joint_words))
#     i = 0
#     wordset = set(words)
#     for joint_word in joint_words:
#         if joint_word in wordset:
#             # word in joint_words found in sentence, just populate the index
#             wovec[i] = windex[joint_word]
#         else:
#             # word not in joint_words, find most similar word and populate
#             # word_vector with the thresholded similarity
#             sim_word, max_sim = most_similar_word(joint_word, wordset)
#             if max_sim > ETA:
#                 wovec[i] = windex[sim_word]
#             else:
#                 wovec[i] = 0
#         i = i + 1
#     return wovec

# def word_order_similarity(sentence_1, sentence_2):
#     """
#     Computes the word-order similarity between two sentences as the normalized
#     difference of word order between the two sentences.
#     """
#     words_1 = nltk.word_tokenize(sentence_1)
#     words_2 = nltk.word_tokenize(sentence_2)
#     joint_words = list(set(words_1).union(set(words_2)))
#     windex = {x[1]: x[0] for x in enumerate(joint_words)}
#     r1 = word_order_vector(words_1, joint_words, windex)
#     r2 = word_order_vector(words_2, joint_words, windex)
#     return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

# ######################### overall similarity ##########################

# def similarity(sentence_1, sentence_2, info_content_norm):
#     """
#     Calculate the semantic similarity between two sentences. The last 
#     parameter is True or False depending on whether information content
#     normalization is desired or not.
#     """
#     return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
#         (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)


if __name__ == '__main__':
    recreate_books()
    get_first_20_pages()

    # books = get_cleaned_books()
    # print(similarity(books[0], books[1], False))
