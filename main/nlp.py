import os
import logging
import json

from datetime import datetime
from api import AlchemyAPI
from nlp_exc.exception import NLPValueError
from bag_of_words import get_cleaned_books, get_book_names, FILE_FOLDER
from preprocessing import tokenize_and_stem

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config'))
LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))


class NLPHandler(object):
    def __init__(self):
        self.alchemy_api = AlchemyAPI()
        self.logger = logging.Logger('NLPLogger', level=logging.INFO)
        self.logger.addHandler(logging.FileHandler(filename=os.path.join(LOG_PATH, 'nlp.log'), mode='a+'))

    def get_combined_result(self, text=''):
        opts = {
            'sentiment': 1
        }
        response = self.alchemy_api.combined('text', text, options=opts)

        if response['status'] == 'OK':
            return response
        else:
            raise NLPValueError('Error in combined call: ' + response['statusInfo'])

    def run_handler(self):
        run_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        file_path = os.path.join(FILE_FOLDER, 'nlp_files')

        print("")
        print("Starting AlchemyAPI calls.")
        print("")

        book_list = get_cleaned_books('first_pages')
        book_names = get_book_names()
        counter = 0

        parsed_books = []

        for root, dirs, json_results in sorted(os.walk(file_path)):
            for item in sorted(json_results):
                parsed_books.append(item.replace("_nlp_result.json", ""))

        try:
            for book in book_list:
                book_name = book_names[counter]

                if book_name not in parsed_books:
                    print("Running NLP for " + book_name)

                    result = self.get_combined_result(book)
                    with open(os.path.join(FILE_FOLDER, 'nlp_files', book_name + '_nlp_result.json'),
                              'w+') as json_outfile:
                        json.dump(result, json_outfile)
                else:
                    print("NLP Result parsing already done for " + book_name)

                counter += 1

        except NLPValueError as err:
            self.logger.exception(run_time + " - " + str(err.message))

    @staticmethod
    def print_nlp_results():
        file_path = os.path.join(FILE_FOLDER, 'nlp_files')

        for root, dirs, json_results in sorted(os.walk(file_path)):
            for book_res in sorted(json_results):

                with open(os.path.join(FILE_FOLDER, 'nlp_files', book_res)) as book_nlp_result:
                    result = json.load(book_nlp_result)

                print("# ######################### BOOK NAME ##########################")
                book_name = book_res.replace("_nlp_result.json", "")
                print(book_name)

                print("")
                print("# ######################### KEYWORDS ##########################")
                keywords = result['keywords']
                for keyword in keywords:
                    print(keyword['text'] + ' - ' + keyword['relevance'])

                print("")
                print("# ######################### CONCEPTS ##########################")
                concepts = result['concepts']
                for concept in concepts:
                    print(concept['text'] + ' - ' + concept['relevance'])

                print("")
                print("")
                print("")

    @staticmethod
    def parse_nlp_results():
        file_path = os.path.join(FILE_FOLDER, 'nlp_files')
        nlp_stemmed_texts = []

        for root, dirs, json_results in sorted(os.walk(file_path)):
            for book_res in sorted(json_results):
                book_words = []

                with open(os.path.join(FILE_FOLDER, 'nlp_files', book_res)) as book_nlp_result:
                    result = json.load(book_nlp_result)
                    for concept in result['concepts']:
                        concepts = concept['text'].split()
                        for conc in concepts:
                            book_words.append(conc.lower())
                    for keyword in result['keywords']:
                        keywords = keyword['text'].split()
                        for keyw in keywords:
                            book_words.append(keyw.lower())

                    meaningful_words = [w for w in book_words if len(w) > 2]

                    book_text = " ".join(tokenize_and_stem(" ".join(meaningful_words)))
                    nlp_stemmed_texts.append(book_text)

        return nlp_stemmed_texts

    @staticmethod
    def create_sparse_matrix():
        file_path = os.path.join(FILE_FOLDER, 'nlp_files')
        nlp_matrix = []
        nlp_big_matr = []

        for root, dirs, json_results in sorted(os.walk(file_path)):
            for book_res in sorted(json_results):
                with open(os.path.join(FILE_FOLDER, 'nlp_files', book_res)) as book_nlp_result:
                    result = json.load(book_nlp_result)
                    for concept in result['concepts']:
                        conc = concept['text'].lower()
                        if conc not in nlp_matrix:
                            nlp_matrix.append(conc)
                    for keyword in result['keywords']:
                        keyw = keyword['text'].lower()
                        if keyw not in nlp_matrix:
                            nlp_matrix.append(keyw)

            for book_result in sorted(json_results):
                with open(os.path.join(FILE_FOLDER, 'nlp_files', book_result)) as book_nlp_result:
                    new_matr = nlp_matrix[:]
                    book_words = {}
                    nlp_result = json.load(book_nlp_result)

                    for concept in nlp_result['concepts']:
                        conc = concept['text'].lower()
                        if conc not in book_words:
                            book_words[conc] = concept['relevance']
                    for keyword in nlp_result['keywords']:
                        keyw = keyword['text'].lower()
                        if keyw not in book_words:
                            book_words[keyw] = keyword['relevance']

                    for wrd_key, wrd in enumerate(new_matr):
                        if wrd in book_words:
                            new_matr[wrd_key] = book_words[wrd]
                        else:
                            new_matr[wrd_key] = 0
                    nlp_big_matr.append(new_matr)

        return nlp_big_matr


if __name__ == '__main__':
    nlp_handler = NLPHandler()
    # nlp_handler.run_handler()
    # nlp_handler.print_nlp_results()
    # nlp_stemmed_books = nlp_handler.parse_nlp_results()
    nlp_matr = nlp_handler.create_sparse_matrix()

    print(nlp_matr)
