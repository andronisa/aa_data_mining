import os
import logging
import json

from datetime import datetime
from api import AlchemyAPI
from nlp_exc.exception import NLPValueError
from bag_of_words import get_cleaned_books

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

    def get_sentiment_result(self, text=''):
        response = self.alchemy_api.sentiment('text', text)
        if response['status'] == 'OK':
            return response
        else:
            raise NLPValueError('Error in sentiment analysis call: ' + response['statusInfo'])

    def run_handler(self):
        run_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        print("Starting AlchemyAPI calls. Please check nlp.log inside 'logs' folder for business_id")

        book_list = get_cleaned_books('first_pages')

        try:
            for book in book_list:
                result = self.get_combined_result(book)
                print json.dumps(result, indent=4, sort_keys=True)
            pass
            # self.logger.info("Business " + str(business['_id']) + " finished.")
        except NLPValueError as err:
            self.logger.exception(run_time + " - " + str(err.message))


if __name__ == '__main__':
    nlp_handler = NLPHandler()
    nlp_handler.run_handler()
