from HTMLParser import HTMLParser
import os

FILE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'gap_html'))



class MyHTMLParser(HTMLParser):
    # def handle_starttag(self, tag, attrs):
    #     print "Encountered a start tag:", tag
    #
    # def handle_endtag(self, tag):
    #     print "Encountered an end tag :", tag

    def handle_data(self, data):
        data.strip()
        print "Encountered some data  :", data


# Read the file

with open(os.path.join(FILE_FOLDER, 'gap_-C0BAAAAQAAJ', '00000023.html'), 'r') as f:
    data = f.read()


# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed(data)
