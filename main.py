import xml.sax
from leven import levenshtein
import numpy as np
from sklearn.cluster import dbscan
import matplotlib.pyplot as plt
# Author: Jasper Volders
# Author: Berne Sannen


class Handler(xml.sax.ContentHandler):
    def __init__(self, field):
        self.title = ""
        self.year = 0
        self.key = ""
        self.field = field
        self.skip = False

        self.data = []

    # Call when an element starts
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        # get the scientific field of the article and set it to skip if it is an other field
        if tag == "article" or tag == "inproceedings" or tag == "proceedings" or tag == "book" or \
                tag == "incollection" or tag == "phdthesis" or tag == "mastersthesis" or tag == "www":
            self.skip = False
            self.key = attributes["key"]
            self.key = self.key.split("/")
            if self.key[1] != self.field:
                self.skip = True

    # Call when an elements ends
    def endElement(self, tag):
        if self.skip:
            self.CurrentData = ""
            return
        # if the element is an article add the articles combinations to the bucket
        if tag == "article" or tag == "inproceedings" or tag == "proceedings" or tag == "book" or \
                tag == "incollection" or tag == "phdthesis" or tag == "mastersthesis" or tag == "www":
            if self.year > 1990 and self.year <= 2000:
                self.data.append(self.title)
        self.CurrentData = ""

    # Call when a character is read
    def characters(self, content):
        if self.skip:
            return
        if self.CurrentData == "year":
            self.year = int(content)
        if self.CurrentData == "title":
            self.title = content

def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(handler.data[i], handler.data[j])

if __name__ == '__main__':
    source = "dblp50000.xml"

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler

    handler = Handler("pkdd")
    parser.setContentHandler(handler)


    #parse file with the first pass
    parser.parse(source)


    print(handler.data)
    print(len(handler.data))

    X = np.arange(len(handler.data)).reshape(-1, 1)
    dbscan(X, metric=lev_metric, eps=5, min_samples=2)









