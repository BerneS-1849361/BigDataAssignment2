import xml.sax

import numpy
from leven import levenshtein
import numpy as np
from sklearn.cluster import dbscan, DBSCAN
import matplotlib.pyplot as plt
# Author: Jasper Volders
# Author: Berne Sannen

class Cluster():
    def __init__(self, dimension, maxDistance):
        self.centroid = [0] * dimension
        self.maxDistance = maxDistance
        self.n = 0
        self.sum = [0] * dimension
        self.sumq = [0] * dimension

    def addPoint(self, point):
        self.n += 1
        for i, number in enumerate(point):
            self.sum[i] += number
            self.sumq[i] += (number ** 2)
            self.centroid[i] = self.sum[i] / self.n


    def isClose(self, point):
        return self.getDistance(point) <= self.maxDistance

    def getDistance(self, point):
        return numpy.linalg.norm(numpy.subtract(self.centroid, point))


class BFR:
    def __init__(self, numberOfClusters, dimension, maxDistance, startPoints):
        self.numberOfClusters = numberOfClusters

        self.dimension = dimension
        self.maxDistance = maxDistance
        self.clusters = []
        for index in range(numberOfClusters):
            cluster = Cluster(dimension, maxDistance)
            cluster.addPoint(startPoints[index])
            self.clusters.append(cluster)
        self.compressionSets = []
        self.retainSet = []

    # check if point is close to a cluster or the retained points
    # if so add it to said compression or cluster
    # if not add to retained set
    def add(self, point):
        # check if point should be in cluster
        minCluster = (None, numpy.infty)
        for cluster in self.clusters:
            distance = cluster.getDistance(point)
            if distance < minCluster[1]:
                minCluster = (cluster, distance)
        #add point to cluster
        if minCluster[0].isClose(point):
            minCluster[0].addPoint(point)
            return


        # check if point should be in compression set
        minCompressionSet = (None, numpy.infty)
        for compressionSet in self.compressionSets:
            distance = compressionSet.getDistance(point)
            if distance < minCompressionSet[1]:
                minCompressionSet = (compressionSet, distance)
        # add point to compression set
        if minCompressionSet[0] and minCompressionSet[0].isClose(point):
            minCompressionSet[0].addPoint(point)
            return

        closePoints = []
        for retainer in self.retainSet:
            if numpy.linalg.norm(numpy.subtract(retainer, point)) <= self.maxDistance:
                self.retainSet.remove(retainer)
                closePoints.append(retainer)

        if len(closePoints) != 0:
            closePoints.append(point)
            self.compress(closePoints)
            return


        # add point to retained sets
        self.retain(point)


    # add point to retain set
    def retain(self, point):
        self.retainSet.append(point)

    # add point to a compression set and forget point
    def compress(self, points):
        compressionSet = Cluster(self.dimension, self.maxDistance)
        for point in points:
            compressionSet.addPoint(point)
        self.compressionSets.append(compressionSet)


class Handler(xml.sax.ContentHandler):
    def __init__(self, field, intervalSize, overlapSize, startYear):
        self.title = ""
        self.year = 0
        self.key = ""
        self.field = field
        self.skip = False

        self.intervalSize = intervalSize
        self.overlapSize = overlapSize
        self.startYear = startYear

        self.intervalArray = self.createIntervals()
        self.data = [[]] * len(self.intervalArray)
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
            for index in range(len(self.intervalArray)):
                if self.intervalArray[index][0] <= self.year <= self.intervalArray[index][1]:
                    self.data[index].append(self.title)
        self.CurrentData = ""

    # Call when a character is read
    def characters(self, content):
        if self.skip:
            return
        if self.CurrentData == "year":
            self.year = int(content)
        if self.CurrentData == "title":
            self.title = content


    def createIntervals(self):
        intervalArray = []
        x = self.startYear
        while x <= 2021:
            intervalArray.append((x, x + self.intervalSize))
            x += self.intervalSize - self.overlapSize
        return intervalArray

def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(handler.data[i], handler.data[j])

if __name__ == '__main__':
    # source = "dblp50000.xml"
    #
    # # create an XMLReader
    # parser = xml.sax.make_parser()
    # # turn off namepsaces
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    #
    # # override the default ContextHandler
    #
    # handler = Handler("pkdd", 10, 2, 1936)
    # parser.setContentHandler(handler)
    #
    #
    # #parse file with the first pass
    # parser.parse(source)
    #
    #
    # print(handler.data)

    bfr = BFR(2, 4, 3, [[5, 6, 5, 5], [10, 11, 10, 10]])
    bfr.add([5, 5, 6, 5])
    bfr.add([5, 5, 5, 5])
    bfr.add([10, 10, 11, 10])
    bfr.add([10, 10, 10, 10])

    print(bfr.clusters[0].centroid, bfr.clusters[1].centroid)










