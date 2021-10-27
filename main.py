import random
import xml.sax
from collections import Counter

import numpy
from leven import levenshtein


# Author: Jasper Volders
# Author: Berne Sannen


# cluster class for using with kmeans
class ClusterKmeans:
    def __init__(self, dimension):
        self.dimension = dimension
        self.centroid = [0] * dimension
        self.points = []
        self.titles = []
        self.words = {}

        self.skipWords = ["of", "for", "in", "and", "a", "by", "the", "on", "using", "from", "to", "with", "an"]

    # adds a point to the cluster
    def addPoint(self, point, title):
        self.points.append(point)
        self.titles.append(title)

    # removes a point from the cluster
    def removePoint(self, point):
        index = self.points.index(point)
        similarWord = self.titles[index]
        del self.titles[index]
        del self.points[index]

        return similarWord

    # returns the distance of the given point to the centroid of the cluster
    def getDistance(self, point):
        return numpy.linalg.norm(numpy.subtract(self.centroid, point))

    # updates the centroid of the cluster so it represents all the points in the cluster atm
    def updateCentroid(self):
        sumPoint = [0] * self.dimension
        for point in self.points:
            for index, number in enumerate(point):
                sumPoint[index] += number

        for index, number in enumerate(sumPoint):
            if len(self.points) > 0:
                self.centroid[index] = number / len(self.points)
            else:
                self.centroid[index] = 0

    # returns all the possible topics in a dict with their frequency
    def getTopics(self):
        for title in self.titles:
            title = title.lower()
            title = self.removePunctuation(title)
            words = title.split(" ")
            for word in words:
                if word in self.skipWords:
                    continue
                if word not in self.words:
                    self.words[word] = 1
                else:
                    self.words[word] += 1
        return self.words

    # source: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
    def removePunctuation(self, str):
        # initializing punctuations string
        punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        # Removing punctuations in string
        # Using loop + punctuation string
        for ele in str:
            if ele in punc:
                str = str.replace(ele, "")

        return str


# Kmeans clustering algorithm
class Kmeans:
    def __init__(self, numberOfClusters, dimension, startPoints, startTitles):
        self.numberOfClusters = numberOfClusters

        self.dimension = dimension
        self.clusters = []
        for index in range(numberOfClusters):
            cluster = ClusterKmeans(dimension)
            cluster.addPoint(startPoints[index], startTitles[index])
            cluster.updateCentroid()
            self.clusters.append(cluster)

    # adds the point to the cluster that is closed to the point
    def add(self, point, title):
        # check if point should be in cluster
        minCluster = self.getMinCluster(point)
        # add point to cluster
        minCluster.addPoint(point, title)

    # shuffles the points between clusters after updating their centroids so that they now are again in the closed cluster
    def reassignPoints(self):
        for cluster in self.clusters:
            cluster.updateCentroid()

        pointsToReassign = []
        for cluster in self.clusters:
            for point in cluster.points:
                minCluster = self.getMinCluster(point)
                if minCluster != cluster:
                    pointsToReassign.append((point, cluster, minCluster))
        for point, clusterFrom, clusterTo in pointsToReassign:
            similarWord = clusterFrom.removePoint(point)
            clusterTo.addPoint(point, similarWord)
        return len(pointsToReassign) > 0

    # returns the cluster that is closed to the given point
    def getMinCluster(self, point):
        minCluster = (None, numpy.inf)
        for cluster in self.clusters:
            distance = cluster.getDistance(point)
            if distance < minCluster[1]:
                minCluster = (cluster, distance)
        return minCluster[0]

    # to string function so the clusters can be printed
    def __str__(self):
        # string = "dimension: " + str(self.dimension) + "\n"

        string = "clusters: " + str(len(self.clusters)) + "\n"
        for cluster in self.clusters:
            string += "\ttotal number of articles: " + str(len(cluster.points)) + "\n"
            words = cluster.getTopics()
            k = Counter(words)
            string += "\tpossible topics: occurrences \n"
            for word in k.most_common(5):
                string += "\t\t- " + word[0] + ": " + str(word[1]) + "\n"
            string += "\n\n"
        return string


# class for handling the xml parsing
class Handler(xml.sax.ContentHandler):
    def __init__(self, fields, intervalSize, overlapSize, startYear):
        self.title = ""
        self.year = 0
        self.key = ""
        self.fields = fields
        self.skip = False

        self.intervalSize = intervalSize
        self.overlapSize = overlapSize
        self.startYear = startYear

        self.intervalArray = self.createIntervals()
        self.data = []
        for i in range(len(self.intervalArray)):
            self.data.append((self.intervalArray[i], []))

    # Call when an element starts
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        # get the scientific field of the article and set it to skip if it is an other field
        if tag == "article" or tag == "inproceedings" or tag == "proceedings" or tag == "book" or \
                tag == "incollection" or tag == "phdthesis" or tag == "mastersthesis" or tag == "www":
            self.skip = False
            self.key = attributes["key"]
            self.key = self.key.split("/")
            if self.key[1] not in self.fields:
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
                    self.data[index][1].append(self.title)
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


# get the distancematrix of all the titles in the interval
def getDistanceMatrix(interval):
    matrix = [[]] * len(interval)
    for row in range(len(interval)):
        matrix[row] = [0] * len(interval)
        for col in range(len(interval)):
            lcs = LCSubStr(interval[row], interval[col])
            matrix[row][col] = calculateDistance(interval[row], interval[col])

    return matrix


# distancefunction
def calculateDistance(str1, str2):
    return levenshtein(str1, str2)


# source: https://www.geeksforgeeks.org/longest-common-substring-dp-29/
def LCSubStr(X, Y):
    m = len(X)
    n = len(Y)
    # Create a table to store lengths of
    # longest common suffixes of substrings.
    # Note that LCSuff[i][j] contains the
    # length of longest common suffix of
    # X[0...i-1] and Y[0...j-1]. The first
    # row and first column entries have no
    # logical meaning, they are used only
    # for simplicity of the program.

    # LCSuff is the table with zero
    # value initially in each cell
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

    # To store the length of
    # longest common substring
    result = 0

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (X[i - 1] == Y[j - 1]):
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


# returns the point that is furthest away from the already chosen points
def getFurthestPoint(chosenPoints, distanceMatrix):
    maxpoint = (None, 0, 0)
    idx = -1
    for row in distanceMatrix:
        idx += 1
        if row in chosenPoints:
            continue
        distance = 0
        for point in chosenPoints:
            distance += numpy.linalg.norm(numpy.subtract(row, point))
        if distance > maxpoint[1]:
            maxpoint = (row, distance, idx)
    return maxpoint[0], maxpoint[2]


if __name__ == '__main__':
    source = "dblp50000.xml"
    fields = ["kdd", "pkdd", "icdm", "sdm"]
    intervalSize = 5
    overlapSize = 2
    startYear = 1992

    numberOfClusters = 4

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler

    handler = Handler(fields, intervalSize, overlapSize, startYear)
    parser.setContentHandler(handler)

    # parse file with the first pass
    parser.parse(source)

    # for each interval
    for year, interval in handler.data:
        print(year)
        if len(interval) <= numberOfClusters:
            print("not enough articles found")
            continue

        # get the distance matrix of all the titles in the given interval
        distanceMatrix = getDistanceMatrix(interval)

        # select 1 startpoint for every cluster
        startPoints = []
        startTitles = []

        randomIndex = random.randint(0, len(interval) - 1)
        startPoints.append(distanceMatrix[randomIndex])
        startTitles.append(interval[randomIndex])

        while len(startPoints) < numberOfClusters:
            furthestPointTitle = getFurthestPoint(startPoints, distanceMatrix)
            startPoints.append(furthestPointTitle[0])
            startTitles.append(interval[furthestPointTitle[1]])

        # initialize kmeans with the above created startpoints
        kmeans = Kmeans(numberOfClusters, len(interval), startPoints, startTitles)

        # add all other points to kmeans
        for index, row in enumerate(distanceMatrix):
            if row not in startPoints:
                kmeans.add(row, interval[index])

        # while there are points being reassigned keep reassigning the points to new clusters
        while kmeans.reassignPoints():
            pass

        print(kmeans)
