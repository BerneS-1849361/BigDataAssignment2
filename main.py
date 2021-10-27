import random
import string
import xml.sax
import numpy
# Author: Jasper Volders
# Author: Berne Sannen
from leven import levenshtein
from collections import Counter

class ClusterKmeans:
    def __init__(self, dimension):
        self.dimension = dimension
        self.centroid = [0] * dimension
        self.points = []
        self.titles = []
        self.words = {}

        self.skipWords = ["of", "for", "in", "and", "a", "by", "the", "on", "using", "from", "to", "with", "an"]



    def addPoint(self, point, title):
        self.points.append(point)
        self.titles.append(title)

    def removePoint(self, point):
        index = self.points.index(point)
        similarWord = self.titles[index]
        del self.titles[index]
        del self.points[index]

        return similarWord

    def getDistance(self, point):
        return numpy.linalg.norm(numpy.subtract(self.centroid, point))

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


class ClusterBFR:
    def __init__(self, dimension, maxDistance):
        self.dimension = dimension
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

    def addSimilarWord(self, word):
        self.similarWords.append(word)


    def isClose(self, point):
        return self.getDistance(point) <= self.maxDistance

    def getDistance(self, point):
        return numpy.linalg.norm(numpy.subtract(self.centroid, point))

    def merge(self, cluster):
        self.n += cluster.n
        for i in range(self.dimension):
            self.sum[i] += cluster.sum[i]
            self.sumq[i] += cluster.sumq[i]
            self.centroid[i] = self.sum[i] / self.n


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

    # check if point is close to a cluster or the retained points
    # if so add it to said compression or cluster
    # if not add to retained set
    def add(self, point, title):
        # check if point should be in cluster
        minCluster = self.getMinCluster(point)
        # add point to cluster
        minCluster.addPoint(point, title)

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

    def getMinCluster(self, point):
        minCluster = (None, numpy.inf)
        for cluster in self.clusters:
            distance = cluster.getDistance(point)
            if distance < minCluster[1]:
                minCluster = (cluster, distance)
        return minCluster[0]

    def __str__(self):
        #string = "dimension: " + str(self.dimension) + "\n"

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


# todo
class BFR:
    def __init__(self, numberOfClusters, dimension, maxDistance, startPoints):
        self.numberOfClusters = numberOfClusters

        self.dimension = dimension
        self.maxDistance = maxDistance
        self.clusters = []
        for index in range(numberOfClusters):
            cluster = ClusterBFR(dimension, maxDistance)
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
        # add point to cluster
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
                closePoints.append(retainer)

        for point in closePoints:
            self.retainSet.remove(point)

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
        compressionSet = ClusterBFR(self.dimension, self.maxDistance)
        for point in points:
            compressionSet.addPoint(point)
        self.compressionSets.append(compressionSet)

    def cleanUp(self):
        for compressionSet in self.compressionSets:
            minCluster = (None, numpy.infty)
            for cluster in self.clusters:
                distance = cluster.getDistance(compressionSet.centroid)
                if distance < minCluster[1]:
                    minCluster = (cluster, distance)
            if minCluster[0]:
                minCluster[0].merge(compressionSet)
        self.compressionSets.clear()

        for point in self.retainSet:
            minCluster = (None, numpy.infty)
            for cluster in self.clusters:
                distance = cluster.getDistance(point)
                if distance < minCluster[1]:
                    minCluster = (cluster, distance)
            if minCluster[0]:
                minCluster[0].addPoint(point)
        self.retainSet.clear()

    def __str__(self):
        string = "dimension: " + str(self.dimension) + "\t max distance: " + str(self.maxDistance) + "\n"

        string += "clusters: " + str(len(self.clusters)) + "\n"
        for cluster in self.clusters:
            string += "\tsize: " + str(cluster.n) + " centroid:" + str(cluster.centroid) + "\n"
        string += "compression: " + str(len(self.compressionSets)) + "\n"
        for cluster in self.compressionSets:
            string += "\tsize: " + str(cluster.n) + str(cluster.centroid) + "\n"
        string += "retain: " + str(len(self.retainSet)) + "\n"
        for point in self.retainSet:
            string += "\t" + str(point) + "\n"

        return string
# todo

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


def getDistanceMatrix(interval):
    matrix = [[]] * len(interval)
    for row in range(len(interval)):
        matrix[row] = [0] * len(interval)
        for col in range(len(interval)):
            lcs = LCSubStr(interval[row], interval[col])
            matrix[row][col] = calculateDistance(interval[row], interval[col])

    return matrix


def calculateDistance(str1, str2):
    return levenshtein(str1, str2)

# def calculateDistance(str1, str2):
#     count = 0
#     words = str1.split(' ')
#     for word in str2.split(' '):
#         words.append(word)
#     for word in words:
#         if word in str1 and word in str2:
#             count += 1
#     return len(words) - count

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
    return maxpoint[0],maxpoint[2]

if __name__ == '__main__':
    source = "dblp50000.xml"

    # create an XMLReader
    parser = xml.sax.make_parser()
    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # override the default ContextHandler

    fields = ["kdd", "pkdd", "icdm", "sdm"]
    handler = Handler(fields, 5, 2, 1992)
    parser.setContentHandler(handler)


    #parse file with the first pass
    parser.parse(source)
    # print(handler.data)

    for year, interval in handler.data:
        print(year)
        if len(interval) <= 2:
            print("no articles found")
            continue
        distanceMatrix = getDistanceMatrix(interval)


        # todo: make points as far away as possible
        randomPoints = []
        randomTitles = []

        randomIndex = random.randint(0, len(interval) - 1)
        randomPoints.append(distanceMatrix[randomIndex])
        randomTitles.append(interval[randomIndex])

        while len(randomPoints) < 4:
            furthestPointTitle = getFurthestPoint(randomPoints, distanceMatrix)
            randomPoints.append(furthestPointTitle[0])
            randomTitles.append(interval[furthestPointTitle[1]])

        # for i in range(4):
        #     randomIndex = random.randint(0, len(interval) - 1)
        #     randomPoints.append(distanceMatrix[randomIndex])
        #     randomTitles.append(interval[randomIndex])
        kmeans = Kmeans(4, len(interval), randomPoints, randomTitles)

        for index, row in enumerate(distanceMatrix):
            if row not in randomPoints:
                kmeans.add(row, interval[index])

        while kmeans.reassignPoints():
            pass
        print(kmeans)
        # bfr = BFR(4, len(interval), 15, randomPoints)
        # for row in distanceMatrix:
        #     if row not in randomPoints:
        #         bfr.add(row)
        #
        # bfr.cleanUp()
        #
        # print(bfr)


    # bfr = BFR(2, 4, 3, [[5, 6, 5, 5], [10, 11, 10, 10]])
    # bfr.add([5, 5, 6, 5])
    # bfr.add([5, 5, 5, 5])
    # bfr.add([10, 10, 11, 10])
    # bfr.add([10, 10, 10, 10])
    # bfr.add([0,0,0,0])
    # bfr.add([0,1,0,2])



