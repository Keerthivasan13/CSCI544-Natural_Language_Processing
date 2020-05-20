import math
import sys
import os
import json
from collections import defaultdict


class TokenData:
    def __init__(self):
        self.hamTokensProb, self.spamTokensProb = defaultdict(int), defaultdict(int)
        self.hamProb, self.spamProb = 0, 0

    def calculateProbabilitiesWithLaplaceSmoothing(self):
        self.vocabularySize = len(set(list(self.hamTokensProb.keys()) + list(self.spamTokensProb.keys())))
        self.totalSpamWords, self.totalHamWords = sum(self.spamTokensProb.values()), sum(self.hamTokensProb.values())

        totalDocCount = self.hamProb + self.spamProb
        self.hamProb /= totalDocCount if totalDocCount else 1
        self.spamProb /= totalDocCount if totalDocCount else 1

class NaiveBayesClassify:
    def __init__(self):
        rootPath = sys.argv[1]
        self.trainData = TokenData()
        self.readTrainData()

        self.classifiedTestData = []
        testDataPaths = self.getTestDataFiles(rootPath)
        self.classifyTestData(testDataPaths)
        self.writeTestData()

    def readTrainData(self):
        with open("nbmodel.txt", "r", encoding="latin1") as file:
            data = json.load(file)
            self.trainData.hamProb = data["ham"]["prob"]
            self.trainData.hamTokensProb = data["ham"]["tokensProb"]
            self.trainData.spamProb = data["spam"]["prob"]
            self.trainData.spamTokensProb = data["spam"]["tokensProb"]

            self.trainData.calculateProbabilitiesWithLaplaceSmoothing()

    def getTestDataFiles(self, rootPath):
        paths = []
        for dirPath, dirNames, files in os.walk(rootPath):
            for file in files:
                if file.endswith(".txt"):
                    paths.append(os.path.join(dirPath, file))
        return paths

    def classifyTestData(self, testDataPaths):
        for path in testDataPaths:
            with open(path, "r", encoding="latin1") as file:
                class_ = self.classifyFile(file.read().split())
                self.classifiedTestData.append(class_ + "\t" + path)

    def classifyFile(self, tokens):
        probFileIsHam = math.log(self.trainData.hamProb)
        probFileIsSpam = math.log(self.trainData.spamProb)

        for token in tokens:
            token = token.lower()
            if token in self.trainData.hamTokensProb:
                probFileIsHam += math.log((self.trainData.hamTokensProb[token] + 1) / (self.trainData.vocabularySize + self.trainData.totalHamWords))
            else:
                probFileIsHam -= math.log((self.trainData.vocabularySize + self.trainData.totalHamWords))
            if token in self.trainData.spamTokensProb:
                probFileIsSpam += math.log((self.trainData.spamTokensProb[token] + 1) / (self.trainData.vocabularySize + self.trainData.totalSpamWords))
            else:
                probFileIsSpam -= math.log(self.trainData.vocabularySize + self.trainData.totalSpamWords)
        return "spam" if probFileIsSpam > probFileIsHam else "ham"

    def writeTestData(self):
        with open("nboutput.txt", "w") as file:
            for data in self.classifiedTestData:
                file.write(data + "\n")


classify = NaiveBayesClassify()
