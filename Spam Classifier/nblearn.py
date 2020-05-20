import sys
import os
import json
from collections import defaultdict


class TokenData:
    def __init__(self):
        self.hamTokens, self.spamTokens = defaultdict(int), defaultdict(int)
        self.hamCount, self.spamCount = 0, 0

    # def calculateProbabilitiesWithLaplaceSmoothing(self):
    #     vocabularySize = len(set(list(self.hamTokens.keys()) + list(self.spamTokens.keys())))
    #     totalSpamWords, totalHamWords = sum(self.spamTokens.values()), sum(self.hamTokens.values())
    #
    #     for key, val in self.hamTokens.items():
    #         self.hamTokens[key] = (val + 1) / (totalHamWords + vocabularySize)
    #
    #     for key, val in self.spamTokens.items():
    #         self.spamTokens[key] = (val + 1) / (totalSpamWords + vocabularySize)
    #
    #     totalDocCount = self.hamCount + self.spamCount
    #     self.hamCount /= totalDocCount
    #     self.spamCount /= totalDocCount

    def getAsJson(self):
        jsonData = defaultdict(dict)
        jsonData["spam"]["prob"] = self.spamCount
        jsonData["spam"]["tokensProb"] = self.spamTokens
        jsonData["ham"]["prob"] = self.hamCount
        jsonData["ham"]["tokensProb"] = self.hamTokens
        return jsonData


class NaiveBayesLearn:
    def __init__(self):
        rootPath = sys.argv[1]
        self.trainData = TokenData()
        self.loadData(rootPath)
        self.storeData()

    def loadData(self, rootPath):
        for dirPath, dirNames, files in os.walk(rootPath):
            for file in files:
                if file.endswith(".txt"):
                    if dirPath.endswith("ham"):
                        self.trainModel(os.path.join(dirPath, file), False)
                    elif dirPath.endswith("spam"):
                        self.trainModel(os.path.join(dirPath, file), True)

    def trainModel(self, filePath, isSpam):
        with open(filePath, "r", encoding="latin1") as file:
            if isSpam:
                self.trainData.spamCount += 1

                for token in file.read().split():
                    token = token.lower()
                    self.trainData.spamTokens[token] += 1

            else:
                self.trainData.hamCount += 1
                for token in file.read().split():
                    token = token.lower()
                    self.trainData.hamTokens[token] += 1

    def storeData(self):
        with open("nbmodel.txt", "w") as file:
            json.dump(self.trainData.getAsJson(), file)


learn = NaiveBayesLearn()
