import sys

class NaiveBayesEvaluate:
    def __init__(self):
        rootPath = sys.argv[1]
        self.results = {"spam": {"spam": 0, "ham": 0}, "ham": {"spam": 0, "ham": 0}}
        self.calculateScore(rootPath)
        self.printResults()

    def calculateScore(self, rootPath):
        with open(rootPath, "r") as file:
            for result in file.readlines():
                predictedClass, filePath = result.split()
                filePath = filePath.rstrip(".txt")
                realClass = "spam" if filePath.endswith("spam") else "ham"
                self.results[predictedClass][realClass] += 1

    def printResults(self):
        totalDoc = self.results["spam"]["spam"] + self.results["spam"]["ham"] + self.results["ham"]["spam"] + self.results["ham"]["ham"]
        accuracy = (self.results["spam"]["spam"] + self.results["ham"]["ham"]) / totalDoc if totalDoc > 0 else 0

        tpfp = (self.results["spam"]["ham"] + self.results["spam"]["spam"])
        precision_spam = self.results["spam"]["spam"] / tpfp if tpfp > 0 else 0

        tpfn = (self.results["ham"]["spam"] + self.results["spam"]["spam"])
        recall_spam = self.results["spam"]["spam"] / tpfn if tpfn > 0 else 0

        den = (precision_spam + recall_spam)
        f1_score_spam = (2 * precision_spam * recall_spam) / den if den > 0 else 0

        tnfn = self.results["ham"]["spam"] + self.results["ham"]["ham"]
        precision_ham = self.results["ham"]["ham"] / tnfn if tnfn > 0 else 0

        tnfp = (self.results["spam"]["ham"] + self.results["ham"]["ham"])
        recall_ham = self.results["ham"]["ham"] / tnfp if tnfp > 0 else 0

        den = (precision_ham + recall_ham)
        f1_score_ham = (2 * precision_ham * recall_ham) / den if den > 0 else 0

        print("Accuracy = " + str(accuracy))
        print("Spam Precision = " + str(precision_spam))
        print("Spam Recall = " + str(recall_spam))
        print("Spam F-1 Score = " + str(f1_score_spam))
        print("Ham Precision = " + str(precision_ham))
        print("Ham Recall = " + str(recall_ham))
        print("Ham F-1 Score = " + str(f1_score_ham))

evaluate = NaiveBayesEvaluate()