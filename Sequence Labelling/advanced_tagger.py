import pycrfsuite
import sys
import hw2_corpus_tool as tool
import random


class Advanced_Tagger:
    def __init__(self):
        self.input_dir = sys.argv[1]
        self.test_dir = sys.argv[2]
        self.output_file = sys.argv[3]
        self.train_model()
        self.evaluate_model()

    def generate_features_from_dialogue(self, dialogue):
        features = []
        for u_idx, utterance in enumerate(dialogue):
            sub_features = []
            if u_idx == 0:
                sub_features.append("FIRST_UTTERANCE")
            else:
                if u_idx == len(dialogue) - 1:
                    sub_features.append("LAST_UTTERANCE")
                if dialogue[u_idx - 1].speaker != utterance.speaker:
                    sub_features.append("SPEAKER_CHANGE")
                else:
                    sub_features.append("SPEAKER_NO_CHANGE")
                    if dialogue[u_idx - 1].pos and  utterance.pos:
                        sub_features.append("PREV_" + dialogue[u_idx - 1].pos[-1].token +"_NOW_" + utterance.pos[0].token)

            sub_features.extend(self.generate_features_from_utterance(utterance))
            features.append(sub_features)

        return features

    def generate_features_from_utterance(self, utterance):
        features = []
        text = utterance.text
        for idx, word in enumerate(text):
            features.append("TEXT_" + word)
            if idx != 0:
                features.append("TEXT_" + text[idx - 1] + "_" + word)

        if not utterance.pos:
            features.append("EMPTY_WORDS")
            return features

        for idx, word in enumerate(utterance.pos):
            if idx != 0:
                features.append("TOKEN_" + utterance.pos[idx - 1].token + "_" + word.token)
                features.append("POS_" + utterance.pos[idx - 1].pos + "_" + word.pos)
            else:
                features.append("FIRST_TOKEN_" + word.token)
                features.append("FIRST_POS_" + word.pos)
            if idx == len(utterance.pos) - 1:
                features.append("LAST_TOKEN_" + word.token)
                features.append("LAST_POS_" + word.pos)
            features.append("TOKEN_" + word.token)
            features.append("POS_" + word.pos)
        return features

    def extract_labels_from_dialogue(self, dialogue):
        return [utterance.act_tag for utterance in dialogue]

    def train_model(self):
        trainer = pycrfsuite.Trainer(verbose=False)
        X_train = [dialogue for dialogue in list(tool.get_data(self.input_dir))]
        y_train = [self.extract_labels_from_dialogue(dialogue) for dialogue in X_train]

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(self.generate_features_from_dialogue(xseq), yseq)

        trainer.set_params({
            'c1': 1.0,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train("dialogue_tagger.crtsuite")

    def evaluate_model(self):
        tagger = pycrfsuite.Tagger()
        tagger.open("dialogue_tagger.crtsuite")

        X_test = [dialogue for dialogue in list(tool.get_data(self.test_dir))]
        y_test = [self.extract_labels_from_dialogue(dialogue) for dialogue in X_test]

        no_of_correct_predictions = 0
        total = 0
        predictions = []

        for xseq, yseq in zip(X_test, y_test):
            predicted = tagger.tag(self.generate_features_from_dialogue(xseq))
            predictions.append(predicted)

            for pred, act in zip(predicted, yseq):
                if act:
                    no_of_correct_predictions += (1 if pred == act else 0)
                    total += 1

        accuracy = no_of_correct_predictions / total
        print("Accuracy = " + str(accuracy))
        self.write_labels(predictions)

    def write_labels(self, predictions):
        with open(self.output_file, "w+") as file:
            for dialogue_pred in predictions:
                for label in dialogue_pred:
                    file.write(label + "\n")
                file.write("\n")

obj = Advanced_Tagger()