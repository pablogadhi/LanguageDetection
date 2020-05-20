import numpy as np
from translator import Translator
from nltk.translate.bleu_score import sentence_bleu


def remove_dim(translation):
    return list(map(lambda x: x[0], translation))


class Detector:
    def __init__(self, load_from=None):
        pass

    def backtranslations(self, train_data, iter_num=3):
        es_translations = []
        de_translations = []
        en_translations = [[train_data, train_data]]
        translator = {"en-es": Translator("en-es"), "es-en": Translator(
            "es-en"), "en-de": Translator("en-de"), "de-en": Translator("de-en")}
        for i in range(0, iter_num):
            es_sentences = remove_dim(
                translator["en-es"].translate(en_translations[i][0])[1])
            es_translations.append(es_sentences)
            de_sentences = remove_dim(
                translator["en-de"].translate(en_translations[i][1])[1])
            de_translations.append(de_sentences)

            es_en_back = remove_dim(
                translator["es-en"].translate(es_sentences)[1])
            de_en_back = remove_dim(
                translator["de-en"].translate(de_sentences)[1])
            en_translations.append([es_en_back, de_en_back])

        return np.array(es_translations, dtype=object).T, np.array(de_translations, dtype=object).T

    def train(self, train_file, validation_data):
        train = open(train_file, "r").read().split("\n")
        train = list(map(lambda x: x.encode(), train))

        es_back, de_back = self.backtranslations(train)
        print(es_back.shape)

    def predict(self, data):
        pass


if __name__ == "__main__":
    detector = Detector()
    detector.train("data/svm_train.txt", None)
