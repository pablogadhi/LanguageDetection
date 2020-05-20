import numpy as np
from translator import Translator
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

LANG_PREFIXES = ['en', 'es', 'de']


def remove_dim(translation):
    return list(map(lambda x: x[0], translation))


def bleu_dist(translations):
    t_translations = list(map(lambda x: word_tokenize(x), translations))

    bleu = []
    for i in range(0, len(translations) - 1):
        bleu.append(sentence_bleu([t_translations[i]], t_translations[i+1]))

    return bleu


class Detector:
    def __init__(self, load_from="classifier_model.joblib"):

        try:
            self.classifier = load("classifier_model.joblib")
        except:
            print("Model not found! Train a new model!")
            self.classifier = None

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

    def bleu_from_backtrans(self, back_trans_list):
        # TODO make single numpy operation
        res = np.apply_along_axis(bleu_dist, 1, back_trans_list[0])
        for i in range(1, len(back_trans_list)):
            lang_bleu = np.apply_along_axis(bleu_dist, 1, back_trans_list[i])
            res = np.hstack((res, lang_bleu))
        return res

    def train(self, train_file, validation_data):
        train = open(train_file, "r").read().split("\n")
        train = list(map(lambda x: x.encode(), train))

        es_back, de_back = self.backtranslations(train)
        x_data = self.bleu_from_backtrans([es_back, de_back])

        y_data = open(validation_data, "r").read().split("\n")
        l_encoder = LabelEncoder()
        l_encoder.fit(LANG_PREFIXES)
        y_data = l_encoder.transform(y_data)

        X, y = shuffle(x_data, y_data)

        self.classifier = svm.SVC(probability=True)
        self.classifier.fit(X, y)
        dump(self.classifier, "classifier_model.joblib")

    def predict(self, data):
        es_back, de_back = self.backtranslations(data)
        real_x = self.bleu_from_backtrans([es_back, de_back])
        return self.classifier.predict_proba(real_x)


if __name__ == "__main__":
    detector = Detector()
    test = ["Occasionally he would stop in the shade of a poplar tree and raise his head as if he were venting that humid and imperceptible air for men, but that the delicate smell of the canine race indicates the source or the coveted puddle where to quench his thirst."]
    print(detector.predict(test))
    # detector.train("data/svm_train.txt", "data/svm_lang.txt")
