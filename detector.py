import torch
import requests
import numpy as np
from nn_classifier import Classifier
from pyonmttok import Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from joblib import load

LANG_POOL = ['en', 'es', 'fr', 'de']


def translate(text, src, tgt):
    response = requests.post(
        'http://localhost:8080/translate', data={'text': text, 'src': src, 'tgt': tgt})
    return response.text


class Detector:
    def __init__(self, model_paths={"nn": "./models/classifier_3k_v3_4000.pth",
                                    "svm": "./models/svm.joblib",
                                    "knn": "./models/knn.joblib",
                                    "dt": "./models/dt.joblib"}):
        self.tokenizer = Tokenizer('conservative')
        self.smoothing = SmoothingFunction()
        self.classifiers = {key: self.load_sklearn_classifier(
            val) for key, val in model_paths.items() if key != "nn"}
        self.classifiers["nn"] = self.load_nn_classifier(model_paths["nn"])
        self.len_norm = self.load_sklearn_classifier(
            "./models/len_norm.joblib")
        self.src_norm = self.load_sklearn_classifier(
            "./models/src_norm.joblib")

    def load_sklearn_classifier(self, model_path):
        try:
            return load(model_path)
        except FileNotFoundError:
            return None

    def load_nn_classifier(self, model_path):
        classifier = Classifier()
        classifier.load_state_dict(torch.load(model_path))
        classifier.float()
        return classifier

    def predict(self, text, src, algorithm):
        other_langs = [l for l in LANG_POOL if l != src]
        results = {lang: [text] for lang in other_langs}
        data = []

        last_back = text
        for lang in other_langs:
            for _ in range(0, 2):
                translation = translate(last_back, src, lang)
                last_back = translate(translation, lang, src)
                results[lang].append(last_back)

        for lang in LANG_POOL:
            if lang != src:
                for i in range(0, 2):
                    ref_sent, _ = self.tokenizer.tokenize(results[lang][i])
                    hypothesis, _ = self.tokenizer.tokenize(
                        results[lang][i + 1])
                    bleu = sentence_bleu(
                        [ref_sent], hypothesis, smoothing_function=self.smoothing.method4, weights=(0.25, 0.25, 0.25, 0.25))
                    data.append(bleu)
            else:
                data += [0.0, 0.0]

        data.append(self.src_norm.transform(
            np.array([LANG_POOL.index(src)]).reshape(-1, 1))[0][0])
        data.append(self.len_norm.transform(
            np.array([len(self.tokenizer.tokenize(text)[0])]).reshape(-1, 1))[0][0])

        data = np.array(data).reshape(1, -1)

        prediction = None
        if algorithm != "nn":
            prediction = self.classifiers[algorithm].predict_proba(data)
        else:
            tensor = torch.from_numpy(data)
            prediction = self.classifiers["nn"](tensor.float()).tolist()

        return list(prediction[0])


if __name__ == "__main__":
    dummy_detector = Detector()
    text = "Podemos ver aquí que todo el proceso empezó demasiado tarde. En tercer lugar, para detener todas las negociaciones sobre cuestiones de migración con países que no facilitan ninguna garantía de respeto por los derechos humanos; Señor Presidente, señor Comisario, estos puntos son los puntos que quería hacer. Como ha señalado el Sr. Ford, el Sr. Spidla ha realizado una valiosa contribución al actual proceso de paz en Burundi, y espero que se realice una investigación plena en su sacrificio que haga honor a su muerte. Estas han sido sometidas a petición de la propia Comisión y se han referido al año financiero en 1999, la más reciente que era la información final. Después, debemos centrarnos en la reducción de las temperaturas, siempre que sigan el camino. Este fue un primer paso positivo."
    print(dummy_detector.predict(text, 'es', 'nn'))
