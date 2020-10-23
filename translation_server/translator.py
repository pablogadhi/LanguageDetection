import os
import time
import spacy


class Translator:
    def __init__(self, inter_threads=1, intra_threads=4, beam_size=4, max_batch_size=50):
        self.translator = self.load_translator(inter_threads, intra_threads)
        self.tokenizer = Tokenizer('models/bpe')
        self.beam_size = beam_size
        self.max_batch_size = max_batch_size
        self.paragraph_indices = [0]
        print('Setting translation engine up with a beam size of {} and a batch size of {}.'.format(
            beam_size, max_batch_size))

    def join_batches(self, paragraphs, src):
        nlp = spacy.load(src)
        batch = []
        for index, paragraph in enumerate(paragraphs):
            sentences = [s.text for s in nlp(paragraph).sents]
            if sentences:
                self.paragraph_indices.append(
                    len(sentences) + self.paragraph_indices[index])
                batch += sentences

        return batch

    def translate(self, src, tgt, text):
        start = time.time()
        batch = self.join_batches(text.split('\n'), src)
        end = time.time()
        print("Batching: ", end - start)
        start = time.time()
        tok_batch = self.tokenizer.tokenize(src, tgt, batch)
        end = time.time()
        print("Tokenizing: ", end - start)
        start = time.time()
        translations = self.translator.translate_batch(
            tok_batch, beam_size=self.beam_size, max_batch_size=self.max_batch_size)
        end = time.time()
        print("Translating: ", end - start)
        start = time.time()
        detok_translations = self.tokenizer.detokenize(
            translations, tgt, self.paragraph_indices)
        end = time.time()
        print("Detokenizing: ", end - start)
        self.clean()
        return detok_translations

    def load_translator(self, inter_threads, intra_threads):
        import ctranslate2
        translator = ctranslate2.Translator(
            './models/multi_lang_v2',
            device='auto',
            device_index=0,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            compute_type="default")

        print('Using device:', translator.device)

        return translator

    def clean(self):
        self.paragraph_indices = [0]


class Tokenizer:
    def __init__(self, bpe_path):
        self.core = self.init_core(bpe_path)
        self.tokenizer = self.init_tokenizer_chooser(os.path.isdir(bpe_path))

    def init_core(self, bpe_path):
        import pyonmttok
        if os.path.isdir(bpe_path):
            files = [(file.split('_')[-1], file)
                     for file in os.listdir(bpe_path)]
            return {key: pyonmttok.Tokenizer('conservative', joiner_annotate=True, bpe_model_path=os.path.join(bpe_path, file)) for key, file in files}
        return pyonmttok.Tokenizer('conservative', joiner_annotate=True, bpe_model_path=bpe_path)

    def init_tokenizer_chooser(self, mutimodel):
        if mutimodel:
            return lambda lang: self.core[lang]
        return lambda lang: self.core

    def tokenize(self, src, tgt, text_list):
        return [['_src_{}_tgt_{}'.format(src, tgt)] + self.tokenizer(src).tokenize(x)[0] for x in text_list]

    def detokenize(self, text, tgt, new_line_indices):
        detok_text = ""
        for index, translation in enumerate(text):
            detok_text += self.tokenizer(
                tgt).detokenize(translation[0]['tokens'])

            if index + 1 in new_line_indices:
                detok_text += "\n"
            else:
                detok_text += " "

        return detok_text


if __name__ == "__main__":
    test_translator = Translator()
    text = open('data/translated/fr_es.txt', 'r').read()
    # print(test_translator.translate('es', 'en',
    #                                 'Permítanme proponer algo que la Asamblea, en su sabiduría y libertad, podrá aceptar o rechazar. Reconozco que esta cuestión no se ve afectada por el presupuesto en esta ocasión, pero no deberíamos pasar por alto los cambios que se están produciendo en la esfera de la energía con Save y Altener. Es importante, como han dicho varios diputados, que hagamos todo lo posible por que se cumplan estas sanciones. Por otra parte, el etiquetado debe ser coherente. Ese es el espíritu, y la letra también, de los acuerdos de Tampere y del Tratado de Amsterdam, aunque habría que discutir el fundamento jurídico del procedimiento común de asilo y del estatuto.\nHa sido un trabajo inmenso, pero extraordinariamente satisfactorio. Creo que debemos preguntarnos qué significa realmente la responsabilidad económica. Señor Presidente, voy a renunciar a más de la mitad de mi tiempo, a fin de que sea posible celebrar la votación esta mañana. En el Parlamento Europeo, con la tolerancia y el respeto que nos dispensamos mutuamente, sin duda podemos llegar a un acuerdo sobre los modelos modernos de familia, que a la vez también sirva de ejemplo para los debates en los Estados miembros. Creo que la idea de crear un espacio europeo de libertad, seguridad y justicia surgió hace ahora unos 25 años.'))
    translation = test_translator.translate('es', 'en', text)
    open('translator_test.txt', 'w').write(translation)
