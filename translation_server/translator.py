class Translator:
    def __init__(self, inter_threads=1, intra_threads=4, beam_size=2, max_batch_size=30):
        self.translator = self.load_translator(inter_threads, intra_threads)
        self.tokenizer = Tokenizer()
        self.beam_size = beam_size
        self.max_batch_size = max_batch_size
        print('Setting translation engine up with a beam size of {} and a batch size of {}.'.format(
            beam_size, max_batch_size))

    def translate(self, src, tgt, text):
        batch = self.tokenizer.tokenize(src, tgt, text.split('\n'))
        translation = self.translator.translate_batch(
            batch, beam_size=self.beam_size, max_batch_size=self.max_batch_size)
        return self.tokenizer.detokenize(translation)

    def load_translator(self, inter_threads, intra_threads):
        import ctranslate2
        translator = ctranslate2.Translator(
            './models/multi_lang_final',
            device='auto',
            device_index=0,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            compute_type="default")

        print('Using device:', translator.device)

        return translator


class Tokenizer:
    def __init__(self):
        import pyonmttok
        self.tokenizer = pyonmttok.Tokenizer(
            'conservative', joiner_annotate=True, bpe_model_path='models/bpe_model')

    def tokenize(self, src, tgt, text_list):
        return [['_src_{}_tgt_{}'.format(src, tgt)] + self.tokenizer.tokenize(x)[0] for x in text_list]

    def detokenize(self, text):
        return '\n'.join([self.tokenizer.detokenize(translation[0]['tokens']) for translation in text])


if __name__ == "__main__":
    test_translator = Translator()
    print(test_translator.translate('es', 'de', 'Es tiempo de cambiar!'))
