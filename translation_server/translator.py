import os


class Translator:
    def __init__(self, inter_threads=1, intra_threads=4, beam_size=2, max_batch_size=30):
        self.translator = self.load_translator(inter_threads, intra_threads)
        self.tokenizer = Tokenizer('models/bpe')
        self.beam_size = beam_size
        self.max_batch_size = max_batch_size
        print('Setting translation engine up with a beam size of {} and a batch size of {}.'.format(
            beam_size, max_batch_size))

    def translate(self, src, tgt, text):
        batch = self.tokenizer.tokenize(src, tgt, text.split('\n'))
        translation = self.translator.translate_batch(
            batch, beam_size=self.beam_size, max_batch_size=self.max_batch_size)
        return self.tokenizer.detokenize(translation, tgt)

    def load_translator(self, inter_threads, intra_threads):
        import ctranslate2
        translator = ctranslate2.Translator(
            './models/multi_lang_diff_bpe_2',
            device='auto',
            device_index=0,
            inter_threads=inter_threads,
            intra_threads=intra_threads,
            compute_type="default")

        print('Using device:', translator.device)

        return translator


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

    def detokenize(self, text, tgt):
        return '\n'.join([self.tokenizer(tgt).detokenize(translation[0]['tokens']) for translation in text])


if __name__ == "__main__":
    test_translator = Translator()
    print(test_translator.translate('en', 'es',
                                    'In any case, it is scheduled that aid for orders will come to an end in December next year and I do not think we can reconcile ourselves to a laissez-faire policy.'))
