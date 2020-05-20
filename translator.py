import torch
import onmt.translate
import onmt.opts as opts
import onmt.inputters as inputters
from onmt.model_builder import load_test_model
from onmt.utils.parse import ArgumentParser
from onmt.translate.translator import build_translator
from onmt.utils.misc import split_corpus
from onmt.utils.logging import init_logger


class Translator:
    def __init__(self, lang_pair):
        gpu = 0 if torch.cuda.is_available() else -1

        # Fake arguments
        parser = ArgumentParser(conflict_handler='resolve')
        opts.translate_opts(parser)
        parser.add('--gpu', '-gpu', type=int,
                   default=gpu, help="Device to run on")
        parser.add('--model', '-model', dest='models', metavar='MODEL',
                   nargs='+', type=str, default=["models/{}-model_step_20000.pt".format(lang_pair)])
        parser.add('--output', '-output', type=str,
                   default="pred.txt")
        parser.add('--src', '-src', required=False,
                   help="Source sequence to decode (one line per "
                   "sequence)")
        parser.add('--verbose', '-verbose', action="store_true", default=False,
                   help='Print scores and predictions for each sentence')

        opt = parser.parse_args()

        self.core = build_translator(opt, report_score=True)

    def translate(self, src, tgt=None):

        # src_shards = split_corpus(src, 10000)
        # tgt_shards = split_corpus(tgt, 10000)
        # shard_pairs = zip(src_shards, tgt_shards)

        # logger = init_logger("")

        # for i, sent in enumerate(src):
        #     print(sent)
        #     logger.info("Translating shard %d." % i)
        # print(src_shard)
        return self.core.translate(
            src=src,
            tgt=None,
            src_dir="",
            batch_size=30,
            batch_type="sents",
            attn_debug=False,
            align_debug=False
        )


if __name__ == "__main__":
    translator = Translator("en-es")

    test = [b"Flirtation doesn't have to go somewhere; it certainly doesn't need to end up in bed",
            b"Hello from the other side."]
    # test = "data/test.txt"
    print(translator.translate(test, None))
