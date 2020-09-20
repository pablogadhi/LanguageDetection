import os
import math
import argparse
from googletrans import Translator

MAX_BATCH_SIZE = 500


def translate_file(args):
    if not os.path.isfile('.google_trans_checkpoint'):
        open('.google_trans_checkpoint', 'w').write('')

    with open(args.source, "r") as file, open(args.output, 'a') as output, open('.google_trans_checkpoint', 'r+') as checkpoint:
        sentences = file.read().split("\n")
        # Remove trailing space if needed
        if sentences[-1] == '':
            sentences = sentences[:-1]
        iters = math.ceil(len(sentences)/MAX_BATCH_SIZE)

        checkpoint_content = checkpoint.read()
        start = int(checkpoint_content) if checkpoint_content != '' else 0

        translator = Translator(timeout=900000000)
        for i in range(start, iters):
            translations = translator.translate(
                sentences[i*MAX_BATCH_SIZE: (i+1)*MAX_BATCH_SIZE], args.tgt)
            output.write(
                '\n'.join(map(lambda x: x.text.replace('\u200b', ''), translations)))

            checkpoint.seek(0)
            checkpoint.write(str(i+1))
            checkpoint.truncate()

            if i != iters - 1:
                output.write('\n')
            else:
                os.remove('.google_trans_checkpoint')

            print("{} sentences translated for file {}".format(
                (i+1)*MAX_BATCH_SIZE, args.source))


if __name__ == "__main__":
    # Get arguments
    a_parser = argparse.ArgumentParser(
        description='Translate file using Google Translate')
    a_parser.add_argument('--source', action='store', type=str,
                          help='source file', required=True)
    a_parser.add_argument('--output', action='store', type=str,
                          help='output file', required=True)
    a_parser.add_argument('--tgt', action='store', type=str,
                          help='target language', required=True)
    args = a_parser.parse_args()
    translate_file(args)
