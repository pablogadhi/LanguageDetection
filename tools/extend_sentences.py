import argparse
from pyonmttok import Tokenizer


def extend_file(file, size):
    tokenizer = Tokenizer('conservative')
    sentences = file.read().split('\n')
    new_sentence = ""
    new_set = []
    while sentences:
        tok_sent, _ = tokenizer.tokenize(new_sentence)
        if len(tok_sent) < size:
            sent = sentences[0]
            sentences.remove(sent)
            if new_sentence != "":
                new_sentence += " "
            new_sentence += sent
        else:
            new_set.append(new_sentence)
            new_sentence = ""

    if new_sentence != "":
        new_set.append(new_sentence)

    return '\n'.join(new_set)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Extend each row to a minimum amount of words, concatenating nearby sentences")
    parser.add_argument('--src', action='store',
                        type=str, help='source language', required=True)
    parser.add_argument('--output', action='store',
                        type=str, help='output file', required=True)
    parser.add_argument('--min_size', action='store', type=int,
                        help='minimum size of each sentence', required=True)
    return parser


if __name__ == "__main__":
    arg_parser = setup_parser()
    args = arg_parser.parse_args()
    src_file = open(args.src, 'r')
    new_content = extend_file(src_file, args.min_size)
    out_file = open(args.output, 'w')
    out_file.write(new_content)
