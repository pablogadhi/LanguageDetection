import os
import argparse
import pyonmttok


def learn_bpe(tokenizer, src_path, out_path):
    print("Feeding data to learner...")
    learner = pyonmttok.BPELearner(
        tokenizer=tokenizer, symbols=32000)
    learner.ingest_file(src_path)
    print("Learning BPE model...")
    learner.learn(out_path)
    print("BPE model learned!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Scrip para entrenar un modelo de BPE')

    parser.add_argument('--source', action='store',
                        type=str, help='source directory or file', required=True)
    parser.add_argument('--output', action='store',
                        type=str, help='output file path', required=True)
    parser.add_argument('--share_src', action='store_true',
                        help='use all files in src directory to train the same model', required=False)
    parser.add_argument('--symbols', action='store',
                        type=int, help='amount of symbols to use', required=False, default=32000)

    args = parser.parse_args()
    tokenizer = pyonmttok.Tokenizer('conservative', joiner_annotate=True)
    learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=args.symbols)

    print("Learning BPE model...")
    if os.path.isdir(args.source) and args.share_src:
        for file in os.listdir(args.source):
            file_path = os.path.join(args.source, file)
            if os.path.isfile(file_path):
                learner.ingest_file(file_path)
        learner.learn(args.output)

    elif os.path.isdir(args.source):
        for file in os.listdir(args.source):
            file_path = os.path.join(args.source, file)
            if os.path.isfile(file_path):
                learner.ingest_file(file_path)
                # TODO Save a diferent model for each input file
                learner.learn(args.output)
                learner = pyonmttok.BPELearner(
                    tokenizer=tokenizer, symbols=args.symbols)

    else:
        learner.ingest_file(args.source)
        learner.learn(args.output)
    print("BPE model learned!")
