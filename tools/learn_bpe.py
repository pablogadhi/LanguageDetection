import os
import argparse
import pyonmttok


def iterate_files_in_dir(src_dir, execution):
    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)
        if os.path.isfile(file_path):
            execution(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This script trains a BPE model')

    parser.add_argument('--source', action='store',
                        type=str, help='source directory or file', required=True)
    parser.add_argument('--output', action='store',
                        type=str, help='output file path or prefix if using a separate model for each file in a directory', required=True)
    parser.add_argument('--share_src', action='store_true',
                        help='use all files in src directory to train the same model', required=False)
    parser.add_argument('--symbols', action='store',
                        type=int, help='amount of symbols to use', required=False, default=32000)

    args = parser.parse_args()
    tokenizer = pyonmttok.Tokenizer('conservative', joiner_annotate=True)
    learner = pyonmttok.BPELearner(tokenizer=tokenizer, symbols=args.symbols)

    print("Learning BPE model(s)...")
    if os.path.isdir(args.source):
        if args.share_src:
            iterate_files_in_dir(args.source, learner.ingest_file)
            learner.learn(args.output)
        else:
            def learn_model_for_file(file_path):
                self_learner = pyonmttok.BPELearner(
                    tokenizer=tokenizer, symbols=args.symbols)
                self_learner.ingest_file(file_path)
                name = "".join(file_path.split("/")[-1].split(".")[:-1])
                self_learner.learn("{}_{}".format(args.output, name))

            iterate_files_in_dir(args.source, learn_model_for_file)
    else:
        learner.ingest_file(args.source)
        learner.learn(args.output)
    print("BPE model(s) learned!")
