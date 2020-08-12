import os
import argparse
import pyonmttok


def out_transform(path, ext):
    return '.'.join((path.split('.')[:-1])) + ext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wrapper para la libreria de tokenizacion de ONMT')
    # General options
    parser.add_argument('--source', action='store',
                        type=str, help='source directory or file', required=True)
    parser.add_argument('--out_ext', action='store',
                        type=str, help='extension of the output file', required=True)
    parser.add_argument('--subdir', action='store',
                        type=str, help='sub-directory to store the output', required=False, default='')

    # Tokenizer options
    parser.add_argument('--type', action='store', type=str,
                        help='tokenization type', required=False, default='conservative')
    parser.add_argument('--joiner_annotate', action='store_true',
                        help='joiner annotate', required=False)
    parser.add_argument('--bpe_model', action='store', type=str,
                        help='BPE model to use', required=False, default='')

    args = parser.parse_args()

    if args.bpe_model != '':
        print("Using the following model: {}".format(args.bpe_model))

    tokenizer = pyonmttok.Tokenizer(
        args.type, joiner_annotate=args.joiner_annotate, bpe_model_path=args.bpe_model)

    print("Tokenizing...")
    if os.path.isdir(args.source):
        if args.subdir != '' and args.subdir not in os.listdir(args.source):
            os.mkdir(os.path.join(args.source, args.subdir))

        for file in os.listdir(args.source):
            src_path = os.path.join(args.source, file)
            if os.path.isfile(src_path):
                out_path = os.path.join(args.source, os.path.join(
                    args.subdir, out_transform(file, args.out_ext)))
                tokenizer.tokenize_file(src_path, out_path, 4)
    else:
        path_parts = args.source.split('/')
        parent_dir = '/'.join(path_parts[:-1])

        if args.subdir != '' and args.subdir not in os.listdir(parent_dir):
            os.mkdir(os.path.join(parent_dir, args.subdir))

        out_path = os.path.join(parent_dir, os.path.join(
            args.subdir, out_transform(path_parts[-1], args.out_ext)))
        tokenizer.tokenize_file(args.source, out_path, 4)

    print("Tokenization Finished!")
