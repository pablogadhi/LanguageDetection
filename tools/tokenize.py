import os
import argparse
import pyonmttok


def remove_ext(path):
    return '.'.join((path.split('.')[:-1]))


def out_transform(path, new_ext):
    return remove_ext(path) + new_ext


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Wrapper para la libreria de tokenizacion de ONMT')
    # General options
    parser.add_argument('--source', action='store',
                        type=str, help='source directory or file', required=True)
    parser.add_argument('--output', action='store',
                        type=str, help='directory to store the output', required=True, default='')
    parser.add_argument('--out_ext', action='store',
                        type=str, help='extension of the output file', required=False, default='.txt')
    parser.add_argument('--reverse', action='store_true',
                        help='detokenize source instead of tokenizing it (works with files only)', required=False)
    parser.add_argument('--lang_tokens', action='store',
                        type=str, help='file with the tokens used to translate a file, this file will be used to load a specific bpe model for each sentence', required=False, default='')

    # Tokenizer options
    parser.add_argument('--type', action='store', type=str,
                        help='tokenization type', required=False, default='conservative')
    parser.add_argument('--joiner_annotate', action='store_true',
                        help='joiner annotate', required=False)
    parser.add_argument('--bpe_model', action='store', type=str,
                        help='BPE model to use', required=False, default='')

    args = parser.parse_args()

    if args.bpe_model != '':
        print("Using the following model(s): {}".format(args.bpe_model))

    use_single_bpe = os.path.isfile(args.bpe_model) or args.bpe_model == ''

    tokenizer = pyonmttok.Tokenizer(
        args.type, joiner_annotate=args.joiner_annotate, bpe_model_path=args.bpe_model) if use_single_bpe else None

    print("Tokenizing...")
    if os.path.isdir(args.source):
        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        for file in os.listdir(args.source):
            src_path = os.path.join(args.source, file)
            if os.path.isfile(src_path):
                file_name = out_transform(file, args.out_ext)
                out_path = os.path.join(args.output, file_name)
                if not use_single_bpe:
                    m_path_parts = args.bpe_model.split('/')
                    model_name = "{}_{}".format(
                        m_path_parts[-1], remove_ext(file_name))
                    model_path = os.path.join(
                        '/'.join(m_path_parts[:-1]), model_name)
                    print("Specific model loaded: {}".format(model_path))
                    tokenizer = pyonmttok.Tokenizer(
                        args.type, joiner_annotate=args.joiner_annotate, bpe_model_path=model_path)
                tokenizer.tokenize_file(src_path, out_path, 4)

    else:
        if not args.reverse:
            tokenizer.tokenize_file(args.source, args.output, 4)
        else:
            if use_single_bpe:
                if not os.path.isfile(args.output):
                    open(args.output, 'x')
                tokenizer.detokenize_file(args.source, args.output)
            else:
                if args.lang_tokens == '':
                    print(
                        "A token file is needed to decide with bpe model should be used for each sentence!")
                else:
                    tokens = [
                        tok.split('_')[-1] for tok in open(args.lang_tokens, 'r').read().split('\n')]
                    sentences = [sent.split(' ') for sent in open(
                        args.source, 'r').read().split('\n')]

                    # Remove any trailing newline from the sentences
                    if sentences[-1] == ['']:
                        sentences = sentences[:-1]
                        tokens = tokens[:-1]

                    unique = list(set(tokens))
                    # Remove the empty string from the unique model names
                    try:
                        unique.remove('')
                    except ValueError:
                        pass

                    tokenizer_dict = {}
                    for lang in unique:
                        bpe_path = args.bpe_model + "_" + lang
                        tokenizer_dict[lang] = pyonmttok.Tokenizer(
                            args.type, joiner_annotate=args.joiner_annotate, bpe_model_path=bpe_path)

                    detok_sentences = [tokenizer_dict[tok].detokenize(
                        sent) for sent, tok in zip(sentences, tokens)]
                    open(args.output, 'w').write(
                        '\n'.join(detok_sentences))

    print("Tokenization Finished!")
