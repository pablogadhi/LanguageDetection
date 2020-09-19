import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pyonmttok import Tokenizer


def bleu(ref_sentence, hypothesis, **kwargs):
    return sentence_bleu([ref_sentence], hypothesis, smoothing_function=kwargs['smoothing'].method3)


def meteor(ref_sentence, hypothesis, **kwargs):
    return meteor_score([ref_sentence], hypothesis)


def ter(ref_sentence, hypothesis, **kwargs):
    pass


def measure_predictions(metric, metric_kwargs, predictions, targets, lang_tokens):
    scores = {}
    for tok in set(lang_tokens):
        if tok != '':
            scores[tok] = []

    for pred, tgt, tok in zip(predictions, targets, lang_tokens):
        scores[tok].append(metric(tgt, pred, **metric_kwargs))

    return {key: sum(value)/len(value) for (key, value) in scores.items()}


def file_to_list(file_name):
    file = open(file_name, 'r')
    return file.read().split('\n')


def tokenize_list(list):
    tokenizer = Tokenizer('conservative')
    return [tokenizer.tokenize(line)[0] for line in list]


def load_files(pred_file, tgt_file, lt_file):
    predictions = file_to_list(pred_file)
    targets = file_to_list(tgt_file)
    lang_tokens = file_to_list(lt_file) if lt_file != '' else [
        'total'] * len(predictions)
    return predictions, targets, lang_tokens


def print_scores(scores, metric_type):
    for key, val in [(k, v) for k, v in sorted(scores.items(), key=lambda item: item[0])]:
        print("{} score for {}: {:.3f}".format(metric_type, key, val))

    if len(scores.values()) > 1:
        total = sum(scores.values())/len(scores.values())
        print("\nAverage {} score: {:.3f}".format(metric_type, total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to compute de BLEU, METEOR or TER scores of a translation')

    parser.add_argument('--type', action='store', choices={"BLEU", "METEOR", "TER"},
                        type=str, help='type of metric to use, choices are: BLEU, METEOR, and TER', required=True)
    parser.add_argument('--pred', action='store',
                        type=str, help='file with a prediction in every line', required=True)
    parser.add_argument('--target', action='store',
                        type=str, help='file with the real sentences', required=True)
    parser.add_argument('--lang_tokens', action='store',
                        type=str, help='file with the translation tokens, used to separate the scores by language', required=False, default='')

    args = parser.parse_args()
    predictions, targets, lang_tokens = load_files(
        args.pred, args.target, args.lang_tokens)

    metric = None
    metric_kwargs = {}
    if args.type == 'BLEU':
        metric = bleu
        metric_kwargs['smoothing'] = SmoothingFunction()
        predictions = tokenize_list(predictions)
        targets = tokenize_list(targets)
    elif args.type == 'METEOR':
        metric = meteor
    elif args.type == 'TER':
        metric = ter

    scores = measure_predictions(
        metric, metric_kwargs, predictions, targets, lang_tokens)
    print_scores(scores, args.type)
