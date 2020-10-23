import requests
import argparse

TRANSLATION_SERVER = 'http://localhost:8080/translate'


def send_request(args):
    text = open(args.src_file, 'r').read()
    response = requests.post(TRANSLATION_SERVER, data={
                             'text': text, 'src': args.src, 'tgt': args.tgt})
    open(args.output, 'w').write(response.text)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Send the content of a given file to a translation server")
    parser.add_argument('--src_file', action='store',
                        type=str, help='file to translate', required=True)
    parser.add_argument('--src', action='store',
                        type=str, help='source language', required=True)
    parser.add_argument('--tgt', action='store',
                        type=str, help='target language', required=True)
    parser.add_argument('--output', action='store',
                        type=str, help='output file', required=True)
    return parser


if __name__ == "__main__":
    arg_parser = setup_parser()
    args = arg_parser.parse_args()
    send_request(args)
