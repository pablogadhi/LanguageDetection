import argparse
from googletrans import Translator

if __name__ == "__main__":
    # Get arguments
    a_parser = argparse.ArgumentParser(
        description='Traduce archivos utilizando Google Translate')
    a_parser.add_argument('-s', action='store', type=str,
                          help='source file', required=True)
    a_parser.add_argument('-o', action='store', type=str,
                          help='output file', required=True)
    a_parser.add_argument('-tl', action='store', type=str,
                          help='target language', required=True)
    args = a_parser.parse_args()

    with open(args.s, "r") as file:
        sentences = file.read().split("\n")

        translator = Translator()
        translations = translator.translate(sentences, args.tl)

        out_file = open(args.o, "w")
        out_file.write("\n".join(map(lambda x: x.text, translations)))
