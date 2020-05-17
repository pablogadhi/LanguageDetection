import argparse


def translate_text(client, text, t_lang):
    return client.translate(text, target_language=t_lang)


if __name__ == "__main__":
    # Get arguments
    a_parser = argparse.ArgumentParser(
        description='Traduce archivos utilizando Google Translate')
    a_parser.add_argument('-f', action='store', type=str,
                          help='Name of the file')
    args = a_parser.parse_args()

    print(args)
