import waitress
import argparse
from flask import Flask, request
from translator import Translator


def start_app(beam_size, max_batch_size):
    app = Flask(__name__)
    translator = Translator(2, 6, beam_size, max_batch_size)
    # Translate something to initialize all objects
    print(translator.translate('es', 'en', 'Traductor activado!'))

    @app.route('/translate', methods=['POST'])
    def translate():
        if request.method == 'POST':
            return translator.translate(request.form.get('src'), request.form.get('tgt'), request.form.get('text'))
        else:
            # TODO Add ivalid method error
            pass

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Translation server that handles requests to translate sentences between english, spanish, french and german.')

    parser.add_argument('--beam_size', action='store',
                        type=int, help='size of the decoding beam', required=False, default=2)
    parser.add_argument('--batch_size', action='store',
                        type=int, help='maximum batch size for the translation process', required=False, default=30)
    args = parser.parse_args()
    app = start_app(args.beam_size, args.batch_size)
    waitress.serve(app, host='0.0.0.0', port=8080)
    # app.run(debug=True, host='0.0.0.0', port=8080)
