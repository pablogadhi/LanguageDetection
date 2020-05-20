import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from detector import Detector

CATEGORIES = {
    0: "Español",
    1: "Alemán"
}

detector = Detector()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "LangDetector"

intro_md = '''
## Prueba de detección del idioma original de un texto
#### Ingresa un texto y presiona el botón para identificar el idioma en el que se encontraba el texto, antes de ser traducido.
'''

app.layout = html.Div(children=[
    dcc.Markdown(children=intro_md, className="md-title"),
    html.Div(children=[
        dcc.Textarea(
            id='text-input',
            style={'width': '100%', 'height': 200},
        ),
        html.Div(),
        html.Button(children='Detectar', id='detect-btn',
                    n_clicks=0, className="detect-btn"),
        html.Div(id='output-div-container', children="",
                 className="output-div-container")
    ], className="input-div")

], className="app-container")


def output_div(es_p, de_p, pred):
    return html.Div(children=[
        html.P("Probabilidad de que el texto esté en español: {}".format(
            es_p)),
        html.P("Probabilidad de que el texto esté en alemán: {}".format(
            de_p)),
        html.Div(),
        html.P("Predicción final: {}".format(pred))
    ], className="output-div")


@app.callback(
    Output('output-div-container', 'children'),
    [Input('detect-btn', 'n_clicks')],
    [State('text-input', 'value')]
)
def update_output_div(n_clicks, value):
    # TODO separate sentences
    if value != None and value != '':
        pred = detector.predict([value])[0].tolist()
        return output_div(pred[0], pred[1], CATEGORIES[pred.index(max(pred))])
    else:
        return ""


if __name__ == '__main__':
    app.run_server(debug=True)
