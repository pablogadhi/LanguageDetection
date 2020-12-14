import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State
from detector import Detector, translate
from langdetect import detect as get_lang
import assets.strings as strings

CATEGORIES = ["En", "Es", "Fr", "De"]

detector = Detector()
bleu_table = pd.read_csv('assets/final_bleu_scores.csv').round(3)
meteor_table = pd.read_csv('assets/final_meteor_scores.csv').round(3)
example_table = pd.read_csv('assets/gen_table_example.csv').round(3)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "LangDetector"

intro_md = open('assets/intro.md').read()

app.layout = html.Div(children=[
    dcc.Markdown(children=intro_md, className='md-intro'),
    html.Div(
        className='tables-container',
        children=[
            html.H5(strings.BLEU_SCORES, className='table-title'),
            dash_table.DataTable(
                id='bleu_table',
                columns=[{"name": i, "id": i} for i in bleu_table.columns],
                data=bleu_table.to_dict('records'),
                style_cell={'textAlign': 'center'},
            ),
            html.H5(strings.METEOR_SCORES, className='table-title'),
            dash_table.DataTable(
                id='meteor_table',
                columns=[{"name": i, "id": i} for i in meteor_table.columns],
                data=meteor_table.to_dict('records'),
                style_cell={'textAlign': 'center'},
            )
        ]
    ),
    html.P(strings.TABLE_GEN_EXPL),
    html.Img(src='assets/Back-Translations.png'),
    html.P(strings.TABLE_EXAMPLE, style={'marginTop': '2rem'}),
    dash_table.DataTable(id='example_table', columns=[
                         {"name": i, "id": i} for i in example_table.columns], data=example_table.to_dict('records'), style_cell={'textAlign': 'center', 'marginLeft': '1em', 'marginRight': '1em'}),
    html.P(strings.CLASSIFIER_INTRO),
    html.Img(src='assets/scores.png'),
    html.H2(strings.TEST_AREA, style={
            'marginTop': '4em', 'fontWeight': 'bold'}),
    html.Div(
        children=[
            html.Div(children=[
                html.P(strings.TRANSLATE_DESCRIPTION),
                dcc.RadioItems(
                    id='lang-chooser',
                    options=[
                        {'label': 'En', 'value': 'en'},
                        {'label': 'Es', 'value': 'es'},
                        {'label': 'Fr', 'value': 'fr'},
                        {'label': 'De', 'value': 'de'},
                    ],
                    value='en',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Textarea(
                    id='translation-input',
                    style={'width': '100%', 'height': 200},
                ),
                html.Button(children=strings.TRANSLATE_BTN_LABEL, id='translate-btn',
                            n_clicks=0, className="input-btn"),
                dcc.Loading(id='translate-loader', type='circle', children=[
                    html.Div(id='translate-output-container', children="",
                             className="output-container")
                ], parent_className='loader')
            ], className="input-div"),
            html.Div(children=[
                html.P(strings.DETECT_DESCRIPTION),
                dcc.RadioItems(
                    id='algorithm-chooser',
                    options=[
                        {'label': 'Red Neuronal', 'value': 'nn'},
                        {'label': 'SVM', 'value': 'svm'},
                        {'label': 'Árbol de Decisiones', 'value': 'dt'},
                        {'label': 'KNN', 'value': 'knn'},
                    ],
                    value='nn',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Textarea(
                    id='detection-input',
                    style={'width': '100%', 'height': 200},
                ),
                html.Button(children=strings.DETECT_BTN_LABEL, id='detect-btn',
                            n_clicks=0, className="input-btn"),
                dcc.Loading(id='detect-loader', type='circle', children=[
                    html.Div(id='detect-output-container', children="",
                             className="output-container")
                ], parent_className='loader')
            ], className="input-div")
        ], className="testing-container"
    ),
], className="app-container")


def output_div(pred):
    choice = CATEGORIES[pred.index(max(pred))]

    return html.Div(children=[
        html.Div(strings.EN_PROBABILITY.format(
            pred[0])),
        html.Div(strings.ES_PROBABILITY.format(
            pred[1])),
        html.Div(strings.FR_PROBABILITY.format(
            pred[2])),
        html.Div(strings.DE_PROBABILITY.format(
            pred[3])),
        html.Div(),
        html.Div(strings.FINAL_PRED.format(choice),
                 className="prediction-choice")
    ], className="output-div")


@app.callback(
    Output('detect-output-container', 'children'),
    [Input('detect-btn', 'n_clicks')],
    [State('detection-input', 'value'), State('algorithm-chooser', 'value')]
)
def update_output_div(n_clicks, text, algorithm):
    if text is not None and text != '':
        src = get_lang(text)
        pred = detector.predict(text, src, algorithm)
        return output_div(pred)
    return ""


@app.callback(
    Output('translate-output-container', 'children'),
    [Input('translate-btn', 'n_clicks')],
    [State('translation-input', 'value'), State('lang-chooser', 'value')]
)
def update_translation_output(n_clicks, text, tgt):
    if text is not None and text != '':
        src = get_lang(text)
        return translate(text, src, tgt)
    return ""


if __name__ == '__main__':
    app.run_server(debug=True)
