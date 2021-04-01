import yaml
import base64
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objs as go

from constants import CLASSES

# app.yaml
with open('app.yaml') as yaml_data:
    params = yaml.safe_load(yaml_data)
    
MODEL_NN = params['paths']['MODEL_NN']
MODEL_SVM = params['paths']['MODEL_SVM']

IMAGE_WIDTH = params['parameters']['IMAGE_WIDTH']
IMAGE_HEIGHT = params['parameters']['IMAGE_HEIGHT']

# Load DNN model
classifier = tf.keras.models.load_model(MODEL_NN)

def classify_image(image, model, image_box=None):
  """Classify image by model
  Parameters
  ----------
  content: image content
  model: tf/keras classifier
  Returns
  -------
  class id returned by model classifier
  """
  images_list = []
  image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), box=image_box)
                                        # box argument clips image to (x1, y1, x2, y2)
  image = np.array(image)
  images_list.append(image)
  
  return [np.argmax(model.predict(np.array(images_list))), model.predict(np.array(images_list))]


app = dash.Dash('Traffic Signs Recognition') #, external_stylesheets=dbc.themes.BOOTSTRAP)


pre_style = {
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all',
    'whiteSpace': 'normal'
}


# Define application layout
app.layout = html.Div([
    html.H1('Reconnaissance panneaux de signalisation', style = dict(textAlign='center')),
    html.Hr(),
    html.H2('Cette application dash app utilise un réseaux de neuronnes pour classer les images de panneaux de signalisation', style = dict(textAlign='center')),
    dcc.Upload(
        id='bouton-chargement',
        children=html.Div([
            'Cliquer-déposer ou ',
                    html.A('sélectionner une image')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '2px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'backgroundColor': '#1f77b4',
            'margin': '10px',
            'marginLeft': 'auto',
            'marginRight': 'auto',
        }
    ),
    html.Div(id='mon-image'),
    html.Div(id='ma-zone-resultat')
])

@app.callback(Output('mon-image', 'children'),
              [Input('bouton-chargement', 'contents')])

def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        if 'image' in content_type:
            image = Image.open(io.BytesIO(base64.b64decode(content_string)))
            predicted_class = classify_image(image, classifier)[0]
            proba = classify_image(image, classifier)[1][0]
            proba, classes_list = (list(t) for t in zip(*sorted(zip(proba, CLASSES.values()), reverse=True)))
            return html.Div([
                html.Img(src=contents),
                html.H3("Le modèle prédit avec une probabilité de {:.5f}, que le panneau est : {}".format(max(proba),CLASSES[predicted_class])),
                    html.Hr(),
                    html.Div([dcc.Graph(id='graph',figure={'data': [go.Bar(x=classes_list, y=proba)],"layout": {'title': "Probabilité attaché aux types de panneau"}
                                     })
                        ]),
                    html.Hr(),
                #html.Div('Raw Content'),
                #html.Pre(contents, style=pre_style)
            ])
        else:
            try:
                # Décodage de l'image transmise en base 64 (cas des fichiers ppm)
                # fichier base 64 --> image PIL
                image = Image.open(io.BytesIO(base64.b64decode(content_string)))
                # image PIL --> conversion PNG --> buffer mémoire 
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                # buffer mémoire --> image base 64
                buffer.seek(0)
                img_bytes = buffer.read()
                content_string = base64.b64encode(img_bytes).decode('ascii')
                predicted_class = classify_image(image, classifier)[0]
                proba = classify_image(image, classifier)[1][0]
                proba, classes_list = (list(t) for t in zip(*sorted(zip(proba, CLASSES.values()), reverse=True)))
                nb_class=np.arange(1,44)
                # Affichage de l'image
                return html.Div([
                    html.Img(src='data:image/png;base64,' + content_string),
                    html.H3("Le modèle prédit avec une probabilité de {:.5f}, que le panneau est : {}".format(max(proba),CLASSES[predicted_class])),
                    html.Hr(),
                    html.Div([dcc.Graph(id='graph',figure={'data': [go.Bar(x=classes_list, y=proba)],"layout": {'title': "Probabilité attaché aux types de panneau"}
                                     })
                        ]),
                    html.Hr(),
                ])
            except:
                return html.Div([
                    html.Hr(),
                    html.Div('Uniquement des images svp : {}'.format(content_type)),
                    html.Hr(),                
                    html.Div('Raw Content'),
                    html.Pre(contents, style=pre_style)
                ])
           


# Start the application
if __name__ == '__main__':
    app.run_server(debug=True)
