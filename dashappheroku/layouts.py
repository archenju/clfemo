import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import dash_table

#import dash
#import plotly.express as px
import plotly.graph_objects as go
import app

import numpy as np
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


##### NLTK does not work with Heroku #####
#import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk import word_tokenize
#from nltk.corpus import stopwords
#lemma = WordNetLemmatizer()
#stop_words = nltk.corpus.stopwords.words("english")
#exclude = set(string.punctuation)
#stop_words.extend(exclude)
#corpus = [each.lower() for each in corpus]
#corpus = [word_tokenize(each) for each in corpus]
#corpus = [[lemma.lemmatize(word) for word in each if word not in stop_words] for each in corpus]
#corpus = [' '.join(each) for each in corpus]



dataEF = pd.read_csv("./data/Emotion_final.csv")
dataTE = pd.read_csv('./data/text_emotion.csv')
dataModels = pd.read_csv('./data/models.csv')
dataModels2 = pd.read_csv('./data/models2.csv')

corpus = np.array(dataEF["Text"])
targets = list(dataEF["Emotion"].map({'anger':0, 'fear':1, 'happy':2, 'love':3, 'sadness':4, 'surprise':5}))

corpus2 = np.array(dataTE["content"])
targets2 = list(dataTE["sentiment"].map({'anger':0, 'worry':1, 'happiness':2, 'love':3, 'sadness':4, 'surprise':5
                                       , 'neutral':6, 'fun':7, 'relief':8, 'hate':9, 'empty':10,
                                       'enthusiasm':11, 'boredom':12}))

########################## Figure 1 ##########################################
def subsample(x, step=150):
    return np.hstack((x[:30], x[10::step]))

def makefig1(X, words):
    wordsum = np.array(X.sum(0))[0]
    ix = wordsum.argsort()[::-1]
    wordrank = wordsum[ix] 
    labels = [words[i] for i in ix]
    
    freq = subsample(wordrank)
    r = np.arange(len(freq))
    trace = go.Bar(
                    x = r,
                    y = freq,
                    name = "",
                    marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                                 line = dict(color ='rgb(0,0,0)',width =2.5)),
                    text = subsample(labels))
    layout = go.Layout(barmode = "group",
                      title = 'Most frequent words ',
                      yaxis = dict(title = 'word frequncy'),
                      xaxis = dict(title = 'word rank'))
    return go.Figure(data = trace, layout = layout)


vectorizer = CountVectorizer(stop_words= 'english')
X = vectorizer.fit_transform(corpus)
words = vectorizer.get_feature_names()

vectorizer2 = CountVectorizer(stop_words='english')
X2 = vectorizer.fit_transform(corpus2)
words2 = vectorizer.get_feature_names()

fig1 = makefig1(X, words)
fig1b = makefig1(X2, words2)
########################## Figure 2 ##########################################
dftableau = dataEF
ParGroup = dftableau.groupby('Emotion')
y1=ParGroup['Emotion'].size().sort_values(ascending=False)
trace1 = {
  'x': y1.index.get_level_values(0).tolist(),                
  'y': y1,
  'name': 'Income',
  'type': 'bar'
}
data = [trace1];
layout = {
  'xaxis': {'title': 'Emotions'},
  'yaxis': {'title': 'Occurence'},
  'barmode': 'relative',
  'title': 'Emotions ranked by occurence in Emotion_final'
}
fig2 = go.Figure(data = data, layout = layout)

########################## Figure 2b #########################################
ParGroup = dataTE.groupby('sentiment')
y1=ParGroup['sentiment'].size().sort_values(ascending=False)
trace2 = {
  'x': y1.index.get_level_values(0).tolist(),                
  'y': y1,
  'name': 'Income',
  'type': 'bar'
}
data2 = [trace2];
layout2 = {
  'xaxis': {'title': 'Emotions'},
  'yaxis': {'title': 'Occurence'},
  'barmode': 'relative',
  'title': 'Emotions ranked by occurence in text_emotion'
}
fig2b = go.Figure(data = data2, layout = layout2)
##############################################################################
mdown1 = '''
We processed the corpus using 5 classifiers: 
SGDClassifier ("sgd"), SVC ("svm"), LinearSVC ("svml"), LogisticRegression ("logit"), and KNeighborsClassifier ("knn").

Each one was tested with and without a TfidfTransformer ("tfidf") to check if adding it would improve their result.

A second run was executed on the two most promising models with N-grams ("ngram") using the parameter (1,2).

From these tests, we can observe that TFIDF and N-grams offer only marginal improvements, 
with vect-ngram-tfidf-svml, vect-ngram-sgd, and vect-tfidf-svml being the best models.

'''

mdown2 = '''
For the second CSV, fewer classifiers were used. SVC was dropped because of its slowness, 
and all remaining classifiers were used in combination with a TfidfTransformer.

The set of classifiers were run twice, once without preprocessing and once with. 
In both cases the results were subpar for all classifiers, with test scores ranging between 22% and 35%.
'''


layoutHome = html.Div([
    html.H3('HomePage'),
    dcc.Link('Table des données', href='/page-1'),
    html.Br(),
    dcc.Link("Cas d'étude.", href='/page-2')
])

layout2 = html.Div([
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    html.H3('Models tested on Emotion_final', style={'textAlign':'center'}),
    html.Br(),
    dash_table.DataTable(
        id='table-3',
        columns=[{"name": i, "id": i} for i in dataModels.columns],
        data=dataModels.to_dict('records'),
        fixed_rows={'headers': True},
        style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'900px'},
        style_cell_conditional=[{'height': 'auto',
            'minWidth': '80px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal','textAlign':'center'}
        ]

    ),   
    html.Br(),
    html.Div([dcc.Markdown(children=mdown1)]),
    html.Br(),
    html.Div(html.Img(src=app.app.get_asset_url('vec-sgd.jpg'))),
    html.Div(html.Img(src=app.app.get_asset_url('vec-ngram-sgd.jpg'))),
    html.Div(html.Img(src=app.app.get_asset_url('vect-ngram-tfidf-svml.jpg'))),
    html.Br(),
    html.H3('Models tested on text_emotion', style={'textAlign':'center'}),
    html.Br(),
    dash_table.DataTable(
        id='table-4',
        columns=[{"name": i, "id": i} for i in dataModels2.columns],
        data=dataModels2.to_dict('records'),
        fixed_rows={'headers': True},
        style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'900px'},
        style_cell_conditional=[{'height': 'auto',
            'minWidth': '80px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal','textAlign':'center'}
        ]

    ),   
    html.Br(),
    html.Div([dcc.Markdown(children=mdown2)]),
    html.Br(),
    html.Div(id='page-2-content')
    
])


layout1a = html.Div([
    html.H3('Emotion_final data set', style={'textAlign':'center'}),
    html.Br(),
    dash_table.DataTable(
        id='table-1',
        columns=[{"name": i, "id": i} for i in dataEF.columns],
        data=dataEF.to_dict('records'),
        export_format='csv',
        fixed_rows={'headers': True},
        style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'900px'},
        style_cell_conditional=[{'height': 'auto',
            'minWidth': '80px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal','textAlign':'center'}
        ]
    ),
    html.Br(),
    html.Br(),
    html.Div([dcc.Graph(id='fig1', figure=fig1)]),
    html.Br(),
    html.Div([dcc.Graph(id='fig2', figure=fig2)]),
    html.Br(),
    dcc.RadioItems(
                id='input-figemo',
                options=[{'label': i, 'value': i} for i in ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']],
                value='anger',
                labelStyle={'display': 'inline-block'}
            ),
    html.Br(),
    html.Div([dcc.Graph(id='figemo')])
    ])

layout1b = html.Div([
    html.H3('text_emotion data set', style={'textAlign':'center'}),
    html.Br(),
    dash_table.DataTable(
        id='table-1',
        columns=[{"name": i, "id": i} for i in dataTE.columns],
        data=dataTE.to_dict('records'),
        export_format='csv',
        fixed_rows={'headers': True},
        style_table={'overflowX': 'auto','overflowY': 'auto','maxHeight':'900px'},
        style_cell_conditional=[{'height': 'auto',
            'minWidth': '80px', 'width': '120px', 'maxWidth': '180px',
            'whiteSpace': 'normal','textAlign':'center'}
        ]
    ),
    html.Br(),
    html.Br(),
    html.Div([dcc.Graph(id='fig1b', figure=fig1b)]),
    html.Br(),
    html.Div([dcc.Graph(id='fig2b', figure=fig2b)]),
    html.Br(),
    dcc.RadioItems(
                id='input-figemo2',
                options=[{'label': i, 'value': i} for i in ['anger', 'worry', 'happiness', 'love', 'sadness', 'surprise'
                                       , 'neutral', 'fun', 'relief', 'hate', 'empty',
                                       'enthusiasm', 'boredom']],
                value='anger',
                labelStyle={'display': 'inline-block'}
            ),
    html.Br(),
    html.Div([dcc.Graph(id='figemo2')])
    ])

layout1 = html.Div([
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.RadioItems(
                id='input-type',
                options=[{'label': i, 'value': i} for i in ['Emotion_final', 'text_emotion']],
                value='Emotion_final',
                labelStyle={'display': 'inline-block'}
            ),
    html.Br(),
    html.Div(id='output-type'),
    html.Br(),
    html.H3('', style={'textAlign':'center'}),
    html.Br(),
    html.Div(id='page-1-content')
],)