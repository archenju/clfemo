from dash.dependencies import Input, Output

from app import app

import layouts
from layouts import dataEF, dataTE
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go

# @app.callback(
#     Output('app-home-display-value', 'children'),
#     Input('app-home-dropdown', 'value'))
# def display_value(value):
#     return 'You have selected "{}"'.format(value)

@app.callback(
    Output('output-type', 'children'),
    Input('input-type', 'value'))
def display_table(value):
    if value == "Emotion_final":
        return layouts.layout1a
    else:
        return layouts.layout1b

def subsample(x, step=150):
    return np.hstack((x[:30], x[10::step]))

@app.callback(
    Output('figemo', 'figure'),
    Input('input-figemo', 'value'))
def display_emo(value):
    # return layouts.print_words_singleemo(value)
    dataEF_emo = dataEF[dataEF['Emotion'] == value]
    
    corpus1 = np.array(dataEF_emo["Text"])
    vectorizer = CountVectorizer(stop_words= 'english')
    Xs = vectorizer.fit_transform(corpus1)
    words = vectorizer.get_feature_names()

    wordsum = np.array(Xs.sum(0))[0]
    ix = wordsum.argsort()[::-1]
    wordrank = wordsum[ix] 
    labels1 = [words[i] for i in ix]

    freq = subsample(wordrank)
    r = np.arange(len(freq))

    trace = go.Bar(
                    x = r,
                    y = freq,
                    name = "",
                    marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                                 line = dict(color ='rgb(0,0,0)',width =2.5)),
                    text = subsample(labels1))

    layout = go.Layout(barmode = "group",
                      title = 'Most frequent words for '+value+'',
                      yaxis = dict(title = 'word frequncy'),
                      xaxis = dict(title = 'word rank'))
    fig = go.Figure(data = trace, layout = layout)
    return fig



@app.callback(
    Output('figemo2', 'figure'),
    Input('input-figemo2', 'value'))
def display_emo2(value):
    # return layouts.print_words_singleemo(value)
    dataEF_emo = dataTE[dataTE['sentiment'] == value]
    
    corpus1 = np.array(dataEF_emo["content"])
    vectorizer = CountVectorizer(stop_words= 'english')
    Xs = vectorizer.fit_transform(corpus1)
    words = vectorizer.get_feature_names()

    wordsum = np.array(Xs.sum(0))[0]
    ix = wordsum.argsort()[::-1]
    wordrank = wordsum[ix] 
    labels1 = [words[i] for i in ix]

    freq = subsample(wordrank)
    r = np.arange(len(freq))

    trace = go.Bar(
                    x = r,
                    y = freq,
                    name = "",
                    marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                                 line = dict(color ='rgb(0,0,0)',width =2.5)),
                    text = subsample(labels1))

    layout = go.Layout(barmode = "group",
                      title = 'Most frequent words for '+value+'',
                      yaxis = dict(title = 'word frequncy'),
                      xaxis = dict(title = 'word rank'))
    fig = go.Figure(data = trace, layout = layout)
    return fig


# @app.callback(
#     Output('app-2-display-value', 'children'),
#     Input('app-2-dropdown', 'value'))
# def display_value2(value):
#     return 'You have selected "{}"'.format(value)