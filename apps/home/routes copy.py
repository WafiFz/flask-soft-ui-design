# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import pandas as pd
from apps.home import blueprint
from flask import render_template, request, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
import matplotlib.pyplot as plt
import re
from textblob import TextBlob
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans


@blueprint.route('/index')
# @login_required
def index():
    
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'https?:\/\/\S+', '', text) #menghapus link
        text = re.sub(r'@[\w_]+', '', text) #menghapus mention
        text = re.sub(r'[^a-zA-Z\s]', '', text) #menghapus special character

        tokens = text.split()
        tokens = [stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if len(token)>3]

        sentence = " ".join(tokens)
        return sentence

    # read Dataset
    df = pd.read_csv('apps/home/menantea_tweets_temp.csv', sep=';')
    dataset = df.copy()

    #  preprocessing data
    df = df.drop(columns=[kolom for kolom in df.columns if kolom != "full_text"], axis=1)
    df.reset_index(drop=True,inplace=True)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df["clean_text"] = df["full_text"].apply(clean)
    df.drop_duplicates(subset=['clean_text'], keep = 'first',inplace= True)
    data_processed = df.copy()

    data = {
        'dataset': dataset.to_html(classes='table table-striped table-bordered', table_id='dataset'),
        'data_processed': data_processed.to_html(classes='table table-striped table-bordered', table_id='data_prepocessed'),
    }
    return render_template('home/index.html', data=data, segment='index')


@blueprint.route('/<template>')
# @login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
