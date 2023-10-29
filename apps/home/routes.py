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


@blueprint.route('/')
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

    # Read Dataset
    df = pd.read_csv('apps/home/menantea_tweets_temp.csv', sep=';')
    dataset = df.copy()

    #  Preprocessing data
    df = df.drop(columns=[kolom for kolom in df.columns if kolom != "full_text"], axis=1)
    df.reset_index(drop=True,inplace=True)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df["clean_text"] = df["full_text"].apply(clean)
    df.drop_duplicates(subset=['clean_text'], keep = 'first',inplace= True)
    data_prepocessed = df.copy()

    # Clustering
    custom_stop_words = [
        'menantea', 'adalah', 'dari', 'dalam', 'yang', 'oleh', 'pada', 'untuk',
        'dengan', 'sebagai', 'atau', 'saya', 'kami', 'anda', 'mereka', 'kita',
        'belum', 'telah', 'akan', 'jika', 'sudah', 'sekarang', 'bukan',
        'lagi', 'saat', 'karena', 'sehingga', 'itulah', 'bagi', 'maka', 'apakah',
        'tersebut', 'tentang', 'melakukan', 'sebelum', 'sesuatu', 'setelah', 'dapat',
        'harus', 'masih', 'mungkin', 'lebih', 'terlalu', 'kepada', 'lain', 'sama',
        'masa', 'doang', 'bisa', 'udah', 'juga', 'pernah', 'buat', 'tapi', 'jadi',
        'banget', 'kalau', 'punya', 'cuma', 'kalo', 'nanti', 'terus', 'malah', 'mana',
    ]

    tf_idf_vect = CountVectorizer(analyzer='word', ngram_range=(1,1), stop_words=custom_stop_words, min_df = 0.0001)
    tf_idf_vect.fit(df['clean_text'])
    desc_matrix = tf_idf_vect.transform(df["clean_text"])

    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(desc_matrix)
    clusters = km.labels_.tolist()

    tweets = {'Tweet' :df["full_text"].tolist(), 'Tweet (clean)': df["clean_text"].tolist(), 'cluster': clusters}
    clustering_result = pd.DataFrame(tweets, index = [clusters])

    # Word Counts
    cluster_word_counts = {}
    for cluster_id in range(num_clusters):
        cluster_df = clustering_result[clustering_result['cluster'] == cluster_id]
        cluster_text = ' '.join(cluster_df['Tweet (clean)'])
        words = cluster_text.split()
        filtered_words = [word for word in words if word not in custom_stop_words]

        word_counts = {}
        for word in filtered_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        cluster_word_counts[cluster_id] = word_counts

    word_count_result = []
    for cluster_id, word_counts in cluster_word_counts.items():
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = sorted_word_counts[:10]  # Ambil 10 kata teratas
        word_count_result.append({
            'cluster_id': cluster_id,
            'top_words': top_words
        })

    data = {
        'dataset': dataset.to_html(classes='table table-striped table-bordered', table_id='dataset'),
        'data_prepocessed': data_prepocessed.to_html(classes='table table-striped table-bordered', table_id='data_prepocessed'),
        'clustering_result': clustering_result.to_html(classes='table table-striped table-bordered', table_id='clustering_result'),
    }
    return render_template('home/index.html', data=data, segment='index')
