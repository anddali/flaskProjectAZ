
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import preprocess_tweets
import json


app = Flask(__name__)



log_model=None
tfidf_vectorizer=None
tokenizer=None
errormsg=''
nnmodel = None

df = pd.read_csv('final_data.csv')

try:
    log_model = pickle.load(open('logistic_model.sav', 'rb'))
    errormsg += ' logistic_model ok;\n'
except Exception as e:
    errormsg += str(e)
try:
    tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
    errormsg += ' vect ok;\n'
except Exception as e:
    errormsg += str(e)
try:
    tokenizer = pickle.load(open('tokenizer.sav', 'rb'))
    errormsg += ' token ok;\n'
except Exception as e:
    errormsg += str(e)

    #nnmodel = keras.models.load_model('saved_models')
    #errormsg += ' token ok;'



def logistic_score(sentence):
    try:
        sentiment_to_text = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        tf_idf = tfidf_vectorizer.transform([sentence])
        result = log_model.predict(pd.DataFrame(tf_idf.toarray()))
        return sentiment_to_text[result[0]]
    except Exception as e:
        return 'NA'


def vader_sentiment_score(sentence):
    try:
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores([sentence])
        if sentiment_dict['compound'] >= 0.05:
            return 'Positive'
        elif sentiment_dict['compound'] <= - 0.05:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception as e:
        return 'NA'


def neural_network_score(sentence):

    # try:
    #    labels = ['Negative', 'Neutral', 'Positive']
    #    s = tokenizer.texts_to_sequences([sentence])
    #    s = sequence.pad_sequences(s, maxlen=100)
    #    pred = nnmodel.predict(s)
    #    print(pred)
    #    return labels[np.argmax(pred)]
    # except Exception as e:
    #    print(e)
    return 'NA'


def get_sentiment_counts():
    result = df['class'].value_counts().to_dict()
    result['Negative'] = result.pop(-1)
    result['Neutral'] = result.pop(0)
    result['Positive'] = result.pop(1)
    print(result)
    print(type(result))
    return result


def get_sentiment_counts_per_topic():
    topics = df['query'].unique().tolist()
    negative = []
    neutral = []
    positive = []

    for topic in topics:
        negative.append(len(df[(df['query'] == topic) & (df['class'] == -1)]))
        neutral.append(len(df[(df['query'] == topic) & (df['class'] == 0)]))
        positive.append(len(df[(df['query'] == topic) & (df['class'] == 1)]))

    return {'negative': negative, 'neutral': neutral, 'positive': positive}

def transform_sun_data(sun):
    indexes=[-1,0,1]
    res = []
    for i in indexes:
        try:
            x = int(sun[i])
        except Exception as e:
            x = 0
        res.append(x)
    return res


def get_data_for_sunburst():
    country_continent = {'England': 'Europe', 'Nigeria': 'Africa', 'Usa': 'North America', 'Uganda': 'Africa', 'Cameroon': 'Africa', 'Germany': 'Europe',
                     'Sweden': 'Europe', 'Ghana': 'Africa', 'Denmark': 'Europe', 'Kenya': 'Africa', 'South Africa': 'Africa',
                     'Ireland': 'Europe',
                     'Canada': 'North America', 'Belgium': 'Europe', 'India': 'Asia', 'Australia': 'Australia', 'UAE': 'Asia', 'Japan': 'Asia',
                     'Bangladesh': 'Asia', 'Austria': 'Europe', 'Thailand': 'Asia', 'Switzerland': 'Europe', 'China': 'Asia',
                     'Poland': 'Europe', 'Pakistan': 'Asia', 'France': 'Europe', 'Turkey': 'Europe', 'Brasil': 'South America', 'Mexico': 'South America',
                     'Malaysia': 'Asia', 'Iran': 'Asia', 'Netherlands': 'Europe', 'Spain': 'Europe', 'Singapore': 'Asia',
                     'Ukraine': 'Europe',
                     'Russia': 'Europe', 'Greece': 'Europe', 'Somalia': 'Africa', 'Luxembourg': 'Europe',
                     'Iceland': 'Europe', 'Israel': 'Africa', 'Venezuela': 'South America', 'Italy': 'Europe', 'Lebanon': 'Africa',
                     'Portugal': 'Europe', 'Puerto Rico': 'South America', 'New Zealand': 'Asia', 'Indonesia': 'Asia', 'Lithuania': 'Europe',
                     'Serbia': 'Europe', 'Malta': 'Europe', 'Belarus': 'Europe', 'Finland': 'Europe', 'Zambia': 'Africa',
                     'Uzbekistan': 'Asia',
                     'Latvia': 'Europe'}
    df['Continent'] = df.apply(lambda row: country_continent[row.user_location], axis=1)
    europe = transform_sun_data(df.loc[df['Continent'] == 'Europe', ['class']].value_counts())
    asia = transform_sun_data(df.loc[df['Continent'] == 'Asia', ['class']].value_counts())
    australia = transform_sun_data(df.loc[df['Continent'] == 'Australia', ['class']].value_counts())
    africa = transform_sun_data(df.loc[df['Continent'] == 'Africa', ['class']].value_counts())
    north_america = transform_sun_data(df.loc[df['Continent'] == 'North America', ['class']].value_counts())
    south_america = transform_sun_data(df.loc[df['Continent'] == 'South America', ['class']].value_counts())
    sun = [{
    'id': '0.0',
    'parent': '',
    'name': 'Tweets'
    },{
    'id': '1.1',
    'parent': '0.0',
    'name': 'Europe'
    },{
    'id': '1.2',
    'parent': '0.0',
    'name': 'Asia'
    },{
    'id': '1.3',
    'parent': '0.0',
    'name': 'Africa'
    },{
    'id': '1.4',
    'parent': '0.0',
    'name': 'Australia'
    },{
    'id': '1.5',
    'parent': '0.0',
    'name': 'North America'
    },{
    'id': '1.6',
    'parent': '0.0',
    'name': 'South America'
    },{
    'id': '2.1',
    'parent': '1.1',
    'name': 'Negative',
    'value': europe[0]
    },{
    'id': '2.2',
    'parent': '1.1',
    'name': 'Neutral',
    'value': europe[1]
    },{
    'id': '2.3',
    'parent': '1.1',
    'name': 'Positive',
    'value': europe[2]
    },{
    'id': '2.1',
    'parent': '1.2',
    'name': 'Negative',
    'value': int(asia[0])
    },{
    'id': '2.2',
    'parent': '1.2',
    'name': 'Neutral',
    'value': int(asia[1])
    },{
    'id': '2.3',
    'parent': '1.2',
    'name': 'Positive',
    'value': int(asia[2])
    },{
    'id': '2.1',
    'parent': '1.3',
    'name': 'Negative',
    'value': int(africa[0])
    },{
    'id': '2.2',
    'parent': '1.3',
    'name': 'Neutral',
    'value': int(africa[1])
    },{
    'id': '2.3',
    'parent': '1.3',
    'name': 'Positive',
    'value': int(africa[2])
    },{
    'id': '2.1',
    'parent': '1.4',
    'name': 'Negative',
    'value': int(australia[0])
    },{
    'id': '2.2',
    'parent': '1.4',
    'name': 'Neutral',
    'value': int(australia[1])
    },{
    'id': '2.3',
    'parent': '1.4',
    'name': 'Positive',
    'value': int(australia[2])
    },{
    'id': '2.1',
    'parent': '1.5',
    'name': 'Negative',
    'value': int(north_america[0])
    },{
    'id': '2.2',
    'parent': '1.5',
    'name': 'Neutral',
    'value': int(north_america[1])
    },{
    'id': '2.3',
    'parent': '1.5',
    'name': 'Positive',
    'value': int(north_america[2])
    },{
    'id': '2.1',
    'parent': '1.6',
    'name': 'Negative',
    'value': int(south_america[0])
    },{
    'id': '2.2',
    'parent': '1.6',
    'name': 'Neutral',
    'value': int(south_america[1])
    },{
    'id': '2.3',
    'parent': '1.6',
    'name': 'Positive',
    'value': int(south_america[2])
    }

    ]

    return sun


def get_data_for_map():
    country_codes = {'England': 'GB', 'Nigeria': 'NG', 'Usa': 'US', 'Uganda': 'UG', 'Cameroon': 'CM', 'Germany': 'DE',
                     'Sweden': 'SE', 'Ghana': 'GH', 'Denmark': 'DK', 'Kenya': 'KE', 'South Africa': 'ZA',
                     'Ireland': 'IE',
                     'Canada': 'CA', 'Belgium': 'BE', 'India': 'IN', 'Australia': 'AU', 'UAE': 'AE', 'Japan': 'JP',
                     'Bangladesh': 'BG', 'Austria': 'AT', 'Thailand': 'TH', 'Switzerland': 'CH', 'China': 'CN',
                     'Poland': 'PL', 'Pakistan': 'PK', 'France': 'FR', 'Turkey': 'TR', 'Brasil': 'BR', 'Mexico': 'MX',
                     'Malaysia': 'MY', 'Iran': 'IR', 'Netherlands': 'NL', 'Spain': 'ES', 'Singapore': 'SG',
                     'Ukraine': 'UA',
                     'Russia': 'RU', 'Greece': 'GR', 'Somalia': 'SO', 'Luxembourg': 'LU',
                     'Iceland': 'IS', 'Israel': 'IL', 'Venezuela': 'VE', 'Italy': 'IT', 'Lebanon': 'LB',
                     'Portugal': 'PT', 'Puerto Rico': 'PR', 'New Zealand': 'NZ', 'Indonesia': 'ID', 'Lithuania': 'LT',
                     'Serbia': 'RS', 'Malta': 'MT', 'Belarus': 'BY', 'Finland': 'FI', 'Zambia': 'ZM',
                     'Uzbekistan': 'UZ',
                     'Latvia': 'LV'}
    countries = df['user_location'].unique()
    result = []
    for country in countries:
        result.append(
            {'country': country, 'code': country_codes[country], 'z': len(df[df['user_location'] == country])})
    return result

def get_words(text):
    word_counts = {}
    text = text.split(' ')
    for word in text:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

    result = []

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    max_words = 40

    for word in sorted_words:
        result.append({'name':word[0], 'weight':word[1]})
        max_words = max_words - 1
        if max_words == 0:
            break
    return result

def get_wordcloud():
    negative = df.loc[df['class']==-1]
    negative = negative['cleaned']
    negative = ' '.join(negative)

    positive = df.loc[df['class']==1]
    positive = positive['cleaned']
    positive = ' '.join(positive)

    neutral = df.loc[df['class']==0]
    neutral = neutral['cleaned']
    neutral = ' '.join(neutral)

    return get_words(negative), get_words(neutral), get_words(positive)

@app.route('/')
def visualisations():
    sun = get_data_for_sunburst()
    pie_data = get_sentiment_counts()
    points = [[100, 100], [200, 200], [300, 300]]
    return render_template('index.html', pie_data=pie_data,
                           points=points,
                           topics=df['query'].unique().tolist(),
                           topic_data=get_sentiment_counts_per_topic(),
                           country_data=get_data_for_map(),
                           sun=json.dumps(sun)
                           )
# @app.route('/')
# def visualisations():
#     # pie_data = get_sentiment_counts()
#     points = [[100, 100], [200, 200], [300, 300]]
#     return 'Hi! '

#
@app.route('/words', methods=['GET', 'POST'])
def wordcloud():
    negative_wordcloud, neutral_wordcloud, positive_wordcloud = get_wordcloud()
    print(wordcloud)
    return render_template('words.html', wordcloud_neg=json.dumps(negative_wordcloud),
                           wordcloud_neu=json.dumps(neutral_wordcloud),
                           wordcloud_pos=json.dumps(positive_wordcloud))

#
@app.route('/predictions/', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        try:
            query = request.form['query']
            cleaned_query = preprocess_tweets.process_tweet(query)
            vader_result = vader_sentiment_score(cleaned_query)
            logistic_result = logistic_score(cleaned_query)
            #nn_score = neural_network_score(cleaned_query)
            return render_template('results.html', q=query, clean_text='Cleaned text: ' + cleaned_query,
                                   vader_result=vader_result,
                                   logistic_result=logistic_result,
                                   nn_score='Not implemented'
                                   )
        except Exception as ee:
            return errormsg + str(ee)
    else:
        return render_template('results.html',
                               vader_result='NA',
                               logistic_result='NA',
                               nn_score='NA')


if __name__ == '__main__':
    app.run()
