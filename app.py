
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

    return None

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
    pie_data = get_sentiment_counts()
    points = [[100, 100], [200, 200], [300, 300]]
    return render_template('index.html', pie_data=pie_data,
                           points=points,
                           topics=df['query'].unique().tolist(),
                           topic_data=get_sentiment_counts_per_topic(),
                           country_data=get_data_for_map()
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
                                   #nn_score=nn_score
                                   )
        return errormsg
    else:
        return render_template('results.html',
                               vader_result='NA',
                               logistic_result='NA',
                               nn_score='NA')


if __name__ == '__main__':
    app.run()
