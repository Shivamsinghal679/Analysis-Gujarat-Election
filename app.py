import pandas as pd
import re
import string
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
import nltk
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask,render_template,redirect,jsonify,request
app = Flask(__name__)

#.................functions...................
stop_words = set(stopwords.words('english'))
# clean text functions
def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text
 
def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
 
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    filtered_sentence = [w for w in word_tokenize(text) if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)
 
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)
 
def clean(text):
    '''
    REMOVES ENTITIES,LINKS and makes text lowercase
    '''
    text = text.lower()
    text = strip_links(text)
    text = remove_punctuation(text)
    #text = remove_stopwords(text)
    text = strip_all_entities(text)
    return text

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def func(text,keywords):
    c=0
    for i in keywords:
        if(findWholeWord(i)(text) !=None):
            c = c+1
    return c

#........................... Sentimental Analysis .........................

def sentiment_vader(sentence):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']
 
    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"
 
    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"
 
    else :
        overall_sentiment = "Neutral"
 
    return overall_sentiment

#  party polarity keywords
keywordsBJP = ["bjp",'modi','bharatiya janata party','janata party','shah','amit']
keywordsAAP = ["aap",'kejriwal','aam aadmi party','aadmi party']
keywordsCONG = ['congress','inc','rahul','pappu','gandhi']

def get_party_polarity(text):
    bjp_score = func(text,keywordsBJP)
    aap_score = func(text,keywordsAAP)
    cong_score = func(text,keywordsCONG)
    if((bjp_score==0) and (aap_score==0) and (cong_score==0)):
        return 'Has no bias for a political party'
    if((bjp_score!=0) and (aap_score==0) and (cong_score==0)):
        return 'BJP'
    if((bjp_score==0) and (aap_score!=0) and (cong_score==0)):
        return 'AAP'
    if((bjp_score==0) and (aap_score==0) and (cong_score!=0)):
        return 'Congress'
    return "Mixed Tweets are currently not supported"


def model(text):
    text = GoogleTranslator(source='auto', target='en').translate(text=text)
    text = clean(text)
    sentiment = sentiment_vader(text)
    political_party = get_party_polarity(text)
 
    if(len(political_party) == 40):
        return political_party
    obj = political_party + ' and sentiment as ' + sentiment
    return obj

#...................Charts........................


new_df = pd.read_csv('final_df.csv')
 
# Party-wise Dataframe
bjp = new_df[new_df['About_Party'] == 'BJP']
aap = new_df[new_df['About_Party'] == 'AAP']
cong = new_df[new_df['About_Party'] == 'CONG']
 
 
#Getting scores
bjp_p = bjp[bjp['Sentiment'] == 'Positive'].shape[0]
bjp_n = bjp[bjp['Sentiment'] == 'Negative'].shape[0]
bjp_a = bjp[bjp['Sentiment'] == 'Neutral'].shape[0]
aap_p = aap[aap['Sentiment'] == 'Positive'].shape[0]
aap_n = aap[aap['Sentiment'] == 'Negative'].shape[0]
aap_a = aap[aap['Sentiment'] == 'Neutral'].shape[0]
cong_p = cong[cong['Sentiment'] == 'Positive'].shape[0]
cong_n = cong[cong['Sentiment'] == 'Negative'].shape[0]
cong_a = cong[cong['Sentiment'] == 'Neutral'].shape[0]
 
#sentiments for piechart
bjp_avg_sentiment = (bjp_p - bjp_n)/(bjp_p + bjp_n + bjp_a)
aap_avg_sentiment = (aap_p - aap_n)/(aap_p + aap_n + aap_a)
cong_avg_sentiment = (cong_p - cong_n)/(cong_p + cong_n + cong_a)


def time_analysis(df):
    df['Date'] = pd.to_datetime(df['Date'])
    date_df = df[df['Date'] > '2022-09']
    weekly_count = date_df.groupby(pd.Grouper(key='Date', freq='W-MON'))['Clean-Text'].count().reset_index().sort_values('Date')
    count = weekly_count['Clean-Text'].to_list()
    date = weekly_count['Date'].dt.strftime('%Y-%m-%d').to_list()
    return count,date

# bjp variables
count,date = time_analysis(bjp[bjp['Sentiment'] == 'Positive'])
coun,dat=time_analysis(bjp[bjp['Sentiment'] == 'Negative'])

# aap variable
cou,da = time_analysis(aap[aap['Sentiment'] == 'Positive'])
co,d=time_analysis(aap[aap['Sentiment'] == 'Negative'])

# congress variables
con_cou,con_da = time_analysis(cong[cong['Sentiment'] == 'Positive'])
con_co,con_d=time_analysis(cong[cong['Sentiment'] == 'Negative'])

# ........................dataset_object...........

Odata = {
    'bjp_p':count,
    'date':date,
    'aap_p':cou,
    'cong_p':con_cou,
    'bjp_n':coun,
    'aap_n':co,
    'cong_n':con_co,
    'bjp_pie':bjp_avg_sentiment,
    'cong_pie':cong_avg_sentiment,
    'aap_pie':aap_avg_sentiment

}

# ..................api..................
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/register')
def about():
    return render_template('register.html')

@app.route('/parties')
def part():
    return render_template('parties.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/model')
def test():
    return render_template('testmodel.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html',Odata=Odata)

@app.route('/get_prediction',methods=['POST'])
def get_prediction():
    q = request.json
    text = q['text']
    print(text)
    obj = model(text)
    
    return jsonify(obj)


app.run(debug=True)