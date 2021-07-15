import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

import psycopg2 as psycopg2
from sklearn import preprocessing

from db import *
from helpFunctions import *

# Import technical analysis packages
from ta.trend import CCIIndicator
from ta.trend import MACD
from ta.trend import EMAIndicator
from ta.trend import SMAIndicator
from ta.momentum import StochasticOscillator
from ta.momentum import WilliamsRIndicator
from ta.momentum import RSIIndicator
from ta.volume import on_balance_volume
from ta.others import DailyReturnIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

# import textanalysis packages
from string import punctuation
from nltk.corpus import stopwords
from nltk import tokenize, WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import regex

# Create list of punctuations and stopwords to remove
PUNCTUATION = [char for char in punctuation if char not in ["!", "%", "#"]]
STOPWORDS = stopwords.words("english")

"""
This file contains all function that will 
1. Retrieve data from the database set up in POSTGRESQL
2. Clean Data if necessary 
3. Calculate features e.g. technical indicators, sentiment, etc. 
4. Combine all data into one table and split this up into a train and test set 
"""

# Retrieve technical analysis data from database
def get_technical_table():
    query = "select * from technical"  # Create query
    column_names = ['ticker', 'date', 'open', 'close', 'high', 'low', 'adjclose', 'volume'] # Specify column names for dataframe that will be created
    prices = postgresql_to_dataframe(query, column_names) # Retrieve dataframe from db
    prices = prices.sort_values(by=['ticker', 'date']).reset_index(drop=True) # Sort observations according to date
    return prices


# Retrieve Yahoo finance news from database
def get_yf_news():
    # RETRIEVE DATA
    query = 'select * from news' # Create a selection query
    column_names = ['title', 'text', 'date'] # Specify the column names of the dataframe we want to create
    news_df = postgresql_to_dataframe(query, column_names) # Function retrieves the data from the database in the previously specified format

    # CLEAN DATAFRAME
    """
    Due to the use of an external dataset we have obtained a 
    far larger number of articles per day than using our own scraping method
    Therefore we decide to downsample the number of articles for each day to 100
    More information about the number of articles collected per day can be found in the ppNaturalLanguage.ipynb notebook 
    """
    news_df = news_df.sort_values(by='date').reset_index(drop=True) # Sort and clean dataframe
    news_df = news_df.drop_duplicates(subset='title') # Drop duplicates
    count = news_df.groupby(by='date', as_index=False).count() # Count the articles we have collected for each day
    news = pd.DataFrame(columns=['date', 'title', 'text']) # Create an empty dataframe
    trading_days = get_trading_days(date(2020, 3, 22)) # Get all the official trading days starting from 22/3 until now
    for val in trading_days: # Loop through each official trading day; we do not want weekends
        try:
            if int(count.loc[count['date'] == val, 'text']) > 100: # Check if we have more than a hundred articles for that day
                news_df[news_df['date'] == val].sample(100) # Take a random sample of a 100 articles of that day
                news = news.append(news_df[news_df['date'] == val].sample(100)) # Store the random selection of articles in our newly created dataset
            else:
                news = news.append(news_df[news_df['date'] == val]) # If we have less than a hundred we can add them all in our dataset
        except:
            pass

    news_df = news # Set our news_df equal to our news dataframe (Needed since previous step was not included in earlier development)

    # CLEAN TEXT
    news_df['title'] = news_df['title'].apply(lambda x: x[1:-1]) # Remove first and last character since this is always "[" and "]"
    news_df['text'] = news_df['text'].apply(lambda x: x[1:-1])
    news_df['title_cleaned_tokenized'] = news_df['title'].apply(lambda x: clean_news(x)) # Clean title and tokenize
    news_df['text_cleaned_tokenized'] = news_df['text'].apply(lambda x: clean_news(x)) # Clean article content and tokenize
    news_df['title_cleaned'] = news_df['title_cleaned_tokenized'].apply(lambda x: ' '.join([str(w) for w in x])) # Create column with full sentence

    # CALCULATE FEATURES
    news_df['polarity'] = news_df['title_cleaned'].apply(lambda x: get_polarity(x)) # Calculate polarity
    news_df['subjectivity'] = news_df['title_cleaned'].apply(lambda x: get_subjectivity(x)) # Calculate subjectivity
    news_df = topic_detection(news_df) # Calculate topic of article (Politics, health, ...)
    news_df = source_detection(news_df) # Calculate source of article (Reuters, Bloomberg, ...)
    news_df = stock_detection(news_df) # Calculate mention of a company (Apple, Google, ... in article content or title)
    news_df = extract_human_sentiment(news_df) # Calculate Human sentiment (Mention of buy, hold or sell)

    #CLEAN DATAFRAME
    news_df['date'] = pd.to_datetime(news_df.date).dt.date # Change type and remove time from datetime

    return news_df # Return statement


# Retrieve macro-economical data from database
def get_macroeconomic_table():
    # RETRIEVE DATA
    query = "select * from macroeconomic" # Create query
    column_names = ['date', 'usd_eur', 'usd_cny', 'usd_gbp', 'usd_cad', 'usd_mxn', 'usd_jpy', 'libor', 'gdp',
                    'crude_oil', 'brent_crude_oil', 'thirtheen_week_treasury_bill', 'treasury_yield_five',
                    "treasury_yield_ten", 'treasury_yield_thirty', 'usd_aud', 'dji', 'dax', 'nikkei', 'gold', 'bitcoin'] # Specify columns to be retrieved
    macro_table = postgresql_to_dataframe(query, column_names) # Retrieve data from database
    return macro_table


# define function to remove punctuation
def remove_punct(text):
    # remove punctuation
    text = "".join([char for char in text if char not in punctuation])
    return (text)


# define function to remove stopwords
def remove_stops(text_tokenized):
    # remove stopwords
    text_tokenized = [word for word in text_tokenized if word not in STOPWORDS]
    return (text_tokenized)


# define function to get subjectivity score of text document
def get_subjectivity(row):
    textBlob_review = TextBlob(row)
    return textBlob_review.sentiment[1]


# define function to get polarity score of text document
def get_polarity(row):
    textBlob_review = TextBlob(row)
    return textBlob_review.sentiment[0]


# Create features for the source of article if given
def source_detection(news_df):
    # Check both in title as content if source is mentioned
    news_df['BL'] = np.where(
        news_df.title_cleaned.str.contains('bloomberg') | news_df.text_cleaned_tokenized.str.contains('bloomberg'), 1, 0)
    news_df['RT'] = np.where(
        news_df.title_cleaned.str.contains('reuters') | news_df.text_cleaned_tokenized.str.contains('reuters'), 1, 0)
    news_df['YF'] = np.where(
        news_df.title_cleaned.str.contains('yahoo') | news_df.text_cleaned_tokenized.str.contains('yahoo finance'), 1,  0)
    news_df['CNBC'] = np.where(
        news_df.title_cleaned.str.contains('cnbc') | news_df.text_cleaned_tokenized.str.contains('cnbc'), 1,   0)
    return news_df


# Create feature for the topic of the article
def topic_detection(yfNews_df):
    # Check both in title as content if topic is mentioned
    yfNews_df['Business'] = np.where(
        yfNews_df.title_cleaned.str.contains('business') | yfNews_df.text_cleaned_tokenized.str.contains('business'), 1,
        0)
    yfNews_df['News'] = np.where(
        yfNews_df.title_cleaned.str.contains('news') | yfNews_df.text_cleaned_tokenized.str.contains('news'), 1, 0)
    yfNews_df['Politics'] = np.where(
        yfNews_df.title_cleaned.str.contains('politics') | yfNews_df.text_cleaned_tokenized.str.contains('politics'), 1,
        0)
    yfNews_df['Health'] = np.where(
        yfNews_df.title_cleaned.str.contains('health') | yfNews_df.text_cleaned_tokenized.str.contains('health'), 1, 0)
    yfNews_df['World'] = np.where(
        yfNews_df.title_cleaned.str.contains('world') | yfNews_df.text_cleaned_tokenized.str.contains('world'), 1, 0)

    return yfNews_df


# Indicate if stock mentioned in title or content of article
def stock_detection(news):
    for key, value in get_names().items():
        news[key] = np.where(news.title_cleaned.str.contains(value) | news.text_cleaned_tokenized.str.contains(value) |
                             news.title.str.contains(value) | news.text.str.contains(value),  1, 0)

    return news

# Clean title or content of article
def clean_news(news_text):
    lower_text = str.lower(news_text) # Lower text
    text_no_punc = remove_punct(lower_text) # Remove punctuation from text
    tokenized_text = tokenize.word_tokenize(text_no_punc)# Tokenize text
    words_without_stopwords = [word for word in tokenized_text if word not in STOPWORDS] # Remove stopwords
    lemmatizer = WordNetLemmatizer() # Initilize word lemmatizer
    text_lemma = [lemmatizer.lemmatize(word) for word in words_without_stopwords] # Lemmatize words

    return text_lemma


# Sentiment of future of stock given in tweet
def extract_human_sentiment(df):
    df['hold'] = np.where(df.title.str.contains('hold') | df.text.str.contains('hold'), 1, 0)
    df['buy'] = np.where(df.title.str.contains('buy') | df.text.str.contains('buy'), 1, 0)
    df['sell'] = np.where(df.title.str.contains('sell') | df.text.str.contains('sell'), 1, 0)

    return df


# Calculate technical indicators for each company
def calculate_technical_indicators(prices):

    # Sort values by date so that indicators are correctly calculated and reset index to avoid other problems
    prices = prices.sort_values(by='date') \
        .reset_index(drop=True)
    # Calculate indicators
    prices['CCI'] = CCIIndicator(prices.high, prices.low, prices.close).cci()
    prices['MACD'] = MACD(prices.close).macd()
    prices['EMA'] = EMAIndicator(prices.close, window=20).ema_indicator()
    prices['SMA20'] = SMAIndicator(prices.close, window=20).sma_indicator()
    prices['SMA10'] = SMAIndicator(prices.close, window=10).sma_indicator()
    prices['STOCHOSC'] = StochasticOscillator(prices.high, prices.low, prices.close).stoch()
    prices['RSI'] = RSIIndicator(prices.close).rsi()
    prices['WILLR'] = WilliamsRIndicator(prices.high, prices.low, prices.close).williams_r()
    prices['OBV'] = on_balance_volume(prices.close, prices.volume).values
    prices['RET'] = DailyReturnIndicator(prices.close).daily_return()
    prices['BBHB'] = BollingerBands(prices.close).bollinger_hband()
    prices['BBLB'] = BollingerBands(prices.close).bollinger_lband()
    prices['ATR'] = AverageTrueRange(prices.high, prices.low, prices.close).average_true_range()


    # Return the table with added technical indicators
    return prices


# Create basetable for the specified stock
def create_basetable(tck, start_independent=date(2020, 3, 22), end_independent=date(2020, 12, 31),
                     start_dependent=date(2021, 1, 1), end_dependent=date(2021, 3, 22), standardize=True):
    # Retrieve technical analysis data from database
    techn_df = get_technical_table()

    # Filter stock data
    techn_df = techn_df[techn_df.ticker == tck]

    # Calculate technical indicators
    techn_df = calculate_technical_indicators(techn_df)

    # Calculate label: Up or down: 1 or 0
    techn_df = techn_df.sort_values(by='date').reset_index(drop = True)
    techn_df['label'] = np.where(techn_df['close'].shift(-1) < techn_df['close'], 0, 1)
    # Delete ticker column
    techn_df = techn_df.drop(columns='ticker')

    # Retrieve news data from database
    news = get_yf_news()

    # Select columns
    news = news.drop(columns=['title', 'text', 'title_cleaned_tokenized', 'title_cleaned'])

    # Aggregate to values per day
    news_gr = news.groupby(by='date', as_index=False).agg(
        {'BL': 'sum', 'RT': 'sum', 'YF': 'sum', 'CNBC': 'sum', 'Business': 'sum', 'News': 'sum', 'Politics': 'sum',
         'Health': 'sum', 'World': 'sum', 'polarity': 'mean', 'subjectivity': 'mean', 'AAPL': 'sum', 'FB': 'sum',
         'GOOG': 'sum',
         'AMZN': 'sum', 'NFLX': 'sum', 'hold': 'sum', 'buy': 'sum', 'sell': 'sum'})

    # Retrieve macro-economic data from database
    macro = get_macroeconomic_table()

    # Delete unnecessary columns
    macro = macro.drop(columns=['gdp','libor'])

    # Change date type to datetime
    techn_df['date'] = pd.to_datetime(techn_df.date)
    macro['date'] = pd.to_datetime(macro.date)
    news_gr['date'] = pd.to_datetime(news_gr.date)

    # Combine all tables together
    basetable = pd.merge(left=techn_df, right=macro, how='left', on='date')
    basetable['date'] = pd.to_datetime(basetable['date'])
    basetable = pd.merge(left=basetable, right=news_gr, how='left', on='date')
    basetable['date'] = pd.to_datetime(basetable['date']).dt.date

    #ADD lagged variables for all technical features by including information of the past 5 days
    for c in techn_df.columns:
        if c != 'date':
            for i in range(1,5):
                name = c + str(i)
                basetable[name] = basetable[c].shift(i)

    # Create training set and testing set
    train_set = basetable[(basetable['date'] > start_independent) & (basetable['date'] < end_independent)]
    train_set.reset_index(drop=True, inplace=True)
    test_set = basetable[(basetable['date'] > start_dependent) & (basetable['date'] < end_dependent)]
    test_set.reset_index(drop=True, inplace=True)

    # Split dependent variable from independent
    train_X = train_set.drop(columns=['date', 'label'])
    train_y = train_set['label']
    test_X = test_set.drop(columns=['date', 'label'])
    test_y = test_set['label']

    # Impute missing values training set
    train_X = train_X.fillna(train_X.mean())
    # Impute missing values test set
    test_X = test_X.fillna(test_X.mean())

    # Standardize data otherwise use use min-max scaler
    if standardize:
        scaler = preprocessing.StandardScaler().fit(train_X)
        scaled_train_X = scaler.transform(train_X)
        scaled_test_X = scaler.transform(test_X)
    else:
        scaler = preprocessing.MinMaxScaler().fit(train_X)
        scaled_train_X = scaler.transform(train_X)
        scaled_test_X = scaler.transform(test_X)

    return scaled_train_X, scaled_test_X, train_y, test_y
