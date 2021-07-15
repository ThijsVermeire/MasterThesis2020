
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd
import seaborn as sns

import psycopg2 as psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from db import *

# Retrieves tickers from database, returns an array of strings
def get_tickers():
    tickers = []
    try:
        connection = get_connection()
        cursor = connection.cursor()
        select_query = """select ticker from companies"""
        cursor.execute(select_query,)
        records = cursor.fetchall()
        for row in records:
            tickers.append(row[0])
        close_connection(connection)
    except (Exception, psycopg2.Error) as error:
        print("Error while getting data", error)
    return tickers

# Retrieves tickers from database, returns an array of strings

def get_names():
    return {'AAPL': "apple","AMZN": 'amazon',"FB": 'facebook', "GOOG": 'google',
     "NFLX" : 'netflix'}

def postgresql_to_dataframe(select_query, column_names):
    """
    Tranform a SELECT query into a pandas dataframe
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(select_query)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        cursor.close()
        return 1

    # Naturally we get a list of tupples
    tupples = cursor.fetchall()
    cursor.close()

    # We just need to turn it into a pandas dataframe
    df = pd.DataFrame(tupples, columns=column_names)
    return df

# Create a list of all trading days in our time period
def get_trading_days(start_dt = date(2020,3,22), end_dt = date.today()):
    trading_days = []

    def daterange(date1, date2):
        for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

    weekdays = [6, 7]
    for dt in daterange(start_dt, end_dt):
        if dt.isoweekday() not in weekdays:
            trading_days.append(dt)

    # Remove the holidays that were held during the week and were markets were closed
    holidays = [date(2020,12,25),date(2020,11,26),date(2020,1,1),date(2020,1,18), date(2020,2,15)]
    days = [elem for elem in trading_days if elem not in holidays]

    return days

# Help functions that returns a table with the absolute and relative number of missing values for each column of a dataframe
def get_NAN_columns(df):
    columns_na = df.columns[df.isna().any()].tolist()
    sum = df.isnull().sum()
    percentage = (df.isnull().sum()/df.isnull().count())
    values = pd.DataFrame([sum,percentage])
    values = values.rename(index= {0: 'Absolute value', 1: 'Percentage'})
    return values[columns_na].round(2).T

# Create principal components in function of variance function for PCA
def plot_pca(pca, pca_train_X):
    fig, ax = plt.subplots()
    xi = np.arange(1, pca_train_X.shape[1] +1 , step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, pca_train_X.shape[1] +2 , step=2)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    # Create line at 95% variance as cut off point
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()

# Create Receiver operator curve
def plot_roc(test_y, pred):
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred)
    plt.plot(fpr, tpr)
    x = [0.0, 1.0]
    plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='random')
    plt.show()

# Function that returns measures for the classification model
def evaluate_model(pred, test_y):
    auc = metrics.roc_auc_score(test_y, pred)
    print("AUC: {}".format(auc))
    print(metrics.classification_report(test_y, pred))
    plot_roc(test_y, pred)
    plot_confusion_matrix(test_y, pred)


def buy_sell_observations(train_y, test_y):
    print('--------Training set--------')
    print('Total number of observations: ', len(train_y))
    print('Number of sell observations : ',(train_y == 0).sum())
    print('Number of buy observations : ',(train_y == 1).sum())
    print('\n--------Test set--------')
    print('Total number of observations: ', len(test_y))
    print('Number of sell observations : ',(test_y == 0).sum())
    print('Number of buy observations : ',(test_y == 1).sum())

def plot_confusion_matrix(test_y, pred):
    # Create confusion matrix
    ax = plt.subplot()
    cm = metrics.confusion_matrix(test_y, pred)
    sns.heatmap(cm, annot=True, ax=ax, cmap='tab20c')  # annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Sell', 'Buy'])
    ax.yaxis.set_ticklabels(['Sell', 'Buy'])
    plt.show()



def svmClassifier(pca_train_X, train_y, pca_test_X, test_y):
    svc_classifier = SVC(random_state=123, verbose=1)
    svc_classifier.fit(pca_train_X, train_y)
    pred = svc_classifier.predict(pca_test_X)
    evaluate_model(pred, test_y)

def lrClassifier(scaled_train_X, train_y, scaled_test_X, test_y):
    lr = LogisticRegression(verbose=1, random_state=123, solver='liblinear')
    lr.fit(scaled_train_X, train_y)
    pred = lr.predict(scaled_test_X)
    evaluate_model(pred, test_y)


def mlpClassifier(scaled_train_X, train_y, scaled_test_X, test_y):
    # Create the model
    clf = MLPClassifier(random_state=123, max_iter=300).fit(scaled_train_X, train_y)
    # Test the model after training
    pred = clf.predict(scaled_test_X)
    evaluate_model(pred, test_y)


