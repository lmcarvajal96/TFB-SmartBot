import random 

import joblib

import sys
import os


import random
random.seed(42)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve

import matplotlib.pyplot as plt

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras import layers, Sequential
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, concatenate, Dropout
from keras.models import Model

df = pd.read_excel(r"training_chatbot.xlsx")
df.head()

#Gretting classifier
def Greeting(x):
    if x =="Greeting":
        x = 1
    else:
        x = 0
    return x

def Search(x):
    if x =="Search":
        x = 1
    else:
        x = 0
    return x

def Suggestions(x):
    if x =="Suggestions":
        x = 1
    else:
        x = 0
    return x

df["Greeting"] = df["Intent type"].apply(lambda x: Greeting(x))
df["Search"] = df["Intent type"].apply(lambda x: Search(x))
df["Suggestions"] = df["Intent type"].apply(lambda x: Suggestions(x))



df_train, df_test = train_test_split(df, train_size=0.8, 
test_size=0.2, random_state=42, shuffle=True, stratify=df["Intent type"])


def processing(df, pretreatment = False, Tfidf = True, cv = None, stopwords = []):
  # Normalizamos y limpiamos el corpus 
  if pretreatment == True:
    df["Sentence"] = df['Sentence'].apply(lambda x: word_treatment(x))
    print("El corpus ha sido pretratado")

  # Transformamos nuestro corpus a un vector Tfidf
  if Tfidf == True:

    if cv == None:
      cv = TfidfVectorizer(
        stop_words= stopwords,
        ngram_range=(1, 4),
        strip_accents='ascii',
        max_df=0.99,
        min_df=0,
        max_features=100
      )
      cv.fit(df["Sentence"])
      X = cv.transform(df["Sentence"])
      print("Se ha realizado una vectorización Tfidf")
      return df, X, cv

    else:
      X = cv.transform(df["Sentence"])
      print("Se ha realizado una vectorización Tfidf basado en el corpus suministrado por cv")
    return df, X
  else:
    return df


df_train, X_train, cv = processing(df_train)
df_test, X_test = processing(df_test, cv = cv)

y_train = df_train[["Greeting","Search","Suggestions"]]
y_test = df_test[["Greeting","Search","Suggestions"]]

X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, train_size=0.7, 
test_size=0.3, random_state=42, shuffle=True, stratify=df_train["Intent type"])

shape= X_train.shape[1]

X_train.sort_indices()
X_validation.sort_indices()
#y_train.sort_indices()
#y_validation.sort_indices()

# Search Intent
def mlp_greeting(shape):
# define our MLP network
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42)
    model = Sequential()
    model.add(Dense(16, input_dim=shape, kernel_initializer = initializer, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="relu"))
# check to see if the regression node should be added
    #if regress:
    model.add(Dense(1, activation="sigmoid"))
    #Compile model
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss='binary_crossentropy', metrics ="accuracy", optimizer=opt)
# return our model
    return model


mlp_greeting = mlp_greeting(shape)
history = mlp_greeting.fit(X_train, np.asarray(y_train["Greeting"]).reshape(-1,1),
                  validation_data=(X_validation, np.asarray(y_validation["Greeting"]).reshape(-1,1)),
    epochs=200,
    workers = 2, use_multiprocessing= True, verbose = 2)

#Intent Search
def mlp_search(shape):
# define our MLP network
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42)
    model = Sequential()
    model.add(Dense(16, input_dim=shape, kernel_initializer = initializer, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="relu"))
# check to see if the regression node should be added
    #if regress:
    model.add(Dense(1, activation="sigmoid"))
    #Compile model
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss='binary_crossentropy', metrics ="accuracy", optimizer=opt)
# return our model
    return model


mlp_search = mlp_search(shape)
history = mlp_search.fit(X_train, np.asarray(y_train["Search"]).reshape(-1,1),
                  validation_data=(X_validation, np.asarray(y_validation["Search"]).reshape(-1,1)),
    epochs=250,
    workers = 2, use_multiprocessing= True, verbose = 2)

#Intent suggestion
def mlp_suggestion(shape):
# define our MLP network
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=42)
    model = Sequential()
    model.add(Dense(16, input_dim=shape, kernel_initializer = initializer, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="relu"))
# check to see if the regression node should be added
    #if regress:
    model.add(Dense(1, activation="sigmoid"))
    #Compile model
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(loss='binary_crossentropy', metrics ="accuracy", optimizer=opt)
# return our model
    return model


mlp_suggestion = mlp_suggestion(shape)
history = mlp_suggestion.fit(X_train, np.asarray(y_train["Search"]).reshape(-1,1),
                  validation_data=(X_validation, np.asarray(y_validation["Search"]).reshape(-1,1)),
    epochs=250,
    workers = 2, use_multiprocessing= True, verbose = 2)








