#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:37:02 2023

@author: yashchonkar
"""


import time
import json
import pickle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
import requests
import re
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Normalization, Dropout


'''
Functions used to build dataset and train model
'''


pd.set_option('display.max_colwidth', 25)

def get_game_stat_arrays(box_score_url): #returns array of each players advanced stats given a link to the box score
    #create dataframe for each team from url
        req = Request(box_score_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
      
        home_stats = pd.read_html(str(webpage))[15]
        away_stats = pd.read_html(str(webpage))[7]
        
        tensor_list = [] # in order of minutes played, as ordered on basketball refrence. Home team followed by away team
        #add array for each starter to list
        for i in range(5):
            tensor = list(home_stats.iloc[i].values[2:].astype(float))
            tensor_list.append(tensor)
        for i in range(5):
            tensor = list(away_stats.iloc[i].values[2:].astype(float))
            tensor_list.append(tensor)
            
            
        return tensor_list




def load_nba_game_data(year):#load data used to train game mode into .pkl 
    delay = 3.1 #used between requests since bball refrence has 20 req/min rate limit
    
    player_tensors = [] #features that will be used as X of dataset
    matchup_outcomes = [] #probability [0,1] that home team win, used as Y of dataset
    months = ['november', 'december', 'january' , 'february','march']
    if year == 2020 or year == 2019: #bubble year(covid)
        return
    for month in months:
        if year == 2012 and month == 'november': #NBA lockout!
            continue
        #make webpage into soup object
        try:
            url = 'https://www.basketball-reference.com/leagues/NBA_'+str(year)+'_games-'+str(month)+'.html' 
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'}) 
            time.sleep(delay)
            webpage = urlopen(req).read()
            #dataframe storing results of each game in month
            matchups = pd.read_html(str(webpage))[0]
            matchups_length =  matchups.shape[0]
            
            
            # Convert the 'bytes' object to a string
            webpage_string = webpage.decode('utf-8')
    
            # Create a Beautiful Soup object from the string
            webpage = soup(webpage_string, 'html.parser')
            
            #get link to every box score
            target_word = 'Box Score'
            tags = webpage.find_all('a', text=re.compile(target_word, re.IGNORECASE))
            box_score_links = ['https://www.basketball-reference.com' + tag['href'] for tag in tags]
            
            #adds player advanced stats array to player_tensors
            for box_score in box_score_links:
                player_tensors += get_game_stat_arrays(box_score)
                time.sleep(delay)
            
                
                
            #update matchups_outcomes with 1 or 0 for each matchup in the month
            for i in range(matchups_length):
                result = 1 if matchups.loc[i][3] < matchups.loc[i][5] else 0
                matchup_outcomes.append(result)
        except Exception as e:
            print(str(year) + " " + month+  "Error")
            continue
        
            
    #to store data to load into dataframe and use
    player_pkl = []
    matchup_pkl = []
    with open('player_tensors.pkl', 'rb') as file:
       player_pkl = pickle.load(file)
       
       matchup_pkl = []
       with open('matchup_outcomes.pkl', 'rb') as file:
           matchup_pkl = pickle.load(file)
       
    player_pkl += player_tensors
    matchup_pkl += matchup_outcomes
    
    with open('player_tensors.pkl', 'wb') as file:
        pickle.dump(player_pkl, file)
  
    with open('matchup_outcomes.pkl', 'wb') as file:
        pickle.dump(matchup_pkl, file)
        
        
        
    #data frame creation
    player_tensors = []
    matchup_outcomes = []
    with open('player_tensors.pkl', 'rb') as file:
       player_tensors = pickle.load(file)
       
    with open('matchups.pkl', 'rb') as file:
        matchup_outcomes = pickle.load(file)
          
    columns = ['Player1', 'Player2', 'Player3', 'Player4', 'Player5', 'Player6', 'Player7', 'Player8', 'Player9', 'Player10', 'Matchup']
    nba_dataset = pd.DataFrame(columns=columns)
    nba_dataset['Matchup'] = nba_dataset['Matchup'].astype(int) 
    
    for column in (columns - ['Matchup']):
        nba_dataset[column] = nba_dataset[column].values.astype(np.ndarray)
        
    for i, matchup in enumerate(matchup_outcomes):
        #add respective player tensors for game
        start_idx = i * 10
        end_idx = start_idx + 10  
        current_player_arrays = player_tensors[start_idx:end_idx]
        for p in range(1,11):
            nba_dataset.loc[i,'Player'+str(p)] = np.array(current_player_arrays[p-1])
        #add matchup value
        nba_dataset.loc[i,'Matchup'] = int(matchup)
        
    # Save dataset as single dataframe
    with open('nba_dataset.pkl', 'wb') as file:
        pickle.dump(nba_dataset, file)
        
        
        
    
    
def create_NBA_game_model():
    #model creation
    input_layer = Input(shape=(10,))  # stat arrays as input
    
    
    hidden_layer_1 = Dense(units=64, activation='relu')(input_layer)
    hidden_layer_2 = Dense(units=32, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(units=16, activation='relu')(hidden_layer_2)
    
    
    output_layer = Dense(units=1, activation='sigmoid')(hidden_layer_3)
    
  
    game_model = Model(inputs=input_layer, outputs=output_layer)
    
   
    game_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #game_model.summary()
    
    return game_model


def train_model(model):
    nba_dataset = pd.DataFrame()
    with open('nba_dataset.pkl','rb') as file:
        nba_dataset = file
    labels = nba_dataset['Matchup'].to_numpy()
    features = nba_dataset.iloc[:, 0:10].to_numpy()
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model.fit(features_train, labels_train, epochs=10, batch_size=32)
    
    return model
    

'''
program to run and build model
'''
model = create_NBA_game_model()
load_nba_game_data(2010)
nba_picker = train_model(model)





