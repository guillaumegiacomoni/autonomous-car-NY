#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:26:33 2019

@author: guillaumegiacomoni
"""

import pandas as pd
import numpy as np


def select_data(file, n_trips, L_dim, l_dim, method_selection =  None) :
    """
    Select all taxis from file that are in relative square L_dim x l_dim 
    
    args :
        file(Pandas DF): data of the taxi trips in NY
        n_trips(int): the number of trips you want to select
        L_dim(float), l_dim(float): dimension relative de la grid par rapport à la taille de New York
        method_selection(str): How to select the data specificelly
    
    Returns :
        Pandas Df
    """
    
    my_columns = ['pickup_datetime','dropoff_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude']
    data_taxi1 = file.copy()
    data_taxi2 = data_taxi1.loc[:, my_columns]
    
    data_taxi = data_taxi2.drop((data_taxi2[(data_taxi2.pickup_longitude < -75) & 
                              (data_taxi2.pickup_longitude > -73) &
                              (data_taxi2.dropoff_longitude < -75) &
                              (data_taxi2.dropoff_longitude > -73) &
                              (data_taxi2.pickup_latitude < 38.) &
                              (data_taxi2.pickup_latitude > 42.) & 
                              (data_taxi2.dropoff_latitude < 38.) &
                              (data_taxi2.dropoff_latitude > 42.)]).index)
                                                

    
    min_longitude = min(min(list(data_taxi.loc[:,'pickup_longitude'])),
                    min(list(data_taxi.loc[:,'dropoff_longitude'])))
    max_longitude = max(max(list(data_taxi.loc[:,'pickup_longitude'])),
                    max(list(data_taxi.loc[:,'dropoff_longitude'])))
    min_latitude = min(min(list(data_taxi.loc[:,'pickup_latitude'])),
                    min(list(data_taxi.loc[:,'dropoff_latitude'])))
    max_latitude = max(max(list(data_taxi.loc[:,'pickup_latitude'])),
                       max(list(data_taxi.loc[:,'dropoff_latitude'])))

    e_longitude, e_latitude = l_dim * (max_longitude - min_longitude), L_dim * (max_latitude - min_latitude)
    condition = False
    iteration = 0
    
    if method_selection == "spread" :
        random_latitude, random_longitude = (np.random.uniform(min_latitude, max_latitude - e_latitude),
                                             np.random.uniform(min_longitude, max_longitude - e_longitude))
        
        if n_trips == 0:
            return(pd.DataFrame(my_columns).iloc[:])
        
        c = np.sqrt(e_latitude * e_longitude / n_trips)
        list_special_taxis = []
        for k_1 in range(0, int(e_longitude / c)) :
            for k_2 in range(0, int(e_latitude / c)) :
                
                df = data_taxi.loc[(data_taxi.pickup_longitude > random_longitude + k_1 * c) & 
                                    (data_taxi.pickup_longitude < random_longitude + min((k_1 + 1) * c, e_longitude)) & 
                                    (data_taxi.pickup_latitude > random_latitude + k_2 * c) &
                                    (data_taxi.pickup_latitude < random_latitude + min((k_2  + 1)* c, e_latitude)) & 
                                    (data_taxi.dropoff_longitude > random_longitude) & 
                                    (data_taxi.dropoff_longitude < random_longitude + e_longitude) & 
                                    (data_taxi.dropoff_latitude > random_latitude) & 
                                    (data_taxi.dropoff_latitude < random_latitude + e_latitude)]
                df.head()
                if df[my_columns[0]].count() > 0 :
                    print(df.iloc[0])
                    list_special_taxis.append(df.iloc[[0]])
        if list_special_taxis != [] :
            return((pd.concat([list_special_taxis[i] for i in range(0, len(list_special_taxis))])).iloc[:])
        
        
    while not(condition) and iteration < 10  :
        random_latitude, random_longitude = (np.random.uniform(min_latitude, max_latitude - e_latitude),
                                             np.random.uniform(min_longitude, max_longitude - e_longitude))
        select_taxi = data_taxi.loc[(data_taxi.pickup_longitude > random_longitude) & 
                                    (data_taxi.pickup_longitude < random_longitude + e_longitude) & 
                                    (data_taxi.pickup_latitude > random_latitude) &
                                    (data_taxi.pickup_latitude < random_latitude + e_latitude) & 
                                    (data_taxi.dropoff_longitude > random_longitude) & 
                                    (data_taxi.dropoff_longitude < random_longitude + e_longitude) & 
                                    (data_taxi.dropoff_latitude > random_latitude) & 
                                    (data_taxi.dropoff_latitude < random_latitude + e_latitude)]
        if n_trips < len(select_taxi) :
            condition = True
        iteration += 1
    return select_taxi.iloc[:min(n_trips, len(select_taxi))] 


class taxi(object):
    """
    A class that contains basic information about the taxis
    
    Attributes :
        departure(tuple of int): departure position in the grid
        arrival(tuple of int) : arrival position in the grid
        position(tuple of int): current position of the taxi in the grid
        trip(list of tuple): positions of the taxi at each instant
    """
    
    def __init__(self,departure,arrival,position):
        self.departure = departure 
        self.arrival = arrival
        self.position = position
        self.trip = None
    

        

def convert_data (data_taxi,density):
    """
    Convert the raw coordinates of the taxi from data_taxi to int coordinates in a grid 
    that respect the density parameter
    
    Args :
        data_taxi(pandas DF): the original coordinates from the NYC taxi
        density(float): the number of taxi in each cell of the grid
        
    Returns :
        taxis(list of taxi)
        long,large(int) the length of the grid
    """
    
    n_trips = len(data_taxi)
    
    min_longitude = min(min(list(data_taxi.loc[:,'pickup_longitude'])),
                    min(list(data_taxi.loc[:,'dropoff_longitude'])))
    max_longitude = max(max(list(data_taxi.loc[:,'pickup_longitude'])),
                    max(list(data_taxi.loc[:,'dropoff_longitude'])))
    min_latitude = min(min(list(data_taxi.loc[:,'pickup_latitude'])),
                    min(list(data_taxi.loc[:,'dropoff_latitude'])))
    max_latitude = max(max(list(data_taxi.loc[:,'pickup_latitude'])),
                       max(list(data_taxi.loc[:,'dropoff_latitude'])))
    
    e_longitude = max_longitude - min_longitude
    
    e_latitude = max_latitude - min_latitude
    
    scale =np.sqrt( n_trips/( e_longitude* e_latitude * density) )

    taxis = []
    
    for i in range(n_trips):
        selected_taxi =  data_taxi.iloc[i]
        departure = [int((selected_taxi.pickup_longitude - min_longitude) * scale),
                     int((selected_taxi.pickup_latitude - min_latitude) * scale),
                     ]
        
        arrival = [
                     int((selected_taxi.dropoff_longitude - min_longitude) * scale),
                     int((selected_taxi.dropoff_latitude - min_latitude) * scale)]
        
        taxis.append(taxi(departure,arrival,departure))
    return taxis,int(scale*(e_latitude))+1,int(scale*(e_longitude))+1


    

    
    
    
    
    
    