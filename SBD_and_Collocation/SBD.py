#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:06:51 2018

@author: billzhan
"""
#==========  EECS 595 Assignment 1.1: Sentence Boundary Detection  ==========# 

import os
import re
import numpy as np
#--------------#
#--- Global ---#
#--------------#

WKDIR = "/home/billzhan/OneDrive/Academic/Umich/2018Fall/SI561/asgmt1"
TRAIN_FILE = "SBD.train"
TEST_FILE = "SBD.test"
VOWEL_REGEX = re.compile(r'[aeiou]') 
CONS_REGEX = re.compile(r'[b-df-hj-np-tv-z]')


#-----------------#
#--- Functions ---#
#-----------------#

#--- load files
def load_files():
    '''
    load train and test data
    
    @return: two numpy arrays (idx,word,label)
    '''
    train_path = os.path.join(WKDIR,TRAIN_FILE)
    test_path = os.path.join(WKDIR,TEST_FILE)
    train_data, test_data = [], []
    # read train
    with open(train_path) as train_f:
        for line in train_f.readlines():
            line = line.strip()
            line_list = re.split(' ',line)
            train_data.append(line_list)
    # read test
    with open(test_path) as test_f:
        for line in test_f.readlines():
            line = line.strip()
            line_list = re.split('\s+',line)
            test_data.append(line_list)
    return np.array(train_data), np.array(test_data)

#--- word2idx


#--- extract five core features
def five_features(data, idx):
    '''
    extract five core features of each period:
        1. Word to the left of “.” (L) 
        2. Word to the right of “.” (R) 
        3. Length of L < 3
        4. Is L capitalized
        5. Is R capitalized
    
    @param data: NumpyArray, the dataset that contains this token
    @param idx: Str, the token to evaluate
    
    @return: List, five features: [word_to_left,word_to_right,len_L3,L_cap,R_cap]
    '''
    row_idx = np.where(data[:,0]==idx)[0][0]
    token_vector = data[row_idx,:][0]
    token = token_vector[1]
    assert token_vector[-1]!= 'TOK'
    # word to the left
    word_to_left = re.match(r'[^.]+',token).group()
    # word_to_right
    word_to_right = re.match(r'[^.]+',data[row_idx+1,1]).group()
    # Length L < 3
    len_L3 = 'Yes' if len(word_to_left)<3 else 'No'
    # is L capitalized
    L_cap = 'Yes' if word_to_left[0].isupper() else 'No'
    # is R capitalized
    R_cap = 'Yes' if word_to_right[0].isupper() else 'No'
    
    return [word_to_left,word_to_right,len_L3,L_cap,R_cap]

def three_my_features(data, idx):
    '''
    extract three my selected features:
        1. Length of R: numeric
        2. Does L contains both a vowel and a consonants: boolean
        3. Does R contains both a vowel and a consonants: boolean
    
    @param data: NumpyArray, the dataset that contains this token
    @param idx: Str, the token to evaluate
    
    @return: List, three features: [len_R,L_vow_cons,R_vow_cons]
    '''
    row_idx = np.where(data[:,0]==idx)[0][0]
    token_vector = data[row_idx,:][0]
    token = token_vector[1]
    assert token_vector[-1]!= 'TOK'
    # lenght of R
    word_to_right = re.match(r'[^.]+',data[row_idx+1,1]).group()
    len_R = len(word_to_right)
    # does L contains vowel and consonant
    word_to_left = re.match(r'[^.]+',token).group().lower()
    if re.match(VOWEL_REGEX,word_to_left) and re.match(CONS_REGEX,word_to_left):
        L_vow_cons = 'Yes'
    else:
        L_vow_cons = 'No'
    # does R
    if re.match(VOWEL_REGEX,word_to_right) and re.match(CONS_REGEX,word_to_right):
        R_vow_cons = 'Yes'
    else:
        R_vow_cons = 'No'
        
    return [len_R,L_vow_cons,R_vow_cons]











