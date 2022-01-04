# -*- coding: utf-8 -*-
"""
Program Name: What's Your Point?
Author: Rupak Kumar Das
Course: CS 5242
Date: 11/04/21

Program Details:
    This program creates a a word by word co-occurrence matrix for a corpus.The cell values in this matrix 
    should be the Pointwise Mutual Information (PMI) between 2 words.
    
Code Usage:
    The user must provide below input in below format:
        similiar-pmi.py  6  ./PA4-News-2011  input.txt
        where,
        similiar-pmi.py = Program file name
        6 = window size
        ./PA4-News-2011 = location of training corpus
        input.txt = contains pair words
        
    the output format will be:
        cosine(word1,word2)	word1	word2	count(word1)  count(word2) count(word1,word2) PMI(word1,word2)
        
Program Algorithm:
    1) Read data file
    1.1: loop through each file and read each line
    1.2: remove all non-alphanumeric characters and convert all text to lower case.
    
    2) Create the word by word co-occurrence matrix
    2.1: create word pairs with a given window size and store them in a dictionary with frequency count.
    2.2: create a n*n dataframe where n is the number of unique words. This is a word by word co-occurrence matrix
        where the cell value is frequency.
    2.3: Create another dataframe where cell values are PMI values (using PMI equation)
    
    3) Get the result
    3.1) Read the input file line by line
    3.2) for each pair of words, find the PMI value from the dataframe. Convert to numpy array,
        find the dot product and finally the cosine value.
    3.3: Display the output
    
        
        
"""
# Import libraries

import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import math



#Data Processing

def dataProcess(directory,window):
    total_token = 0
    
    # Read all files
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='UTF-8') as f:
            for sentence in f:
                
                # lowercase the sentence
                sentence = sentence.lower()
                punc = '''!()-[]{};:'"1234567890\,<>./?@+=#$%^&*_~'''
                
                # Remove non-alphanumeric characters
                for ele in sentence:
                    if ele in punc:
                        sentence = sentence.replace(ele, "")
                        
                # Create Dictionary of co-occurance words    
                co_occurrence(sentence, window)
                
                # Count token nymber 
                for word in sentence.split():
                    total_token = total_token+1
                        
            
    print("There are {} number of tokens.".format(total_token))
    print("Number of unique token: ", len(vocab))


# Create Dictionary of co-occurance words
def co_occurrence(my_string, window):
        
    my_string = my_string.split()
    # For every sentence creating and adding word pairs with window size
    for i in range(len(my_string)):
        token = my_string[i]
        vocab.add(token)  # add to vocab
        next_token = my_string[i+1 : i+window]
        for t in next_token:
            key = tuple([token,t])
            word_dic[key] += 1
            
# Create co-occurance matrix (cell value frequency)           
def create_frame():
    
    # create dataframe of n*n 
    frame = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                          index=sorted(vocab),
                          columns=sorted(vocab))
    
    #Adding values to cell
    for key, value in word_dic.items():
        frame.at[key[0], key[1]] = value
        frame.at[key[1], key[0]] = value
    return frame

# Create co-occurance matrix (cell value PMI)            
def PMI (df):
    
    # Used formula  pmi = log2 P(x;y)/P(x)*P(y)
    
    # total count of each row and column
    col_totals = frame.sum(axis=0)
    row_totals = frame.sum(axis = 1)
    
    # total occurance
    total = col_totals.sum()
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    
    # To ignore divide error
    with np.errstate(divide='ignore'):       
        df = np.log2(df)
        df[np.isinf(df)] = 0.0 # replace inf by zero
    
    return df
    

def input_file(pmi_matrix):
    
    result = []
    with open(r'C:\Users\12182\Desktop\PA4\input.txt', encoding='utf-8') as file:
                
        for pair in file:             
            pair = pair.split()
            #print(pair)
            word1 = pair[0]
            word2 = pair[1]     
            
            # if both words contain in matrix
            if word1 in pmi_matrix and word2 in pmi_matrix:
                ppi = pmi_matrix[word1][word2] # get PMI value
                ppi = round(ppi,5)
                v = (pmi_matrix.loc[word1]) # get the vector of first word
                w = (pmi_matrix.loc[word2]) # get the vector of second word
                v = v.values # convert to numpy array
                w = w.values # convert to numpy array
                w = np.nan_to_num(w) # replace nan with zero
                v = np.nan_to_num(v) # replace nan with zero              

                dot_product = np.dot(v,w) # find the doc product
                v_length = math.sqrt(sum(i*i for i in v)) # find the length of first word
                w_length = math.sqrt(sum(i*i for i in w)) # find the length of second word
                cosine = dot_product/(v_length*w_length) # Find the cosine
                cosine = round(cosine,5)
                res = [cosine,word1,word2,frame[word1].sum(),frame[word2].sum(),frame[word1][word2].sum(),ppi]
                
            # if first word not in matrix and second word is in matrix   
            elif ((word1 not in pmi_matrix) and (word2 in pmi_matrix)):
                ppi = pmi_matrix[word1][word2]
                cosine = -9999
                res = [cosine,word1,word2,0,frame[word2].sum(),0,ppi]
            # if second word not in matrix and first word is in matrix     
            elif ((word2 not in pmi_matrix) and (word1 in pmi_matrix)):
                ppi = pmi_matrix[word1][word2]
                cosine = -9999
                res = [cosine,word1,word2,frame[word1].sum(),0,0,ppi]
            # if both words do not contain in matrix
            else:
                ppi = pmi_matrix[word1][word2]
                cosine = -9999
                res = [cosine,word1,word2,0,0,0,ppi]
                
            result.append(res)
        result = sorted(result, key=lambda x: x[0])
        
    return result           
                           

if __name__ == "__main__":
    
    word_dic = defaultdict(int)
    vocab = set() 

    # Parser to create command line arguments.
    parser = argparse.ArgumentParser(description='PMI co-occurance matrix') # This is the description
    parser.add_argument('Window', type = int,help='Provide the window') # Sentence selection
    parser.add_argument('Directory', type=str, help='Provide the directory') # Model selection
    parser.add_argument('Input', type = str,help='The name(s) of the file')  # Input File selection
    args = parser.parse_args()
    if args.Window and args.Directory and args.Input:
        print("This program genrates word-by-word co-occurance matrix based on PMI value. This prgram \
              is written by Rupak.")
        print("Processing data ...........")
        dataProcess(args.Directory,args.Window)
        print("Creating Dictionary..........")
        print("Creating PMI Matrix..........")
        frame = create_frame()
        pmi_matrix = PMI(frame)
        result = input_file(pmi_matrix)
        
        for val in result:
            print(val)
        

        