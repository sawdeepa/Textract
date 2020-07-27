##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np
from collections import OrderedDict
import nltk
import os
import re
#import win32com.client as win32
#from win32com.client import constants


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#Internal Libraries
###################
from tmt_settings import Settings

 


#############################################
#Sentence Data Processor
#Steps Included
#1) Removal of Numbers and Special Characters
#2) Convert to lowercase
#3) Lemmatization
#4) Remove Stop Words
#5) Tokenization
#6) Noise Filtering
#############################################
def process_sentence_data(text_column,train=False,tokenizer=None):    
    
    Settings.logger.info("Preprocessing Text")
    
    #Temporary Dataframe
    ####################
    temp_text_df = pd.DataFrame()
    temp_text_df["text"] = text_column
    
    
    #Removing Number and Special Characters from String and Convert to LowerCase, Lemmatization and Stop Word Removal
    #################################################################################################################
    stopwords_english = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()    
    
    
    def custom_text_func(x):
        
        x = re.sub('[^ a-zA-Z]', '', x).lower()
        tokens = nltk.word_tokenize(x)
        tokens_pos = pos_tag(tokens)
        cleaned_tokens = []
        
        for (token, pos) in tokens_pos:
            if token not in stopwords_english:
                important_tag = pos[0].lower()
                if important_tag == "j": important_tag = "a"
                important_tag = important_tag if important_tag in ['a', 'r', 'n','v'] else None
                
                if important_tag is None:
                    lemma = token
                else:
                    lemma = lemmatizer.lemmatize(token,important_tag)
            
                cleaned_tokens.append(lemma)
                
        
        cleaned_text = " ".join(cleaned_tokens)
        return cleaned_text
                
    text_column = temp_text_df["text"].apply(custom_text_func)
    return text_column

################
#Text Tokenizer
###############
def initiate_tokenizer(text_corpus):
    
    tokenizer = Tokenizer(num_words=Settings.max_vocabulary_size,oov_token="OUTOFVOCAB")
    tokenizer.fit_on_texts(text_corpus)

    Settings.logger.info("Initiating Tokenizer")
    return tokenizer
    
    
        

               