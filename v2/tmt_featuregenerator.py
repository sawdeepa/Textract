# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:21:41 2018

@author: nchakravarthy
"""

##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE

#Internal Libraries
###################
from tmt_settings import Settings



############################
#Embedding Feature Generator
############################
def create_embedding_features(embedding_matrix, sentence_lengths, column_suffix,movingaverage_embedding_matrix=None):
    
    Settings.logger.info("Creating Embedding Features")
    embedding_features = pd.DataFrame()
    
    if movingaverage_embedding_matrix is None:
        movingaverage_embedding_matrix = embedding_matrix
    
    sentence_embedding_sum = []
    moving_averages = []
    window_size = int(Settings.max_sentence_length/Settings.moving_average_windows) if Settings.moving_average_windows != 0 else 0
    
    for sentence_embedding, movingaverage_sentence_embedding, sentence_length in zip(embedding_matrix,movingaverage_embedding_matrix,sentence_lengths):

        #Basic Summation of Embedding Features
        ######################################
        sentence_embedding_sum.append(list(np.mean(sentence_embedding,axis=0)))

        
        sentence_moving_averages = []
        #Moving Average of Embeddings
        #############################
        for window_number in range(Settings.moving_average_windows):

            window_sentence_embedding = movingaverage_sentence_embedding[(window_number)*window_size:min((window_number+1)*window_size,Settings.max_sentence_length)]
            sentence_moving_averages.append(list(np.mean(window_sentence_embedding,axis=0)))
        
        moving_averages.append(sentence_moving_averages)
            
        
    sentence_embedding_sum = np.array(sentence_embedding_sum)
    moving_averages = np.array(moving_averages)
    

    for iterator in range(sentence_embedding_sum.shape[1]):
        embedding_features[column_suffix+"_EmbeddingSum_"+str(iterator+1)] = sentence_embedding_sum[:,iterator]
    
    for iterator in range(moving_averages.shape[1]):
        for iterator_2 in range(moving_averages.shape[2]):
            embedding_features[column_suffix+"_MovingAverage_"+str(iterator+1)+"_"+str(iterator_2+1)] = moving_averages[:,iterator,iterator_2]
    
    return embedding_features
        

##########################
#Keyword Feature Generator
########################## 
def create_keyword_features(text_corpus,total_text_corpus=None,bigram_vectorizer=None):
    
    Settings.logger.info("Creating Keyword Features")
    text_corpus = [text_corpa.lower() for text_corpa in text_corpus]
    keyword_features = pd.DataFrame()
    
    #Keyword Count Features
    #######################
    count_vectorizer = CountVectorizer(vocabulary=list(set(Settings.keyword_vocabulary)))
    count_features = count_vectorizer.fit_transform(text_corpus).toarray()
    count_features_columnnames = ["CountFeatures_"+columnname for columnname in count_vectorizer.get_feature_names()]
    for feature_number, columnname in zip(range(count_features.shape[1]),count_features_columnnames):
        keyword_features[columnname] = count_features[:,feature_number]

    
    
    #TF-IDF Features
    ################
    tfidf_transformer = TfidfTransformer(smooth_idf=True)
    if total_text_corpus is not None:
        total_text_corpus_count_vectorizer = CountVectorizer(vocabulary=list(set(Settings.keyword_vocabulary)))
        total_text_corpus_count_features = total_text_corpus_count_vectorizer.fit_transform(total_text_corpus).toarray()
        tfidf_transformer = tfidf_transformer.fit(total_text_corpus_count_features)
    else:
        tfidf_transformer = tfidf_transformer.fit(count_features)
        
    tfidf_features = tfidf_transformer.transform(count_features).toarray()
    for feature_number, columnname in zip(range(tfidf_features.shape[1]),count_features_columnnames):
        keyword_features[columnname] = tfidf_features[:,feature_number]
        
        
    
    #Bigrams Count Features
    #######################
    if bigram_vectorizer is None:
        bigram_vectorizer = CountVectorizer(analyzer = "word",ngram_range=(2, 3),min_df=5)
        bigram_vectorizer = bigram_vectorizer.fit(text_corpus)
        
    bigram_features = bigram_vectorizer.transform(text_corpus).toarray()
    bigram_features_columnnames = bigram_vectorizer.get_feature_names()
    
    for feature_number, columnname in enumerate(bigram_features_columnnames):
        #(any(keyword in columnname for keyword in list(set(Settings.keyword_vocabulary))))
        if  any(word in columnname for word in Settings.bigram_keywords) or any(word in columnname for word in Settings.trigram_keywords):
            keyword_features["ngramCountFeatures_"+columnname] = bigram_features[:,feature_number]
        
       
    return keyword_features, bigram_vectorizer
    
    
#########################################
## tSNE Visualization
#########################################
         

def tsne_plot(model,given_word=None):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    
    if given_word:
        # get close words
        close_words = model.wv.similar_by_word(given_word)
        for word in close_words:
            tokens.append(model.wv[word[0]])
            labels.append(word[0])
    else:
        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)
    
    tsne_model = TSNE(n_components=2, init='pca',  random_state=0)# perplexity=40,n_iter=2500,
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def tsne_plot_for_closest_words(model,word):
    
    arr = np.empty((0,300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
           
    

    
    
    
    
        
        
    
                
    
    