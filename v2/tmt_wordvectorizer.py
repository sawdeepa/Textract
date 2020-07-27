# -*- coding: utf-8 -*-
"""
Created on Fri May 18 15:46:27 2018

@author: nchakravarthy
"""

##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np
import gensim
import os
from gensim import utils
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle


#Internal Libraries
###################
from tmt_settings import Settings
#pretrained_vectors = gensim.models.KeyedVectors.load_word2vec_format(Settings.pretrained_vectors_filepath, binary=True,limit=200000)


#########################
#Word Embedding Generator
#########################
def generate_word_embeddings_1(text_corpus,pretrained_type=None):
    
    Settings.logger.info("Generating Word Embeddings")
    if Settings.use_pretrained_vectors:
        if pretrained_type == "word2vec":
            if Settings.use_existing_build:
                Settings.logger.info("Loading pretrained w2v")
                word_embeddings = gensim.models.Word2Vec.load(Settings.pretrained_vectors_folder+"PreProcessedAllText_OverlappingWord2VecEmbeddings")
            else:
                word_embeddings = gensim.models.Word2Vec(text_corpus,size=300)
                word_embeddings.intersect_word2vec_format(Settings.pretrained_vectors_word2vec_filepath,lockf=0,binary=True)
                word_embeddings.train(text_corpus,total_examples=word_embeddings.corpus_count,epochs=word_embeddings.iter)
                word_embeddings.save(Settings.pretrained_vectors_folder+"PreProcessedAllText_OverlappingWord2VecEmbeddings")
        elif pretrained_type == "fasttext":
            if Settings.use_existing_build:
                Settings.logger.info("Loading pretrained fasttext")
                word_embeddings = gensim.models.Word2Vec.load(Settings.pretrained_vectors_folder+"PreProcessedAllText_OverlappingFastTextEmbeddings")
            else:
                word_embeddings = gensim.models.Word2Vec(text_corpus,size=300)
                word_embeddings.intersect_word2vec_format(Settings.pretrained_vectors_fasttext_filepath,lockf=0,binary=False)
                word_embeddings.train(text_corpus,total_examples=word_embeddings.corpus_count,epochs=word_embeddings.iter)
                word_embeddings.save(Settings.pretrained_vectors_folder+"PreProcessedAllText_OverlappingFastTextEmbeddings")
            
            
    else:
        file_already_present = False
        for file in os.listdir(Settings.pretrained_vectors_folder):
            if file == "PreProcessedAllText_NonOverlappingEmbeddings_"+str(Settings.embedding_dimension):
                word_embeddings = gensim.models.Word2Vec.load(Settings.pretrained_vectors_folder+"PreProcessedAllText_NonOverlappingEmbeddings_"+str(Settings.embedding_dimension))
                file_already_present = True
                break
        if file_already_present != True:
            word_embeddings = gensim.models.Word2Vec(text_corpus,size=Settings.embedding_dimension,workers=1)
            word_embeddings.train(text_corpus,total_examples=word_embeddings.corpus_count,epochs=word_embeddings.iter)
            word_embeddings.save(Settings.pretrained_vectors_folder+"PreProcessedAllText_NonOverlappingEmbeddings_"+str(Settings.embedding_dimension))
        
                
        
        
    return word_embeddings


###############################
#Embedding Matrix Generator - 1
###############################
def create_embedding_matrix_1(tokenizer,word_embeddings):
    
    Settings.logger.info("Creating Embedding Matrix")
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1,Settings.embedding_dimension))
    embedding_vocab = list(word_embeddings.wv.vocab.keys())
    for word, i in word_index.items():
        if word in embedding_vocab:
            embedding_matrix[i] =  word_embeddings[word]
    
    return embedding_matrix
    

################################
#Embedding Feature Generator - 2
################################
def create_embedding_matrix_2(text_corpus,word_embeddings):
    
    Settings.logger.info("Creating Embedding Matrix")
    word_embedding_vocab = list(word_embeddings.wv.vocab.keys())
    
    temp_word_embeddings = {}
    embedding_matrix = []
    sentence_lengths = []
    null_embedding = [0 for iterator in range(word_embeddings.vector_size)]
    
    for sentence in text_corpus:
        sentence_words = sentence.split()
        sentence_embeddings = []
        sentence_length = len(sentence_words)
        sentence_lengths.append(sentence_length)
        
        #Cap length of each sentence
        if sentence_length < Settings.max_sentence_length:
            sentence_words += [''] * (Settings.max_sentence_length - len(sentence_words))
        else:
            sentence_words = sentence_words[:Settings.max_sentence_length]
        
        #Finding embeddings of each word
        for sentence_word in sentence_words:
            if sentence_word not in list(temp_word_embeddings.keys()):
                if sentence_word in word_embedding_vocab:
                    embedding_to_assign = list(word_embeddings[sentence_word])
                else:
                    embedding_to_assign = null_embedding
                    
                sentence_embeddings.append(embedding_to_assign)
                temp_word_embeddings[sentence_word] = embedding_to_assign
            else:
                sentence_embeddings.append(temp_word_embeddings[sentence_word])
                
        
        embedding_matrix.append(sentence_embeddings)
    
    return embedding_matrix, sentence_lengths


##########################################################
def generate_doc2vec_model(processed_text):
    
    text_corpus = [text.split() for text in list(processed_text) ]
    #sentence = LabeledSentence(words=[u'some', u'words', u'here'], labels=[u'SENT_1'])
    sentences= []
    
    for index,sent in enumerate(text_corpus):
        sentence = LabeledSentence(sent,['SENT_'+str(index)])
        sentences.append(sentence)
   
    ## Building vocabulary and defining the model
    model_d2v = Doc2Vec(dm=0,min_count=1,vector_size=100, alpha=0.025, workers=8)#min_alpha=0.001)# dm=0 : only doc vectors
    model_d2v.build_vocab(sentences)
    
    ## Training the model:
    shuffle(sentences)
    model_d2v.train(sentences,total_examples=model_d2v.corpus_count,epochs = 20)
    model_d2v.save(Settings.pretrained_vectors_folder+"model_d2v")
    
    return model_d2v
    
    
def  create_doc2vec_features(doc_indices,model):
    
    doc_vecs = [] 
    doc_index = []
    
    for i in doc_indices:
        doc_vec = model['SENT_'+ str(i)]
        doc_vecs.append(doc_vec)
        doc_index.append(i)
        
    doc_vecs_df = pd.DataFrame(doc_vecs)
    doc_vecs_df['doc_index'] = pd.Series(doc_index)      
    
    return doc_vecs_df
    
    
    
