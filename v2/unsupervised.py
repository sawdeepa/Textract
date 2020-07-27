import pandas as pd
import numpy as np
import gensim
import os
from gensim import utils
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle
import random
import logging


import re
import nltk
from nltk.util import ngrams
import collections
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

import gensim
from gensim import corpora
from scipy.spatial.distance import cdist
random.seed(123)
np.random.seed(123)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

####################################### Doc2Vec ################################


## function-1
def generate_doc2vec_model(processed_text):    
	text_corpus = [text.split() for text in list(processed_text) ]   
	sentences= []    
	for index,sent in enumerate(text_corpus):
		if index % 1000 == 0:
			print(index)
		sentence = LabeledSentence(sent,['SENT_'+str(index)])
		sentences.append(sentence)   
	## Building vocabulary and defining the model
	model_d2v = Doc2Vec(dm=0,min_count=1,vector_size=100, alpha=0.025, workers=8)#min_alpha=0.001)# dm=0 : only doc vectors
	model_d2v.build_vocab(sentences)    
	## Training the model:
	shuffle(sentences)
	model_d2v.train(sentences,total_examples=model_d2v.corpus_count,epochs = 20)
	#model_d2v.save("./../Data/Saved_Models/self_trained/model_d2v")    
	return model_d2v  

## function-2:
def create_doc2vec_features(doc_indices,model):        
	doc_vecs = [] 
	doc_index = []        
	for i in doc_indices:
		doc_vec = model['SENT_'+ str(i)]
		doc_vecs.append(doc_vec)        
		doc_index.append(i)            
	doc_vecs_df = pd.DataFrame(doc_vecs)
	doc_vecs_df['doc_index'] = pd.Series(doc_index)       
	return doc_vecs_df
	
	
def create_tsne_clusterPlot(d2v_features,cluster_no):
		
		tsne_model = TSNE(n_components=2, init='pca',  random_state=0)# perplexity=40,n_iter=2500,
		d2v_features = d2v_features.iloc[:, :-1]
		new_values = tsne_model.fit_transform(d2v_features)
		
		x = []
		y = []
			#z = []
		for value in new_values:
			x.append(value[0])
			y.append(value[1])
				#z.append(value[2])
				
		#print(len(x))
		#print(len(y))
		#print(len(labels))
			
		temp = {"X":x,"Y":y,"Cluster_Number":cluster_no}
		tsne_cluster_df = pd.DataFrame(temp)
		
		return(tsne_cluster_df)	

def clean(doc):
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation) 
	lemma = WordNetLemmatizer()
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	alpha_only = " ".join(x for x in normalized.split() if x.isalpha())
	return alpha_only
		
def format_topics_sentences(ldamodel, corpus, texts, df,text_col):
		# Init output
		sent_topics_df = pd.DataFrame()
	
		# Get main topic in each document
		for i, row in enumerate(ldamodel[corpus]):
			row = sorted(row, key=lambda x: (x[1]), reverse=True)
			# Get the Dominant topic, Perc Contribution and Keywords for each document
			for j, (topic_num, prop_topic) in enumerate(row):
				if j == 0:  # => dominant topic
					wp = ldamodel.show_topic(topic_num)
					topic_keywords = ", ".join([word for word, prop in wp])
					sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
				else:
					break
		sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
	
		# Add original text to the end of the output
		contents = pd.Series(texts)
		description = pd.Series(df[text_col])                   
		sent_topics_df = pd.concat([sent_topics_df,df], axis=1)
		return(sent_topics_df)		

def preprocessing(data,text_col):
		text = [x for x in data[text_col]]
		logger.info(len(text))
		texts = pd.DataFrame({'text':text})
		texts['text'] = texts['text'].astype(str)
		texts['text'] = texts['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
		texts['text'] = texts['text'].str.replace('[^\w\s]','')
		stop = stopwords.words('english')
		texts['text'] = texts['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		texts['text'] = texts['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
		return texts
	

	
	
	
#data_path = "C:/Users/ashtripathy/Desktop/Ashis/Text_Mining_Tool_1/Data/Airbnb/Airbnb_Texas_Rentals.csv"
#text_col = "description"
#num_clusters
#Num_Topics
#data_for_unsupervised = pd.read_csv(data_path)
	
class Unsupervised:
	
	def k_means(data_for_unsupervised,text_col,num_clusters,categ_col):
	
		logger.info("inside kmeans")
				
		

		## Creating the doc2vec features
		desc_text = data_for_unsupervised[text_col].astype(str)
		## Add the existing build code
		selftrained_d2v_model = generate_doc2vec_model(desc_text)
		selftrained_d2v_features= create_doc2vec_features(data_for_unsupervised.index, model= selftrained_d2v_model)
		logger.info("doc2vec complete")
		############################### KMeans Clustering ###############################
		#num_clusters =     
		DocVecs = selftrained_d2v_features.loc[:, selftrained_d2v_features.columns != 'doc_index']
		
		# Initalize a k-means object and use it to extract centroids
		kmeans_clustering = KMeans( n_clusters = int(num_clusters) )
		idx = kmeans_clustering.fit_predict( DocVecs )           ## idx contains the cluster numbers
		
		## Adding the cluster members to original data set
		data_for_unsupervised1 = data_for_unsupervised.copy()
		data_for_unsupervised1['Cluster_Number'] =  pd.DataFrame(idx)
		logger.info(data_for_unsupervised.columns.values.tolist())
		logger.info(data_for_unsupervised1.columns.values.tolist())
		logger.info("clustering complete")
		## Displaying a scatterplot of doctoVec
		
		
		
		logger.info("Creating_tsne_cluster_plot")
		tsne_cluster_df=create_tsne_clusterPlot(selftrained_d2v_features,idx)
		logger.info("done")
		tsne_cluster_df = tsne_cluster_df[['Cluster_Number','X','Y']] 
		tsne_cluster_df['text'] = data_for_unsupervised[text_col]
		
		
		
		categ = categ_col
		categ_list = list(data_for_unsupervised[categ])
		tsne_cluster_df.insert(loc=0,column=categ,value=categ_list)

		summary = tsne_cluster_df.groupby([categ,'Cluster_Number']).count().reset_index()
		summary = summary.pivot(index=categ, columns='Cluster_Number', values='text').reset_index().fillna(0)
		
		col_list = list(summary)[1:]
		summary['Count'] = summary.sum(axis=1).astype(int)

		for col in col_list:
			summary[col]=summary[col]/summary['Count']*100
			summary[col] = summary[col].astype(int).astype(str) + '%'
		
		
		
		return tsne_cluster_df,summary
		
	def k_means_elbow(data_for_unsupervised,text_col):
	
		logger.info("inside kmeans")
				
		

		## Creating the doc2vec features
		desc_text = data_for_unsupervised[text_col].astype(str)
		## Add the existing build code
		selftrained_d2v_model = generate_doc2vec_model(desc_text)
		selftrained_d2v_features= create_doc2vec_features(data_for_unsupervised.index, model= selftrained_d2v_model)
		logger.info("doc2vec complete")
		############################### KMeans Clustering ###############################
		num_clusters =  2   
		DocVecs = selftrained_d2v_features.loc[:, selftrained_d2v_features.columns != 'doc_index']
		
		
		# k means determine k
		distortions = []
		K = range(1,10)

		for k in K:
			kmeanModel = KMeans(n_clusters=k).fit(DocVecs)
			kmeanModel.fit(DocVecs)
			distortions.append(sum(np.min(cdist(DocVecs, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / DocVecs.shape[0])

		distortions_df = pd.DataFrame({'K':K,'distortion':distortions})
		return distortions_df
	
	
	def topic_modelling(data_for_unsupervised,text_col,Num_Topics):
		logger.info("inside topic modelling")
		
		
		
		
		docs = data_for_unsupervised[text_col].astype(str)
		
		doc_clean = [clean(doc).split() for doc in docs]     
		
		# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
		dictionary = corpora.Dictionary(doc_clean)
		
		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
		
		# Creating the object for LDA model using gensim library
		Lda = gensim.models.ldamodel.LdaModel
		
		# Running and Trainign LDA model on the document term matrix.
		ldamodel = Lda(doc_term_matrix, num_topics= Num_Topics, id2word = dictionary, passes=1)
		
		
		
		df_with_topics = format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix, texts=doc_clean, df = data_for_unsupervised,text_col = text_col)
		df_ret = df_with_topics[[text_col, 'Topic_Keywords', 'Dominant_Topic', 'Perc_Contribution']]
		
		return df_ret

	def sentiment_analysis(data_for_unsupervised,text_col,categ_col):
		logger.info("inside sentiment")
		logger.info(data_for_unsupervised.head())   
				
		texts = preprocessing(data_for_unsupervised,text_col)
		texts['score'] = texts['text'].apply(lambda x: TextBlob(x).sentiment[0] ).round(2)
		texts = texts.sort_values(by='score', ascending=False).reset_index(drop=True)
		
		categ = categ_col
		categ_list = list(data_for_unsupervised[categ])
		texts.insert(loc=0,column=categ,value=categ_list)
		def func_ratings(row):
			val_column = 'score'
			if row[val_column] > 0.60:
				return 'rating4'
			elif row[val_column] >0.15:
				return 'rating3' 
			elif row[val_column] > -0.15:
				return 'rating2' 
			elif row[val_column] > -0.60:
				return 'rating1' 
			else:
				return 'rating0'

		texts['sentiment'] = texts.apply(func_ratings, axis=1)
		summary = texts.groupby([categ,'sentiment']).count().reset_index()
		summary = summary.pivot(index=categ, columns='sentiment', values='text').reset_index().fillna(0)
		
		col_list = list(summary)[1:]
		summary['Count'] = summary.sum(axis=1).astype(int)

		for col in col_list:
			summary[col]=summary[col]/summary['Count']*100
			summary[col] = summary[col].astype(int).astype(str) + '%'
		
		logger.info("sentiment complete")
		return texts,summary
            