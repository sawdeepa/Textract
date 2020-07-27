##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.manifold import TSNE


#Internal Libraries
###################
import sys

sys.path.append("./")
#sys.path.append(r.python_file_path)

from tmt_settings import Settings
from tmt_processor import *
from tmt_models import *
from tmt_wordvectorizer import *
from tmt_featuregenerator import *

#Warning Filter
###############
import warnings
warnings.filterwarnings("ignore")


def run_supervised(sentence_dataframe,text_col,target_col,classifier_name,problem_type,feature_type):
	train_text_holder = []
	test_text_holder = []
	train_targets_holder = []
	test_targets_holder = []
	
	sentence_dataframe = sentence_dataframe.dropna(axis=0, subset=[text_col]).reset_index()
	
	all_text = sentence_dataframe[text_col]
	all_targets = sentence_dataframe[target_col]
	
	#print(all_text.head())
	#print(all_targets.head())
	#print(len(all_text))
	#print(len(all_targets))
	
	if Settings.validation_split == "normal":
		if problem_type == "classification":
			train_text,test_text,train_targets,test_targets = train_test_split(all_text,all_targets,stratify=all_targets,test_size=0.3,random_state=0)
		else:
			train_text,test_text,train_targets,test_targets = train_test_split(all_text,all_targets,test_size=0.3,random_state=0)

		train_text_holder.append(train_text)
		test_text_holder.append(test_text)
		train_targets_holder.append(train_targets)
		test_targets_holder.append(test_targets) 
	else:
		
		stratified_folds = StratifiedKFold(n_splits=5,random_state=0)
		for train_index, test_index in stratified_folds.split(sentence_dataframe[text_col], sentence_dataframe[target_col]):
			train_text_holder.append(all_text[train_index])
			test_text_holder.append(all_text[test_index])
			train_targets_holder.append(all_targets[train_index])
			test_targets_holder.append(all_targets[test_index]) 
		


	#Iterating over each fold for model building and validation
	###########################################################
	test_predictions_holder = []
	test_confusion_matrix_holder = []
	test_classification_report_holder = []
	
	train_pretrained_word2vec_embedding_features=pd.DataFrame()
	train_selftrained_d2v_features = pd.DataFrame()
	train_pretrained_fasttext_embedding_features= pd.DataFrame()
	train_keyword_features=pd.DataFrame()
	test_pretrained_word2vec_embedding_features=pd.DataFrame()
	test_selftrained_d2v_features = pd.DataFrame()
	test_pretrained_fasttext_embedding_features= pd.DataFrame()
	test_keyword_features=pd.DataFrame()
	
	
	for fold_number, (train_text, test_text, train_targets, test_targets) in enumerate(zip(train_text_holder,test_text_holder,train_targets_holder,test_targets_holder)):
		
		Settings.logger.info("Starting Validation - Fold Count %d"%(fold_number+1))
		
		#Text Data Preprocessing
		########################
		train_processed_text = process_sentence_data(text_column=train_text)
		test_processed_text = process_sentence_data(text_column=test_text)
		
		
		if fold_number == 0 and Settings.model == "normal":   
			total_text_corpus_fulltext = sentence_dataframe[text_col].astype(str)
			total_text_corpus_fulltext_raw=sentence_dataframe[text_col].astype(str)
			total_text_corpus = [text_corpus_sentence.split() for text_corpus_sentence in list(total_text_corpus_fulltext)]
			
			if 'WV' in feature_type:   
				Settings.use_pretrained_vectors = True
				Settings.moving_average_windows = 0
				Settings.logger.info("first call")
				Settings.logger.info(Settings.embedding_dimension)
				pretrained_word2vec_word_embeddings = generate_word_embeddings_1(total_text_corpus,pretrained_type="word2vec")
			
			if 'FT' in feature_type:   
				Settings.use_pretrained_vectors = True
				Settings.moving_average_windows = 0 
				Settings.logger.info("Second Call")
				Settings.logger.info(Settings.embedding_dimension)
				pretrained_fasttext_word_embeddings = generate_word_embeddings_1(total_text_corpus,pretrained_type="fasttext")
				
			if 'WV' in feature_type:
				Settings.use_pretrained_vectors = False
				Settings.embedding_dimension = 200
				Settings.moving_average_windows = 0
				Settings.logger.info("Third Call")
				Settings.logger.info(Settings.embedding_dimension)
				selftrained_word_embeddings = generate_word_embeddings_1(total_text_corpus)
			
			
			Settings.embedding_dimension = 20
			Settings.moving_average_windows = 0  ## place to change the number of moving windows
			movingaverage_train_selftrained_embedding_matrix = None
			movingaverage_test_selftrained_embedding_matrix = None
			
			if Settings.moving_average_windows > 0:
				movingaverage_selftrained_word_embeddings = generate_word_embeddings_1(total_text_corpus)
				movingaverage_train_selftrained_embedding_matrix, train_sentence_lengths = create_embedding_matrix_2(train_processed_text,movingaverage_selftrained_word_embeddings)
				movingaverage_test_selftrained_embedding_matrix, test_sentence_lengths = create_embedding_matrix_2(test_processed_text,movingaverage_selftrained_word_embeddings)
				
		
		elif fold_number == 0 and Settings.model != "normal": 
			total_text_corpus = sentence_dataframe[text_col].astype(str)
			total_text_corpus = [text_corpus_sentence.split() for text_corpus_sentence in list(total_text_corpus)]
			tokenizer = initiate_tokenizer(total_text_corpus)
		

			
		
		if Settings.model == "normal":
			
			#Feature Creation--change herrrrrr
			#################
			
			#feature_Word2Vec= False
			#feature_FastText = False
			#feature_Doc2Vec = True
			#feature_BOW = False
			
			#Pretrained Embedding Features - Word2Vecs
			if 'WV' in feature_type:
				Settings.moving_average_windows = 0
				train_pretrained_word2vec_embedding_matrix, train_sentence_lengths = create_embedding_matrix_2(train_processed_text,pretrained_word2vec_word_embeddings)
				test_pretrained_word2vec_embedding_matrix, test_sentence_lengths = create_embedding_matrix_2(test_processed_text,pretrained_word2vec_word_embeddings)
				
				train_pretrained_word2vec_embedding_features = create_embedding_features(train_pretrained_word2vec_embedding_matrix, train_sentence_lengths,column_suffix="PreTrainedWord2Vec")
				test_pretrained_word2vec_embedding_features = create_embedding_features(test_pretrained_word2vec_embedding_matrix, test_sentence_lengths,column_suffix="PreTrainedWord2Vec")
				
			
			#Pretrained Embedding Features - FastText                
			if 'FT' in feature_type:  
				print("I am inside FastText")
				Settings.moving_average_windows = 0
				train_pretrained_fasttext_embedding_matrix, train_sentence_lengths = create_embedding_matrix_2(train_processed_text,pretrained_fasttext_word_embeddings)
				test_pretrained_fasttext_embedding_matrix, test_sentence_lengths = create_embedding_matrix_2(test_processed_text,pretrained_fasttext_word_embeddings)
				train_pretrained_fasttext_embedding_features = create_embedding_features(train_pretrained_fasttext_embedding_matrix, train_sentence_lengths,column_suffix="PreTrainedFastText")
				test_pretrained_fasttext_embedding_features = create_embedding_features(test_pretrained_fasttext_embedding_matrix, test_sentence_lengths,column_suffix="PreTrainedFastText")
				
			
			#Self Trained Embedding Features
			'''
			Settings.moving_average_windows = 0  ### Place to change the moving average length..same as line 163
			train_selftrained_embedding_matrix, train_sentence_lengths = create_embedding_matrix_2(train_processed_text,selftrained_word_embeddings)
			test_selftrained_embedding_matrix, test_sentence_lengths = create_embedding_matrix_2(test_processed_text,selftrained_word_embeddings)
			
			
			train_selftrained_embedding_features = create_embedding_features(train_selftrained_embedding_matrix, train_sentence_lengths,column_suffix="SelfTrained",movingaverage_embedding_matrix=movingaverage_train_selftrained_embedding_matrix)
			test_selftrained_embedding_features = create_embedding_features(test_selftrained_embedding_matrix, test_sentence_lengths,column_suffix="SelfTrained",movingaverage_embedding_matrix=movingaverage_test_selftrained_embedding_matrix)
			'''
			
			if 'DV' in feature_type:                    
			#	if Settings.use_existing_build:
				#	selftrained_d2v_model= gensim.models.Word2Vec.load(Settings.selftrained_d2v_filepath)                    
				#else:
				selftrained_d2v_model = generate_doc2vec_model(all_text) 

										  
				train_selftrained_d2v_features= create_doc2vec_features(train_processed_text.index, model= selftrained_d2v_model)
				test_selftrained_d2v_features = create_doc2vec_features(test_processed_text.index, model= selftrained_d2v_model)
							
			
			
			#Keyword Features
			if 'BW' in feature_type:
				train_keyword_features, bigram_vectorizer = create_keyword_features(train_text,total_text_corpus=total_text_corpus_fulltext_raw)
				test_keyword_features, bigram_vectorizer = create_keyword_features(test_text,total_text_corpus=total_text_corpus_fulltext_raw, bigram_vectorizer=bigram_vectorizer)
				
			
			#Merging All Features
			train_features = pd.concat([train_selftrained_d2v_features,train_pretrained_fasttext_embedding_features,train_keyword_features],axis=1) #train_pretrained_word2vec_embedding_features,train_selftrained_embedding_features,
			test_features = pd.concat([test_selftrained_d2v_features,test_pretrained_fasttext_embedding_features,test_keyword_features],axis=1) #test_pretrained_word2vec_embedding_features,test_selftrained_embedding_features,
			
			train_features_columns = train_features.columns.values.tolist()
			
			#print(train_features_columns)
			
			#Normalizing Features
			scaler = preprocessing.MinMaxScaler().fit(train_features)
			train_features = scaler.transform(train_features)
			test_features = scaler.transform(test_features)
			
		 
			selected_classifier =  classifier_name  ##"MLP"
			problem_type = problem_type
			validation_classifier = train_classifier(train_features,train_targets,selected_classifier=selected_classifier,problem_type=problem_type)
			
			## Uncomment to run the ensemble model.
			#validation_classifier = train_classifier(train_features,train_targets,selected_classifier="ENSEMBLE",ensemble_list=["GB","MLP"])
			train_predictions = validation_classifier.predict(np.array(train_features))
			test_predictions = validation_classifier.predict(np.array(test_features))
			
			if problem_type == "classification": 
				
				Settings.logger.info("Confusion Matrix")
				test_confusion_matrix = confusion_matrix(test_targets,test_predictions)
				print(test_confusion_matrix)
				#plt.imshow(test_confusion_matrix, cmap='binary', interpolation='None')
				#plt.show()
				test_confusion_matrix= pd.DataFrame(test_confusion_matrix,index = sorted(test_targets.unique().tolist()))
				test_confusion_matrix.columns = sorted(test_targets.unique().tolist())
				
				Settings.logger.info("Classification Report")
				#classification_report_text = classification_report(test_targets,test_predictions)
				#print(classification_report_text)
				classification_report_text = precision_recall_fscore_support(test_targets,test_predictions)
				
				def classification_report_df(clf_rep):
					out_dict = {
							 "precision" :clf_rep[0].round(2)
							,"accuracy" : clf_rep[1].round(2)
							,"f1-score" : clf_rep[2].round(2)
							,"no. of observations" : clf_rep[3]
							}
					out_df = pd.DataFrame(out_dict, index = sorted(test_targets.unique().tolist()))
					#avg_tot = (out_df.apply(lambda x: round(x.mean(), 2) if x.name!="no. of observations" else  round(x.sum(), 2)).to_frame().T)
					#avg_tot.index = ["avg/total"]
					#out_df = out_df.append(avg_tot)
					return out_df                    
									
				df_classification_report=classification_report_df(classification_report_text)
				#df_classification_report = df_classification_report[['accuracy','no. of observations']]
			
			
			## Added on 20th July
			test_predictions = pd.Series(test_predictions,index = test_targets.index,name = "Prediction")
			#test_with_predictions = pd.concat([test_text,test_targets,test_predictions],axis=1)
			test_with_predictions = pd.concat([sentence_dataframe.iloc[test_text.index,:],test_predictions],axis=1)       
	
	#test_with_predictions = test_with_predictions.drop(columns=[target_col])
	test_with_predictions = test_with_predictions[['urlDrugName',target_col,text_col,'Prediction']]
	#urlDrugName, review, predict, actual
	#add filer to drug
	
	
	return test_with_predictions,df_classification_report
           
                                    