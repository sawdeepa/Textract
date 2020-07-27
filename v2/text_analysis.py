import numpy as np
import pandas as pd
import re
import nltk
from nltk.util import ngrams
import collections
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.manifold import TSNE
#from textblob import TextBlob
import matplotlib.pyplot as plt
import logging
from gensim.models import word2vec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#sys.path.append('C:/Users/souravkumar/Documents/Competitions/Text Tool')
#data_path = "C:/Users/souravkumar/Documents/Competitions/Text Tool/Airbnb_Texas_Rentals.csv"
#text_col = "description"
#nltk.download('stopwords')
#nltk.download('wordnet')
def preprocessing(data):
				
				text = [x for x in data['text']]
				texts = pd.DataFrame({'text':text})
				#texts['text']= str(texts['text'])  #change
				texts['text'] = texts['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
				texts['text'] = texts['text'].str.replace('[^\w\s]','')
				stop = stopwords.words('english')
				texts['text'] = texts['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
				texts['text'] = texts['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
				return texts
		
		
def nGrams(data, n):
			texts_new = preprocessing(data)
			#texts_new= texts
			dict1 = dict()
			for emmaWords in texts_new['text']:
				emmagrams = list(nltk.ngrams((emmaWords.split(" ")), n))
				emmagramsFreqs = nltk.FreqDist(emmagrams[0:])
				for words, count in emmagramsFreqs.most_common(50):
					if " ".join(list(words)) in dict1:
						dict1[" ".join(list(words))] += count
					else:
						dict1[" ".join(list(words))] = count
			df = pd.DataFrame(list(dict1.items()), columns=['ngrams', 'freq'])
			df_sort = df.sort_values(by=['freq'], ascending=False)
			
			
			return df_sort
		
	
		
def EDA(data):
			texts = preprocessing(data)
			texts['word_count'] = texts['text'].apply(lambda x: len(str(x).split(" ")))
			texts['char_count'] = texts['text'].str.len()
			stop = stopwords.words('english')
			texts['stopwords'] = data['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
			texts['hastags'] = data['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
			texts['numerics'] = data['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
			texts['upper'] = data['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
			print(texts.head())
			return texts
			
def tsne_plot(model,given_word=None):
            "Creates and TSNE model and plots it"
            labels = []
            tokens = []
            
            if given_word:
                # get close words
                close_words = model.wv.similar_by_word(given_word)
                close_words.append(tuple((given_word,model.wv[given_word])))
                logger.info("close words: "+ str(close_words))
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
            #z = []
            for value in new_values:
                x.append(value[0])
                y.append(value[1])
                #z.append(value[2])
                
            print(len(x))
            print(len(y))
            print(len(labels))
            
            temp = {"Label" : labels,"X":x,"Y":y}
            x_y_z_df = pd.DataFrame(temp)
            Most_similar_words = pd.DataFrame(close_words[:-1], columns =['Word','Similarity_score'])
            return(x_y_z_df,Most_similar_words)				
class Text_Ana:
	
		
		
	def nGrams_EDA(data,n):
				#print(data.columns.values.list())
				#data = data.rename(columns = {text_col:'text'})
				#data_for_eda.columns.values[5] = "text"
				nGram_df = nGrams(data, n)
				texts_EDA = EDA(data)
				return nGram_df, texts_EDA
				
	def Extract_pattern(data,text_col,pattern_sel):		
		data_for_eda = data.rename(columns = {text_col:'text'})
		#data_for_eda['text']= str(data_for_eda['text'])

		if pattern_sel== "EP_email":
			result_mail_id = data_for_eda['text'].apply(lambda x: re.findall('[\w\.-]+@[\w\.-]+\.\w+', str(x)))
			Mail_ids = pd.DataFrame({'Mail Ids':result_mail_id})
			#logger.info("Result_mail_id" +result_mail_id)
			#logger.info(Mail_ids)
			#Mail_ids = pd.DataFrame(result_mail_id, columns=['Mail Ids'])
			return Mail_ids
			
		if pattern_sel== "EP_website":
			result_web_links = data_for_eda['text'].apply(lambda x: re.findall(r'(https?://\S+)', str(x))) #hardcoded
			Web_links = pd.DataFrame({'Web links':result_web_links})
			#logger.info("result_web_links" +result_web_links)
			logger.info(Web_links)
			#Web_links = pd.DataFrame(result_web_links, columns=['Web links'])
			return Web_links
				
	def Visualize_Similar_Words(data,text_col,word_search):		
		#data_master =  pd.read_csv("C:/Users/ashtripathy/Desktop/Ashis/Audit Text Mining/Audit_UI_Tool/Final_Files_UI/Outputs/Document Information/Training Data/AllLabeledTextv5_WithoutAnyDuplicates.csv", encoding="latin1")
		np.random.seed(0)
		#data_text = data_for_eda[text_col]
		sentences = []
		data_for_eda = data.rename(columns = {text_col:'text'})
		data_for_eda['text']= pd.DataFrame([str(x) for x in data_for_eda['text']])
		data_text=preprocessing(data_for_eda)
		logger.info("Checking.....")
		for texts in data_text['text']:
			text1 = re.sub("[^a-zA-Z]"," ", str(texts))
			sentences.append(text1.lower().split())
        
        
        ## building the model
		logger.info("Creating W2V Embeddings")

		model_w2v = word2vec.Word2Vec(sentences, workers=4, \
					size= 100,#num_features,\
					min_count = 2,#min_word_count, \
					window = 3,#context
					)

		## Generating the tsne dataframe
		logger.info("Generating tsne dataframe")
		x_y_z_df, Most_similar_words = tsne_plot(model_w2v, str(word_search)) 
		return x_y_z_df, Most_similar_words
			
			

