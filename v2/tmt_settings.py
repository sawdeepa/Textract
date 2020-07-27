##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np
import logging



#################
#Project Settings
#################
class Settings:
    
   
    #Logging Settings
    #################
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    
    #Control Board
    ##############
    validation_classify_sentences = True
    
    
    
    #Data Settings
    ##############
    
    #Input Folders
    data_folder = "./../Data/"
    document_folder = data_folder+"Documents/"
    lookup_folder = data_folder+"Lookups/"
    pretrained_vectors_folder = data_folder+"Pretrained Vectors/"
      
           
    
    
    #Pretrained Word2Vecs
    pretrained_vectors_word2vec_filepath = pretrained_vectors_folder+"GoogleNews-vectors-negative300.bin.gz"
    pretrained_vectors_fasttext_filepath = pretrained_vectors_folder+"wiki.en.vec"
    
    #Selftrained doc2vec:
    selftrained_d2v_filepath = pretrained_vectors_folder+ 'model_d2v'
    
    
    #Output Folder
    output_folder  = "./../Outputs/"
    document_information_folder = output_folder+"Document Information/Training Data/"
    
    #Output Files
    #document_information_file =  "AllLabeledTextv5_WithoutDuplicates.csv"
    
    #All Text File
    #all_preprocessed_text = pd.read_csv(document_information_folder+"PreProcessedAllText_v1.csv",encoding="latin1")
    
    
    
    #Word Embedding Settings
    ########################
    max_vocabulary_size = 10000
    max_sentence_length = 100
    embedding_dimension = 200
    
    moving_average_windows = None
    use_pretrained_vectors = None
    use_existing_build = True
    
    show_tsne_plot= False
    
    #Model Settings
    ###############
    model = "normal"
    validation_split = "normal"  ## anythinh other than "normal" will make KFOLD
    

