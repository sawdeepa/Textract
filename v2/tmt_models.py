##########
#Libraries
##########

#External Libraries
###################
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import VotingClassifier
#from catboost import CatBoostClassifier
#from xgboost import XGBClassifier, XGBRegressor
#import xgboost
'''
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential
from keras import callbacks
'''


#Internal Libraries
###################
from tmt_settings import Settings


#####################################################
#Helper Function - Classification Report to Dataframe
####################################################
def classifaction_report_dataframe(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    return dataframe
    
    


#######################
#Classification Wrapper
#######################
def train_classifier(train_features,train_targets,selected_classifier,problem_type, ensemble_list=None, ):
    
    Settings.logger.info("Training %s Classifier" %(selected_classifier))
    train_features, train_targets = np.array(train_features), np.array(train_targets)

    if selected_classifier == "RF":        
        classifier = rf_classifier(train_features,train_targets,problem_type)
    elif selected_classifier == "GB":
        classifier = gb_classifier(train_features,train_targets,problem_type) 
    elif selected_classifier == "XG":
        classifier = xg_classifier(train_features,train_targets,problem_type) 
    elif selected_classifier == "ET":
        classifier = et_classifier(train_features,train_targets,problem_type) 
    elif selected_classifier == "CAT":
        classifier = cat_classifier(train_features,train_targets,problem_type) 
    elif selected_classifier == "KNN":
        classifier = knn_classifier(train_features,train_targets,problem_type) 
    elif selected_classifier == "MLP":
        classifier = mlp_classifier(train_features,train_targets,problem_type)
    elif selected_classifier == "ENSEMBLE":
        classifier = ensemble_classifier(train_features,train_targets,ensemble_list,problem_type)
        
    
    return classifier



###################
#Ensemble Classifer
###################
'''
def ensemble_classifier(train_features,train_targets,ensemble_list,problem_type):
    classifier_instance_list = []
    for classifier in ensemble_list:
       classifier_instance = globals()[classifier.lower()+"_classifier"](train_features,train_targets,for_ensemble=True) 
       classifier_instance_list.append((classifier,classifier_instance))
       
    ensemble_classifier = VotingClassifier(estimators=classifier_instance_list, voting='soft',n_jobs=2)
    ensemble_classifier = ensemble_classifier.fit(train_features,train_targets)
    
    return ensemble_classifier

'''
########################
#Random Forest Classifer
########################
def rf_classifier(train_features,train_targets,problem_type,for_ensemble=None):

    
    if for_ensemble:
        return rf_classifier
    
    if problem_type=='classification':
        rf_classifier = RandomForestClassifier(n_estimators=100,random_state=0)
        rf_classifier.fit(train_features,train_targets)
        return rf_classifier
    else:
        rf_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
        rf_regressor.fit(train_features,train_targets)
        return rf_regressor 

############################
#Gradient Boosting Classifer
############################
def gb_classifier(train_features,train_targets,problem_type,for_ensemble=None):


    if for_ensemble:
        return gb_classifier

    if problem_type =='classification':
        gb_classifier = GradientBoostingClassifier(n_estimators=100,
                                               learning_rate= 0.1,
                                               max_depth= 8,
                                               max_features='sqrt',
                                               subsample= 0.8,
                                               random_state= 0)
        
        gb_classifier.fit(train_features,train_targets)
        return gb_classifier
    else:
        gb_regressor = GradientBoostingRegressor(n_estimators=1000, 
                                                 max_depth=5, learning_rate=1.0)
        gb_regressor.fit(train_features,train_targets)
        return gb_regressor

####################################
#Extreme Gradient Boosting Classifer
####################################
def xg_classifier(train_features,train_targets,problem_type,for_ensemble=None):

    if for_ensemble:
        return xg_classifier

    if problem_type =='classification':
        
        xg_classifier = xgboost.XGBClassifier(n_estimators=500,random_state=0,seed=0)
        
        xg_classifier.fit(train_features,train_targets)
        return xg_classifier
    else:
        xg_regressor = xgboost.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
        
        xg_regressor.fit(train_features,train_targets)
        return xg_regressor

######################
#Extra Trees Classifer
######################
'''
def et_classifier(train_features,train_targets,for_ensemble=None):

    et_classifier = ExtraTreesClassifier(n_estimators=100,random_state=0)
    if for_ensemble:
        return et_classifier
    et_classifier.fit(train_features,train_targets)

    
    return et_classifier

'''
###########################
#Nearest Neighbor Classifer
###########################
def knn_classifier(train_features,train_targets,problem_type,for_ensemble=None):

    
    if for_ensemble:
        return knn_classifier
    
    if problem_type =='classification':
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(train_features,train_targets)
        return knn_classifier
    else:
        knn_regressor == KNeighborsRegressor(n_neighbors = 5)
        knn_regressor.fit(train_features,train_targets)
        return knn_regressor


##############
#MLP Classifer
##############
def mlp_classifier(train_features,train_targets,problem_type,for_ensemble=None):

    if for_ensemble:
        return mlp_classifier
    
    if problem_type =='classification':
        mlp_classifier = MLPClassifier(solver="lbfgs",max_iter=1000,activation="identity",hidden_layer_sizes=(10,5),random_state=0,early_stopping=False,verbose=False)
        mlp_classifier.fit(train_features,train_targets)
        return mlp_classifier
    else:
        mlp_regressor = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
                             learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000,
                                 random_state=9, verbose=False)
        mlp_regressor.fit(train_features,train_targets)
        return mlp_regressor
        
        
###################
#CatBoost Classifer
###################
'''    
def cat_classifier(train_features,train_targets):
    
    cat_classifier = CatBoostClassifier(iterations=100,random_seed=0)
    cat_classifier.fit(train_features,train_targets)

    
    return cat_classifier

'''
      
        

    
    

    