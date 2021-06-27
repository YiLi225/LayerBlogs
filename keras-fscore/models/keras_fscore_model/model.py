"""Macro F1score in Keras
"""
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import xgboost as xgb
from layer import Featureset, Train
from collections import Counter

#### Load in the packages 
import pandas as pd
import numpy as np
from random import sample, seed, randrange, choice
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

#### if use tensorflow=2.0.0, then import tensorflow.keras.model_selection 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Reshape, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint 


def train_model(train: Train, tf: Featureset("transaction_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `transaction_features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the model,
            passed by Layer when the training of the model starts.
        tf (layer.Featureset): Layer will return a Featureset object,
            an interface to access the features inside the
            `transaction_features`

    Returns:
       model: A trained model object

    """

    # We create the training and label data
    train_df = tf.to_pandas()      
    #### Analysis starts 
    X = train_df.drop(["transactionId", "is_fraud"], axis=1)
    Y = train_df["is_fraud"]
    

    random_state = 42
    test_size = 0.2
    
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    
    ### Check class imbalanced and log it to Layer ###
    class_cnt = Counter(train_df["is_fraud"])
    class_ratio = class_cnt[[k for k in class_cnt.keys()][0]]/class_cnt[[k for k in class_cnt.keys()][1]]
    train.log_parameter("class_ratio", class_ratio)
    train.log_parameter(f"class {[k for k in class_cnt.keys()][0]}", class_cnt[[k for k in class_cnt.keys()][0]])
    train.log_parameter(f"class {[k for k in class_cnt.keys()][1]}", class_cnt[[k for k in class_cnt.keys()][1]])
    
    ### Defining the custom metric function F1
    def custom_f1(y_true, y_pred):    
        def recall_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            
            recall = TP / (Positives+K.epsilon())    
            return recall         
        
        def precision_m(y_true, y_pred):
            TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        
            precision = TP / (Pred_Positives+K.epsilon())
            return precision 
        
        precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
        
        return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
    
    ### Defining the Callback Metrics Object and track in Layer
    class LayerMetrics(Callback):
        def __init__(self, train, validation, current_fold):   
            super(LayerMetrics, self).__init__()
            self.train = train
            self.validation = validation 
            self.curFold = current_fold
                        
        def on_train_begin(self, logs={}):        
            self.val_f1s = []
            self.val_recalls = []
            self.val_precisions = []
         
        def on_epoch_end(self, epoch, logs={}):
            val_targ = self.validation[1]   
            val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()        
        
            val_f1 = round(f1_score(val_targ, val_predict), 4)
            val_recall = round(recall_score(val_targ, val_predict), 4)     
            val_precision = round(precision_score(val_targ, val_predict), 4)
            
            self.val_f1s.append(val_f1)
            self.val_recalls.append(val_recall)
            self.val_precisions.append(val_precision)           
           
            ### Send the performance metrics to Layer to track ###
            self.train.log_metric('Epoch End F1-score', val_f1)          

    ### Building a neural nets 
    def runModel(x_tr, y_tr, x_val, y_val, epos=20, my_batch_size=64):  
        inp = Input(shape = (x_tr.shape[1],))
        
        x = Dense(1024, activation='relu')(inp)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
            
        out = Dense(1, activation='sigmoid')(x)
        model = Model(inp, out)
        
        return model    
    
    ### Preprocess the training and testing data 
    ### save 20% for final testing 
    def Pre_proc(X, Y, current_test_size=0.2, current_seed=42):    
        x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                            test_size=current_test_size, 
                                                            random_state=current_seed)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        y_train, y_test = np.array(y_train), np.array(y_test)
        return x_train, x_test, y_train, y_test
    
    trainX, testY, trainY, testY = Pre_proc(X, Y)
    
    x_train, y_train = trainX, trainY
    x_test, y_test = testY, testY
    
    
    ### CV for the model training
    models = []
    f1_cv, precision_cv, recall_cv = [], [], []
    
    current_folds = 5
    current_epochs = 20 ## 80
    current_batch_size = 16 ### 16
    
    ## macro_f1 = True for Callback 
    macro_f1 = False  
    
    kfold = StratifiedKFold(current_folds, random_state=random_state, shuffle=True)
    
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X=x_train, y=y_train)):
    
        print('---- Starting fold %d ----'%(k_fold+1))
        
        x_tr, y_tr = x_train[tr_inds], y_train[tr_inds]
        x_val, y_val = x_train[val_inds], y_train[val_inds]
        
        model = runModel(x_tr, y_tr, x_val, y_val, epos=current_epochs)
        
        if macro_f1:    
            model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[])  
            model.fit(x_tr, 
                      y_tr, 
                      callbacks=[LayerMetrics(train, validation=(x_val, y_val), current_fold=k_fold)],  
                      epochs=current_epochs, 
                      batch_size=current_batch_size,   
                      verbose=1)
        else:
            model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[custom_f1, 'accuracy'])
            history = model.fit(x_tr, 
                                y_tr,                  
                                epochs=current_epochs, 
                                batch_size=current_batch_size,   
                                verbose=1)
            #### send to metric 
            for val in history.history['custom_f1']:
                train.log_metric('Custom F1 metric', val)
      
        models.append(model)
        
        y_val_pred = model.predict(x_val)
        y_val_pred_cat = (np.asarray(y_val_pred)).round() 
    
        ### Get performance metrics 
        f1, precision, recall = f1_score(y_val, y_val_pred_cat), precision_score(y_val, y_val_pred_cat), recall_score(y_val, y_val_pred_cat)        
       
        ##### Log performance measures for each Fold
        metric_text = f'Fold {k_fold+1} f1 score = '
        train.log_metric(metric_text, f1)
       
        f1_cv.append(round(f1, 4))
        precision_cv.append(round(precision, 4))
        recall_cv.append(round(recall, 4))        
    
    ##### Log performance measures after CV
    train.log_metric('Average f1 score acorss all CV sets', np.round(np.mean(f1_cv), 4))    
        
    return model    
