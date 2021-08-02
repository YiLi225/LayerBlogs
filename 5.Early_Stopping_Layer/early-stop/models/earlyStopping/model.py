"""Early stopping in keras
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


def train_model(train: Train, tf: Featureset("passenger_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this model to be trained along with the parameters.
    Just like the `passenger_features` featureset, you can add more
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

    # create the training and label data
    train_df = tf.to_pandas()

    X = train_df.drop(["PassengerId", "Survived"], axis=1)
    Y = train_df["Survived"]    

    # specify seed and testing size 
    random_state = 9125
    test_size = 0.1    
    train.log_parameter("test_size", test_size)

    # check and log to Layer the class distribution 
    class_cnt = Counter(train_df["Survived"])
    
    class_ratio = class_cnt[[k for k in class_cnt.keys()][0]]/class_cnt[[k for k in class_cnt.keys()][1]]
    train.log_parameter("class_ratio", class_ratio)
    
    ### Building a neural nets 
    def runModel(cols):
        ## # of col = 7
        inp = Input(shape = (cols,))
        
        x = Dense(128, activation='relu')(inp)
        # x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        ## x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # x = Dense(64, activation='relu')(x)
        # x = BatchNormalization()(x)

        out = Dense(1, activation='sigmoid')(x)
        model = Model(inp, out)
        
        return model    
    
    ### Preprocess the training and testing data 
    def Pre_proc(X, Y, current_test_size, current_seed=random_state):    
        x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                            test_size=current_test_size, 
                                                            random_state=current_seed)
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        
        y_train, y_test = np.array(y_train), np.array(y_test)
        return x_train, x_test, y_train, y_test
    
    x_train, x_test, y_train, y_test = Pre_proc(X, Y, current_test_size=test_size)
    
    train.log_parameter('train_columns', x_train.shape[1])
   
    current_epochs = 4000 ## 80
    current_batch_size = 64 ### 16
    current_validation_ratio = 0.25    
        
    ## 6 features in the dataset
    model = runModel(cols=6)
    
    model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
    
    #############==== Select Early Stopping or not ===###################
    # EARLY_STOPPING, CURRENT_PATIENCE = True, 3
    # train.log_parameter("early_stopping_indicator", 1)
    # train.log_parameter("patience", CURRENT_PATIENCE)
    # train.log_parameter("batch_size", current_batch_size)
    # train.log_parameter("validation_ratio", current_validation_ratio)
    
    EARLY_STOPPING = False
    train.log_parameter("early_stopping_indicator", 0)
    train.log_parameter("total_epochs", current_epochs)
    train.log_parameter("batch_size", current_batch_size)
    train.log_parameter("validation_ratio", current_validation_ratio)
    #############==== Select Early Stopping or not ===###################
  
    
    if EARLY_STOPPING: ## v22.1
        CURRENT_PATIENCE = 3
        
        es = EarlyStopping(# monitor='accuracy', 
                           # mode='max', 
                           monitor='val_loss', 
                           mode='min',
                           restore_best_weights=True,
                           verbose=1,
                           patience=CURRENT_PATIENCE)
        
        mc = ModelCheckpoint('best_model.h5', 
                             # monitor='accuracy', 
                             # mode='max',                            
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True, verbose=1, save_weights_only=True)  
        
        history = model.fit(x_train, 
                            y_train, 
                            callbacks=[es, mc], 
                            validation_split=current_validation_ratio, 
                            epochs=current_epochs, 
                            batch_size=current_batch_size,   
                            verbose=1)
        #### send the model loss to Layer for logging
        train.log_metric('Avg. Training Loss', np.round(np.mean(history.history['loss']), 3))
        train.log_metric('Avg. Validation Loss', np.round(np.mean(history.history['val_loss']), 3))
       
    else: ## v20.1       
        history = model.fit(x_train, 
                            y_train,  
                            validation_split=current_validation_ratio,
                            epochs=current_epochs, 
                            batch_size=current_batch_size,   
                            verbose=1)
        
        #### send the model loss to Layer for logging
        train.log_metric('Avg. Training Loss', np.round(np.mean(history.history['loss']), 3))
        train.log_metric('Avg. Validation Loss', np.round(np.mean(history.history['val_loss']), 3))

    
    #### Plot the train history        
    def plot_metric(history, metric):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["training_"+metric, 'validation_'+metric])
        ## plt.show()
        
        plt.savefig(r'C:\Users\entty\ModelTrain_noES.png')
    
    plot_metric(history, 'loss')
    
    ##### Log performance measures after CV
    # train.log_metric('Average f1 score acorss all CV sets', np.round(np.mean(f1_cv), 4))  
    
    return model
   
    
     
    
    
    