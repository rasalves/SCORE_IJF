import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape
from tensorflow.sparse import SparseTensor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback



from tensorflow.keras import layers, Model
import tensorflow as tf

#m: window time observed in the temporal convolutions
#n: number of events observed as input
#k: number of convolutions
#ks: kernel size for the emporal dimension
#d1 / d2: number of neurons layer
#bs: batch size
class Score:
    def __init__(self, m, n, o, k, ks, d1, d2, bs):
        self.m = m
        self.n = n
        self.o = o
        self.k = k
        self.ks = ks
        self.d1 = d1
        self.d2 = d2
        self.bs = bs
        self.model = self.build_model()
    def build_model(self):
        input_layer = layers.Input((self.m, self.n), name='INPUT')
        C = []        
        for i in range(self.k):
            C.append(layers.Conv1D(1, kernel_size=self.ks, activation='relu', use_bias=True, name=f'CONV_TIME_{i}')(input_layer))    
        for i in range(self.k):
            C.append(layers.Conv1D(1, 1, activation='relu', use_bias=True, name=f'CONV_TYPE_{i}')(tf.transpose(input_layer, perm=[0, 2, 1])))   
        x = C[0]
        for i in range(1, len(C)):
            x = tf.concat([x, C[i]], axis=1)
        x = tf.squeeze(x, axis=-1)
        x = layers.Dense(self.d1, activation='relu')(x)
        x = layers.Dense(self.d2, activation='relu')(x)
        x = layers.Dense(self.o, activation='relu')(x)
        x = layers.Activation('softmax', name='SOFTMAX')(x)
        model = Model(input_layer, x)
        model.summary()
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model
    def get_model(self):
        return self.model
    def fit(self,X_train,Y_train,X_val,Y_val,epc=100):
        early_stopping = EarlyStopping(monitor='val_accuracy',patience=5, restore_best_weights=True)
        self.model.fit(x=X_train,y=Y_train,validation_data=[X_val,Y_val],epochs=epc,batch_size=self.bs,callbacks=[early_stopping])





class ScoreAblationConv:
    def __init__(self, m, n, o, k, ks, d1, d2, bs, ablationType):
        self.m = m
        self.n = n
        self.o = o
        self.k = k
        self.ks = ks
        self.d1 = d1
        self.d2 = d2
        self.bs = bs
        self.ablationType  = ablationType 
        self.model = self.build_model()
    def build_model(self):
        input_layer = layers.Input((self.m, self.n), name='INPUT')
        C = []
        if self.ablationType == 1:        
            for i in range(self.k):
                C.append(layers.Conv1D(1, kernel_size=self.ks, activation='relu', use_bias=True, name=f'CONV_TIME_{i}')(input_layer))    
        else:
            for i in range(self.k):
                C.append(layers.Conv1D(1, 1, activation='relu', use_bias=True, name=f'CONV_TYPE_{i}')(tf.transpose(input_layer, perm=[0, 2, 1])))   
        x = C[0]
        for i in range(1, len(C)):
            x = tf.concat([x, C[i]], axis=1)
        x = tf.squeeze(x, axis=-1)
        x = layers.Dense(self.d1, activation='relu')(x)
        x = layers.Dense(self.d2, activation='relu')(x)
        x = layers.Dense(self.o, activation='relu')(x)
        x = layers.Activation('softmax', name='SOFTMAX')(x)
        model = Model(input_layer, x)
        model.summary()
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model
    def get_model(self):
        return self.model
    def fit(self,X_train,Y_train,X_val,Y_val,epc=100):
        early_stopping = EarlyStopping(monitor='val_accuracy',patience=5, restore_best_weights=True)
        self.model.fit(x=X_train,y=Y_train,validation_data=[X_val,Y_val],epochs=epc,batch_size=self.bs,callbacks=[early_stopping])



class ScoreAblationMLP:
    def __init__(self, m, n, o, d1, d2, bs):
        self.m = m
        self.n = n
        self.o = o
        self.d1 = d1
        self.d2 = d2
        self.bs = bs
        self.model = self.build_model()
    def build_model(self):
        input_layer=layers.Input((self.m*self.n,), name='INPUT')
        x = layers.Dense(self.d1,activation='relu')(input_layer)
        x = layers.Dense(self.d2,activation='relu')(x)
        x = layers.Dense(self.o,activation='relu')(x)
        x= layers.Activation('softmax', name = "SOFTMAX")(x)
        model = Model(input_layer, x)
        model.summary()
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model
    def get_model(self):
        return self.model
    def fit(self,X_train,Y_train,X_val,Y_val,epc=100):
        early_stopping = EarlyStopping(monitor='val_accuracy',patience=5, restore_best_weights=True)
        self.model.fit(x=X_train,y=Y_train,validation_data=[X_val,Y_val],epochs=epc,batch_size=self.bs,callbacks=[early_stopping])



def rshp(x):
    return x.reshape((x.shape[0],-1))


def MRR(pred,Y_test):
    rank = np.argsort(-pred,axis=-1)  
    res = rank.copy()
    res[:,:] = 0
    for i in range(pred.shape[1]):
        res[np.arange(Y_test.shape[0]),rank[:,i]] = (i+1) 
    mrr = np.mean(1/res[np.arange(Y_test.shape[0]),Y_test])
    return mrr


def accK(p, y,K):
    ps = np.argsort(-p,axis =1)
    result = np.array([y[i] in (ps[:,0:K])[i] for i in range(y.shape[0])])
    r = result.sum()/ result.shape[0]
    return r