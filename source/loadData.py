import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape
from tensorflow.sparse import SparseTensor
import numpy as np
import pandas as pd





def loaddata(path="",P = False):
    X_I = np.load(f"{path}X_COUNT.npy")
    pen = X_I[X_I[:,2] == 10,:]
    Y_I = np.load(f"{path}Y.npy")
    msWO = Y_I[~np.isin(Y_I[:,3],np.array([4,10,11,16,22,23])),2]
    X_I = X_I[np.isin(X_I[:,0],msWO),:]
    Y_I = Y_I[np.isin(Y_I[:,2],msWO),:]
    np.unique(Y_I[:,3])
    Y_I[Y_I[:,3]>4,3] = Y_I[Y_I[:,3]>4,3] - 1
    np.unique(Y_I[:,3])
    Y_I[Y_I[:,3]>14,3] = Y_I[Y_I[:,3]>14,3] - 1
    np.unique(Y_I[:,3])
    Y_I[Y_I[:,3]>8,3] = Y_I[Y_I[:,3]>8,3] - 2 
    np.unique(Y_I[:,3])
    values = X_I[:,3]
    with tf.device('/CPU:0'):
        indices = tf.constant((X_I[:,0:3])[:, :3], dtype=tf.int64)
        shape = tf.constant(np.max(X_I[:,0:3], axis=0) + 1, dtype=tf.int64)
        X = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)
        X = tf.sparse.to_dense(X).numpy()
    X = X[msWO,:]
    X = np.delete(X, [10, 22], axis=2)
    Y = Y_I[:,3]
    for i in np.unique(Y):
        print(i,(Y==i).sum()/Y.shape[0],(Y==i).sum())
    o = np.unique(Y).shape[0]
    m = X.shape[1]
    n = X.shape[2]
    nm = np.unique(Y_I[:,0]).shape[0]
    ntrain = np.floor(0.90*nm).astype(int)
    nvt = ((nm - ntrain) / 2).astype(int)
    mtrain =  np.sort(np.unique(Y_I[:,0]))[0:ntrain]
    mval =  np.sort(np.unique(Y_I[:,0]))[ntrain:(ntrain+nvt)]
    mtest =  np.sort(np.unique(Y_I[:,0]))[(ntrain+nvt):]
    matches = pd.read_csv(f"{path}matchesDF.csv")
    leagues = np.array(matches['league'])
    matches = np.array(matches['MATCH_ID'])
    matchesLeague = {}
    matchesEvents = {}
    matchesLeague["base"] = [matches,leagues]
    if P:
        Y_I[:,2]=np.arange(Y_I.shape[0])
        uLeague = np.unique(leagues)
        midsTrain = Y_I[np.isin(Y_I[:,0],mtrain),2]
        print(f"#matches_train: {np.unique(Y_I[np.isin(Y_I[:,0],mtrain),0]).shape[0]}")
        print(f"#Y_train: {midsTrain.shape[0]}")
        print(f"#X_train: ",end='')
        print(X[midsTrain,:,:].sum())
        midsVal = Y_I[np.isin(Y_I[:,0],mval),2]
        print(f"#matches_val: {np.unique(Y_I[np.isin(Y_I[:,0],mval),0]).shape[0]}")
        print(f"#Y_val: {midsVal.shape[0]}")
        print(f"#X_val: ",end='')
        print(X[midsVal,:,:].sum())
        midsTest = Y_I[np.isin(Y_I[:,0],mtest),2]
        print(f"#matches_test: {np.unique(Y_I[np.isin(Y_I[:,0],mtest),0]).shape[0]}")
        print(f"#Y_test: {midsTest.shape[0]}")
        print(f"#X_test: ",end='')
        print(X[midsTest,:,:].sum())
        matchesLeague["all"] = [mtrain,mval,mtest]
        matchesEvents["all"] = [midsTrain,midsVal,midsTest]
        for l in uLeague:
            print("")
            print(l)
            ml = matches[leagues == l]
            mtrainl =  mtrain[np.isin(mtrain,ml)]
            print(f"#matches_train: {np.unique(Y_I[np.isin(Y_I[:,0],mtrainl),0]).shape[0]}")
            mvall =  mval[np.isin(mval,ml)]
            mtestl =  mtest[np.isin(mtest,ml)]
            midsTrain = Y_I[np.isin(Y_I[:,0],mtrainl),2]
            print(f"#Y_train: {midsTrain.shape[0]}")
            print(f"#X_train: ",end='')
            print(X[midsTrain,:,:].sum())
            midsVal = Y_I[np.isin(Y_I[:,0],mvall),2]
            print(f"#matches_val: {np.unique(Y_I[np.isin(Y_I[:,0],mvall),0]).shape[0]}")
            print(f"#Y_val: {midsVal.shape[0]}")
            print(f"#X_val: ",end='')
            print(X[midsVal,:,:].sum())
            midsTest = Y_I[np.isin(Y_I[:,0],mtestl),2]
            print(f"#matches_test: {np.unique(Y_I[np.isin(Y_I[:,0],mtestl),0]).shape[0]}")
            print(f"#Y_test: {midsTest.shape[0]}")
            print(f"#X_test: ",end='')
            print(X[midsTest,:,:].sum())
            matchesLeague[l] = [mtrainl,mvall,mtestl]
            matchesEvents[l] = [midsTrain,midsVal,midsTest]
    leagues = uLeague
    return [X,Y,X_I,Y_I,m,n,o,leagues,matchesLeague,matchesEvents]




