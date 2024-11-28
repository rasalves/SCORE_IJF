from loadData import *
from methods import *
import itertools
from datetime import datetime


pathData = "DATA/"
pathRes = "RES/"


X,Y,X_I,Y_I,omega,n,o,leagues,matchesLeague,matchesEvents = loaddata(pathData,True)

leagues = leagues.tolist()
leagues.append("all")


for l1 in leagues:
    for l2 in leagues:
        X = X[:,0:omega,:]
        train,val,test = matchesEvents[l2]
        X_test = X[test,:,:]
        Y_test = Y[test]
        model = load_model(f"RES/SCORE_{l1}.h5")
        pred_test = model.predict(X_test,verbose = 0)
        np.save(f"TRANSFER/SCORE_{l1}_{l2}.npy",pred_test)
        np.save(f"YS/Y_test_{l2}.npy",Y_test)
        print(f"TRANSFER {l1} {l2}")

