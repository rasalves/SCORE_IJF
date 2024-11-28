from loadData import *
from methods import *
import itertools
from datetime import datetime


pathData = "DATA/"
pathRes = "RES/"

D = [8, 16, 24, 32]
parameters = list(itertools.product([1, 2, 4, 8, 12], [1, 3, 5, 10], D, D, [128, 256, 512, 1024]))
epochs = 100

X,Y,X_I,Y_I,omega,n,o,leagues,matchesLeague,matchesEvents = loaddata(pathData,True)

leagues = leagues.tolist()
leagues.append("all")

for league in leagues:
    X = X[:,0:omega,:]
    train,val,test = matchesEvents[league]
    X_train = X[train,:,:]
    X_val = X[val,:,:]
    X_test = X[test,:,:]
    Y_train = Y[train]
    Y_val = Y[val]
    Y_test = Y[test]
    accs = []
    for par in parameters:
        k, ks, d1, d2, bs = par
        print(f"START: League: {league} Parameters: {par} ")
        score = Score(omega, n, o, k, ks, d1, d2, bs)
        score.fit(X_train,Y_train,X_val,Y_val,epochs)
        new = False
        if len(accs) == 0:
            new = True
        else:
            if (np.array(accs) > acc).sum() == 0:
                new = True
        if new:
            pred_test = score.model.predict(X_test)
            Y_pred_test = np.argmax(pred_test,axis=-1)
            acc_test = (Y_pred_test == Y_test).sum() / Y_test.shape[0]
            pars_test = [k,ks,d1,d2,bs]
            score.model.save(f"{pathRes}SCORE_{league}.h5")
            x = np.zeros(pred_test.shape[1])
            x[0:5] = pars_test
            x[5] = acc_test
            R = np.r_[x.reshape(1,x.shape[0]),pred_test]
            np.save(f"{pathRes}PRED_SCORE_{league}.npy",R)
        print(f"END: League: {league} Parameters: {par} ")






