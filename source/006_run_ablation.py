from loadData import *
from methods import *
import itertools
from datetime import datetime


pathData = "DATA/"
pathRes = "RES/"


X,Y,X_I,Y_I,omega,n,o,leagues,matchesLeague,matchesEvents = loaddata(pathData,True)

leagues = leagues.tolist()
leagues.append("all")

#Ablation 1

D = [8, 16, 24, 32]
parameters = list(itertools.product([1, 2, 4, 8, 12], [1, 3, 5, 10], D, D, [128, 256, 512, 1024]))
epochs = 100


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
        print(f"START ABLATION 1: League: {league} Parameters: {par} ")
        score = ScoreAblationConv(omega, n, o, k, ks, d1, d2, bs,ablationType = 1)
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
            score.model.save(f"{pathRes}SCORE_{league}_AB_1.h5")
            x = np.zeros(pred_test.shape[1])
            x[0:5] = pars_test
            x[5] = acc_test
            R = np.r_[x.reshape(1,x.shape[0]),pred_test]
            np.save(f"{pathRes}PRED_SCORE_{league}_AB_1.npy",R)
        print(f"END: League: {league} Parameters: {par} ")




#Ablation 2
D = [8, 16, 24, 32]
parameters = list(itertools.product([1, 2, 4, 8, 12], D, D, [128, 256, 512, 1024]))
epochs = 100



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
        k, d1, d2, bs = par
        print(f"START ABLATION 2: League: {league} Parameters: {par} ")
        score = ScoreAblationConv(omega, n, o, k, 0, d1, d2, bs,ablationType = 2)
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
            pars_test = [k,0,d1,d2,bs]
            score.model.save(f"{pathRes}SCORE_{league}_AB_2.h5")
            x = np.zeros(pred_test.shape[1])
            x[0:5] = pars_test
            x[5] = acc_test
            R = np.r_[x.reshape(1,x.shape[0]),pred_test]
            np.save(f"{pathRes}PRED_SCORE_{league}_AB_2.npy",R)
        print(f"END: League: {league} Parameters: {par} ")



#Ablation MLP

D = [8, 16, 24, 32]
parameters = list(itertools.product(D, D, [128, 256, 512, 1024]))
epochs = 100


for league in leagues:
    X = X[:,0:omega,:]
    train,val,test = matchesEvents[league]
    X_train = rshp(X[train,:,:])
    X_val = rshp(X[val,:,:])
    X_test = rshp(X[test,:,:])
    Y_train = Y[train]
    Y_val = Y[val]
    Y_test = Y[test]
    accs = []
    for par in parameters:
        d1, d2, bs = par
        print(f"START ABLATION MLP: League: {league} Parameters: {par} ")
        score = ScoreAblationMLP(omega, n, o, d1, d2, bs)
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
            pars_test = [0,0,d1,d2,bs]
            x = np.zeros(pred_test.shape[1])
            x[0:5] = pars_test
            x[5] = acc_test
            R = np.r_[x.reshape(1,x.shape[0]),pred_test]
            np.save(f"{pathRes}PRED_SCORE_ABLATION_{league}_MLP.npy",R)
        print(f"END: League: {league} Parameters: {par} ")



k, ks, d1, d2, bs  = (np.load("RES/PRED_SCORE_all.npy")[0,0:5]).astype(int)
epochs = 100



X,Y,X_I,Y_I,omega,n,o,leagues,matchesLeague,matchesEvents = loaddata(pathData,True)

leagues = leagues.tolist()
leagues.append("all")

for league in leagues:
    train,val,test = matchesEvents[league]
    Y_test = Y[test]
    np.save(f"YS/Y_test_{league}.npy",Y_test)


def MRR(pred,Y_test):
    rank = np.argsort(-pred,axis=-1)  
    res = rank.copy()
    res[:,:] = 0
    for i in range(pred.shape[1]):
        res[np.arange(Y_test.shape[0]),rank[:,i]] = (i+1) 
    mrr = np.mean(1/res[np.arange(Y_test.shape[0]),Y_test])
    return mrr


for league in leagues:
    X = X[:,0:omega,:]
    train,val,test = matchesEvents[league]
    X_test = X[test,:,:]
    Y_test = Y[test]
    model = load_model(f"MODEL/M4_{league}.h5")
    res = np.load(f"MODEL/M4_{league}.npy")
    pred_test = model.predict(X_test,verbose = 0)
    Y_pred_test = np.argmax(pred_test,axis=-1)
    acc_test = (Y_pred_test == Y_test).sum() / Y_test.shape[0]
    print(f"{league} {acc_test} {res[0,5]} {acc_test-res[0,5]} {MRR(pred_test,Y_test) - MRR(res[1:,:],Y_test)}")


