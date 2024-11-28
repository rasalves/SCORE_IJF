from methods import *


pred = np.load(f"RES/PRED_SCORE_all.npy")[1:,:]
Y = np.load(f"YS/Y_test_all.npy")


rank = np.argsort(-pred,axis=-1)  
res = rank.copy()
res[:,:] = 0
for i in range(pred.shape[1]):
    res[np.arange(Y.shape[0]),rank[:,i]] = (i+1) 


np.save("RES/FIGURES/RR.npy",np.c_[1/res[np.arange(Y.shape[0]),Y],Y])
