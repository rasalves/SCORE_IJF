from methods import *

pathRes = "RES/"

model = load_model(f"{pathRes}SCORE_all.h5")

for i in range(22):
    x = np.zeros(22*99).reshape(1,99,22)
    x[0,0,i] = 1 
    y = model(x)
    if i == 0:
        res = y[0]
    else:
        res = np.c_[res,y[0]]


np.save("RES/INTERPRETABILITY/NextEvent.npy",res.T)


