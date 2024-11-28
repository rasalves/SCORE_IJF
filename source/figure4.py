from loadData import *
from methods import *
from tqdm import tqdm
from scipy.sparse import coo_matrix
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pathData = "DATA/"
pathRes = "RES/"
_,_,_,_,_,_,_,leagues,matchesLeague,_ = loaddata(pathData,True)

model = load_model(f"{pathRes}SCORE_all.h5")

events = np.array(pd.read_csv(f"{pathData}eventsPostProcessed.csv"))
events = events[events[:,4]!=10,:]
events[events[:,4]==11,4] = 10

sides = events[:,3]
types = events[:,4]
types = ((sides - 1) * 11) + types

events = np.c_[events[:,[0,1,2,3]],types]

eventsTarget = events[~np.isin(events[:,4],np.array([4,10,11,16,22,23])),:]
eventsTarget[eventsTarget[:,4]>4,4] = eventsTarget[eventsTarget[:,4]>4,4] - 1
eventsTarget[eventsTarget[:,4]>14,4] = eventsTarget[eventsTarget[:,4]>14,4] - 1
eventsTarget[eventsTarget[:,4]>8,4] = eventsTarget[eventsTarget[:,4]>8,4] - 2


for m in tqdm(np.unique(events[:,0]),desc="Post processing matches: "):
    e = events[events[:,0]==m,:]
    E = np.array([[0,0,0,0,0,0]]).astype(float)
    times = e[:,2]
    for t in np.unique(times):
        E = np.r_[E,np.c_[e[e[:,2]==t,:],np.linspace(start=t, stop=(t+1), num=(times == t).sum()+1)[:-1]]]
    E=E[1:,:]
    y = E[~np.isin(E[:,4],np.array([4,10,15,21])),:]
    y[y[:,4]>4,4] = y[y[:,4]>4,4] - 1
    y[y[:,4]>9,4] = y[y[:,4]>9,4] - 1
    y[y[:,4]>13,4] = y[y[:,4]>13,4] - 1
    times = np.unique(np.r_[np.arange(np.unique(E[:,5])[0]+1,np.ceil(np.unique(E[:,5])[-1])),E[:,5]])[1:]
    res = np.zeros(99*22).reshape(1,99,22)
    for t in times:     
        eat = E[E[:,5]<t,0:5].astype(int)
        if (E[:,5]==t).sum() == 0:
            tat = t
        else:
            tat = E[E[:,5]==t,2]
        ts = tat - eat[:,2]
        types = eat[:,4]
        types = types[ts<90]
        ts = ts[ts<90]
        minuteM = np.array(coo_matrix((np.ones_like(ts), (ts, types)), shape=(99, 22)).todense()).reshape(1,99,22)
        res = np.vstack((res, minuteM))
    pred = model.predict(res[1:, :, :],verbose=0)
    np.savez(f'{pathRes}MATCHES/MATCH_{m}.npz', pred=np.c_[times,pred], arEvts=res[1:, :, :], tabEvts = y)


