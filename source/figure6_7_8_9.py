from loadData import *
from methods import *
from tqdm import tqdm
from scipy.sparse import coo_matrix
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

pathData = "DATA/"
pathRes = "RES/"
_,_,_,_,_,_,_,leagues,matchesLeague,_ = loaddata(pathData,True)

model = load_model(f"{pathRes}SCORE_all.h5")

layer_name = model.layers[42].name  
latent_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

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


L = []
P = []
layer_name = model.layers[42].name   
latent_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
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
    times = [15,30,45,60,75,90]
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
    lat = latent_model.predict(res[1:, :, :],verbose=0)
    l = np.c_[np.repeat(m,6),np.c_[times,lat]].tolist()
    p = np.c_[np.repeat(m,6),np.c_[times,pred]].tolist()
    L = L + l
    P = P + p



L = np.array(L)
pca = PCA(n_components=5)  # Choose the number of components you want
pca_result = pca.fit_transform(L[:,2:])

tsne = TSNE(n_components=2, perplexity=500, n_iter=1500, n_jobs=64, verbose=True)
tsne_result = tsne.fit_transform(pca_result)

np.save("RES/INTERPRETABILITY/TSNE_INT.npy",tsne_result)
np.save("RES/INTERPRETABILITY/TSNE_PRED_INT.npy",np.array(P))
