from methods import *

baselines = ['POGBA','S2E','RF','LSTM','NAIVEBAYES','XGBOOST']

for b in baselines:
    pred = np.load(f"RES/BASELINES/{b}_all.npy")
    Y = np.load(f"YS/Y_test_all.npy")
    Y_pred = np.argmax(pred,axis=-1)
    acc = f" & {format((Y_pred == Y).sum() / Y.shape[0],'.4f')}"
    for k in [3,5,10]:
        acc = acc +  f" & {format(accK(pred, Y,k),'.4f')}"
    print(f"# {b}{acc}")

pred = np.load(f"RES/PRED_SCORE_all.npy")[1:,:]
Y = np.load(f"YS/Y_test_all.npy")
Y_pred = np.argmax(pred,axis=-1)
acc = f" & {format((Y_pred == Y).sum() / Y.shape[0],'.4f')}"
for k in [3,5,10]:
    acc = acc +  f" & {format(accK(pred, Y,k),'.4f')}"
print(f"# SCORE{acc}")

# POGBA & 0.1601 & 0.3693 & 0.5259 & 0.8014
# S2E & 0.1838 & 0.4038 & 0.5523 & 0.8242
# RF & 0.2160 & 0.4357 & 0.5921 & 0.8561
# LSTM & 0.2161 & 0.4475 & 0.6096 & 0.8683
# NAIVEBAYES & 0.1626 & 0.3555 & 0.4953 & 0.7888
# XGBOOST & 0.2222 & 0.4408 & 0.6053 & 0.8658
