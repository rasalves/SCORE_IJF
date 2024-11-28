from loadData import *
from methods import *

leagues = ['E0', 'F1', 'D1', 'I1', 'SP1','all']
baselines = ['POGBA','S2E','RF','LSTM','NAIVEBAYES','XGBOOST']

for b in baselines:
    mrr = f""
    acc = f""
    for l1 in leagues:
        pred = np.load(f"RES/BASELINES/{b}_{l1}.npy")
        Y_pred = np.argmax(pred,axis=-1)
        Y = np.load(f"YS/Y_test_{l1}.npy")
        acc = acc +  f" & {format((Y_pred == Y).sum() / Y.shape[0],'.4f')}"
        mrr = mrr + f" & {format(MRR(pred,Y), '.4f')}"
    print(f"# {b} {acc} {mrr}")


scoremodels = ['','_MLP','_AB_2','_AB_1']

for score in scoremodels:
    mrr = f""
    acc = f""
    for l1 in leagues:
        pred = np.load(f"RES/PRED_SCORE_{l1}{score}.npy")[1:,:]
        Y_pred = np.argmax(pred,axis=-1)
        Y = np.load(f"YS/Y_test_{l1}.npy")
        acc = acc +  f" & {format((Y_pred == Y).sum() / Y.shape[0],'.4f')}"
        mrr = mrr + f" & {format(MRR(pred,Y), '.4f')}"
    print(f"# SCORE{score} {acc} {mrr}")

# POGBA  & 0.1590 & 0.1506 & 0.1527 & 0.1630 & 0.1567 & 0.1601  & 0.3344 & 0.3292 & 0.3265 & 0.3404 & 0.3306 & 0.3365
# S2E  & 0.1387 & 0.1370 & 0.1622 & 0.1753 & 0.1679 & 0.1838  & 0.3175 & 0.3164 & 0.3409 & 0.3496 & 0.3373 & 0.3607
# RF  & 0.1998 & 0.1947 & 0.1959 & 0.2278 & 0.2164 & 0.2160  & 0.3710 & 0.3727 & 0.3795 & 0.3972 & 0.3866 & 0.3910
# LSTM  & 0.1832 & 0.1945 & 0.1986 & 0.2148 & 0.2149 & 0.2161  & 0.3627 & 0.3751 & 0.3765 & 0.3928 & 0.3876 & 0.3965
# NAIVEBAYES  & 0.1623 & 0.1583 & 0.1554 & 0.1513 & 0.1579 & 0.1626  & 0.3371 & 0.3292 & 0.3261 & 0.3245 & 0.3249 & 0.3303
# XGBOOST  & 0.2129 & 0.2068 & 0.2071 & 0.2278 & 0.2223 & 0.2222  & 0.3881 & 0.3834 & 0.3887 & 0.4017 & 0.3958 & 0.3976
# SCORE  & 0.2174 & 0.2176 & 0.2109 & 0.2309 & 0.2353 & 0.2358  & 0.3941 & 0.3951 & 0.3915 & 0.4068 & 0.4101 & 0.4159
# SCORE_MLP  & 0.1870 & 0.1972 & 0.1959 & 0.2176 & 0.2075 & 0.2225  & 0.3627 & 0.3750 & 0.3700 & 0.3863 & 0.3849 & 0.3969
# SCORE_AB_2  & 0.2008 & 0.2103 & 0.2180 & 0.2201 & 0.2029 & 0.2306  & 0.3808 & 0.3805 & 0.3911 & 0.3840 & 0.3704 & 0.3981
# SCORE_AB_1  & 0.1922 & 0.1895 & 0.1752 & 0.1923 & 0.2062 & 0.2057  & 0.3703 & 0.3727 & 0.3400 & 0.3557 & 0.3848 & 0.3733
