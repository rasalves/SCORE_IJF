from loadData import *
from methods import *
leagues = ['E0', 'F1', 'D1', 'I1', 'SP1']

for l1 in leagues:
    mrr = f""
    acc = f""
    for l2 in leagues:
        pred = np.load(f"TRANSFER/SCORE_{l1}_{l2}.npy")
        Y_pred = np.argmax(pred,axis=-1)
        Y = np.load(f"YS/Y_test_{l2}.npy")
        acc = acc +  f" & {format((Y_pred == Y).sum() / Y.shape[0],'.4f')}"
        mrr = mrr + f" & {format(MRR(pred,Y), '.4f')}"
    print(f"# {l1} {acc} {mrr}")




# E0  & 0.2174 & 0.2138 & 0.2068 & 0.2123 & 0.2223  & 0.3941 & 0.3905 & 0.3877 & 0.3914 & 0.3922
# F1  & 0.2141 & 0.2176 & 0.2211 & 0.2348 & 0.2279  & 0.3918 & 0.3951 & 0.3955 & 0.4066 & 0.4006
# D1  & 0.2114 & 0.2010 & 0.2109 & 0.2205 & 0.2205  & 0.3886 & 0.3811 & 0.3915 & 0.3963 & 0.3920
# I1  & 0.2155 & 0.2095 & 0.2126 & 0.2309 & 0.2322  & 0.3933 & 0.3884 & 0.3910 & 0.4068 & 0.4035
# SP1  & 0.2176 & 0.2130 & 0.2265 & 0.2360 & 0.2353  & 0.3977 & 0.3945 & 0.4022 & 0.4120 & 0.4101