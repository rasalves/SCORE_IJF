import pandas as pd
import numpy as np
from tqdm import tqdm


print("Loading Files")
kaggle_events = pd.read_csv("DATA/events.csv").fillna(-1)
kaggle_matches = pd.read_csv("DATA/ginf.csv")
events_id = pd.read_csv("DATA/events_id.csv").fillna(-1)
match_ids = pd.read_csv("DATA/match_ids.csv")

mids = np.array(kaggle_matches['id_odsp']) 

leagues = []
for m in tqdm(mids,desc ="Extracting league match"):
    l = (np.array(kaggle_matches['league'])[kaggle_matches['id_odsp'] == m])[0]
    leagues.append(l)
    
matchDict = {}
for index, row in tqdm(match_ids.iterrows(), desc="Setting Match ID"):
    matchDict[row.id_odsp] = row.game_id  

match_ids['leagues'] = leagues
events = np.c_[np.array(kaggle_events['id_odsp'].map(matchDict)), np.array(kaggle_events[['sort_order', 'time','side']])]
events = np.c_[events, np.repeat(-1,events.shape[0])]


matchesDF = kaggle_matches[['id_odsp','league','ht','at','fthg','ftag']].copy()
matchesDF['MATCH_ID'] = matchesDF['id_odsp'].map(matchDict)
matchesDF = matchesDF.sort_values(by='MATCH_ID')
matchesDF = matchesDF.reset_index(drop=True)

#Creating event tables

for index, row in tqdm(events_id.iterrows(), desc="Creating event table"):
    id_event = row['id_event']
    event_type = row['event_type']
    event_type2 = row['event_type2']
    shot_outcome = row['shot_outcome']
    is_goal = row['is_goal']
    v = np.array((kaggle_events.event_type == event_type) & (kaggle_events.event_type2 == event_type2)  & (kaggle_events.shot_outcome == shot_outcome)  & (kaggle_events.is_goal == is_goal))
    events[v,4] = id_event


#Processing events

events[(np.array(kaggle_events.event_type) == 9) & (events[:,4] == -1) ,4] = 9
events[(np.array(kaggle_events.event_type) == 10) & (events[:,4] == -1) ,4] = 13
events[(np.array(kaggle_events.event_type) == 3) & (events[:,4] == -1) ,4] = 0
events = events[np.lexsort((events[:, 1], events[:, 0]))]

events = events[events[:,4]!=13,:]
events = events[events[:,4]!=12,:]

XS = []
YS = []
a=0
for nat in tqdm(range(100),desc="Batching events"):
    evts = np.array_split(np.unique(events[:,0]),100)[nat]
    X = np.array([0,0,0])
    Y = np.array([0,0,0,0])
    for match_id in evts:
        match_events = events[events[:,0]==match_id,:]
        sequence = match_events[match_events[:,1] >= (match_events[np.argsort(match_events[:,1]),1])[5] ,1]
        for s in sequence:
            previous_events = match_events[match_events[:,1]<s ,:]
            y = match_events[match_events[:,1]==s ,:]
            times = previous_events[:,2] 
            sides = previous_events[:,3]
            types = previous_events[:,4]
            types = ((sides - 1) * 12) + types
            times = np.max(times) - times
            r =  ((y[0,3] - 1) * 12) + y[0,4]
            X = np.vstack([X,np.vstack([np.repeat(a,times.shape[0]),times,types]).T])
            Y = np.vstack([Y,np.array([y[0,0],y[0,1],a,r])])
            a = a+1
    XS.append(X)
    YS.append(Y)


X = (XS[0])[1:]
Y = (YS[0])[1:]
for i in tqdm(range(1,100),desc="Building final files"):
    x = (XS[i])[1:]
    y = (YS[i])[1:]
    X = np.vstack([X,x])
    Y = np.vstack([Y,y])


print("Processing number of events")

unique_rows, counts = np.unique(X.reshape(-1, 3), axis=0, return_counts=True)
X_I = np.c_[unique_rows,counts]

print("Saving Files")

np.save("DATA/X.npy",X)
np.save("DATA/Y.npy",Y)

columns = ["MATCH_ID", "SEQUENCE", "TIME", "SIDE", "EVENT"]
events_df = pd.DataFrame(events, columns=columns)
events_df.to_csv('DATA/eventsPostProcessed.csv', index=False)

np.save("DATA/X_COUNT.npy",X_I)
matchesDF.to_csv('DATA/matchesDF.csv', index=False)

print("Processing data ended")
