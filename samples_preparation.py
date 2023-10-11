import pandas as pd
import argparse
import time as tm
from tqdm import tqdm
from datetime import datetime
import numpy as np
from utils import samplesConstructor,deltaTimeMatricesConstructor


parser = argparse.ArgumentParser()
parser.add_argument('--prior-hours', type=int, default=24)
args = parser.parse_args()

path="volume/"

prior_hours=args.prior_hours


chartevents=pd.read_csv(f'{path}chartevents_{prior_hours}_imputed.csv',low_memory=False,engine='c')
icustays=pd.read_csv(f'{path}ICUSTAYS.csv',low_memory=False,engine='c')
admissions=pd.read_csv(f'{path}ADMISSIONS.csv',low_memory=False,engine='c')
d_items=pd.read_csv(f'{path}D_ITEMS.csv',engine='c',low_memory=False)
print(chartevents.head(3))
print('\n')

#Data description
nr_features=len(chartevents.ITEMID.unique())
nr_subjects=len(chartevents.SUBJECT_ID.unique())
nr_hadms=len(chartevents.HADM_ID.unique())
print(f'# of features {nr_features}')
print(f'# of subjects {nr_subjects}')
print(f'# of admissions {nr_hadms}')
print('\n')

chartevents=chartevents.loc[chartevents.HADM_ID.isin(icustays.HADM_ID.unique()),:]

#Record the INTIME to the ICU of all patients
ids_stay_intime={}
for s,i in zip(icustays['HADM_ID'].values,icustays['INTIME'].values):
    ids_stay_intime[s]=i

#Get all HADM_ID in chartevents
hadm_ids=chartevents['HADM_ID'].unique()
print(f'# of admissions in ICU:{len(hadm_ids)}')
print('\n')


#Get hadm_id and state of the patietns afetr each admission.
dict_state={}
classs=[]
start= tm.perf_counter()
for idnx,(hadm,statepatient) in enumerate(zip(admissions['HADM_ID'].values,admissions['HOSPITAL_EXPIRE_FLAG'].values)):
    if hadm in hadm_ids:
        dict_state[hadm]=statepatient
        classs.append(statepatient)

finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print('\n')

#Class distribution
print(f'Class distribution {pd.DataFrame(data=classs)[0].value_counts()}')#O for died 1 for alive
print('\n')

#Select only the columns needed
chartevents=chartevents[['HADM_ID','ITEMID','CHARTTIME','VALUENUM','MASK']]
chartevents.head(2)


#Extract physiological measurements, their corresponding timestamps and mask values
data={}
flags,hdms,array_of_of_length=[],[],[]
features=chartevents['ITEMID'].unique()
max_length=0

#tqdm is just for the progress bar
for hadmid in tqdm(hadm_ids):

    #ICU INTIME of the current patient
    init_date=datetime.strptime(ids_stay_intime[hadmid], '%Y-%m-%d %H:%M:%S')
    #All data of the patient at the current admission
    sub_dataframe=chartevents.loc[chartevents['HADM_ID']==hadmid,:].sort_values(by="CHARTTIME")

    data[hadmid]={}

    #We iterate across each selected features (streams) to save their values, timestamps and masks
    for item in features:
        sub_dataframe_tempo=sub_dataframe.loc[sub_dataframe['ITEMID']==item,:].sort_values(by="CHARTTIME").values
        #Temporary array to save timestamps, values and masks on each iteration
        tempo=[]

        #We iterate over the stream
        for idx,(chartime,valuenum,msk) in enumerate(zip(sub_dataframe_tempo[:,2],sub_dataframe_tempo[:,3],sub_dataframe_tempo[:,4])):

            #Current at which the value was recorded
            current_date=datetime.strptime(chartime, '%Y-%m-%d %H:%M:%S')
            #Subtract the current date from the intime date to obtain the timestamp value
            diff = current_date-init_date
            #Convert the timestamp in hours
            timestamp=diff.days*24 + diff.seconds/3600

            #If the timestamp is less than the prior_hours parameter, we save the timestamp, the value and the mask
            #Otherwise, we stop looping over the stream
            if (timestamp<=prior_hours) and (timestamp>=0):
                tempo.append([timestamp,valuenum,msk])
            elif (timestamp>prior_hours):
                break

        #We check if the patient has values for the current feature
        if len(tempo)>0:
            data[hadmid][item]=[]
            data[hadmid][item].append(tempo)
            array_of_of_length.append(len(tempo))#Extract physiological measurements, their corresponding timestamps and mask values


counter=0
if prior_hours==24:
    stream_length=60#This parameter specifies how many values considered by stream
elif prior_hours==48:
    stream_length=120

for k in array_of_of_length:
    if k >stream_length:
        counter=counter+1

print(f'# of samples with lenght { stream_length} == {counter}.')
print(f'Percentage of samples with size grater than {stream_length} : {((counter*100)/len(array_of_of_length))}%.')
print('\n')

means=chartevents.groupby('ITEMID')['VALUENUM'].mean().to_dict()#Mean dictionary of features
means[3348]=1#As 3348 is a categorical feature we cannot use the mean. Instead the mode is used.
means


print('Samples creation')
Values,Timestamps,Masks,y=samplesConstructor(data,features,hadm_ids,stream_length,means,dict_state)
print('Delta tensor creation')
Delta_time=deltaTimeMatricesConstructor(Timestamps,Masks,stream_length)
print('\n')

print(f'Values matrix size  {Values.shape}')
print(f'Masks tensor size  {Masks.shape}')
print(f'Timestamps tensor size  {Timestamps.shape}')
print(f'Delta time tensor size  {Delta_time.shape}')
print(f'Target tensor size  {y.shape}')
print('\n')
# Values.shape,Masks.shape,Timestamps.shape,np.array(y).shape

#Save samples
np.save(f'{path}Values_{prior_hours}_imputed',Values)
np.save(f'{path}Timestamps_{prior_hours}_imputed',Timestamps)
np.save(f'{path}Masks_{prior_hours}_imputed',Masks)
np.save(f'{path}Delta_time_{prior_hours}_imputed',Delta_time)
np.save(f'{path}Targets_{prior_hours}_imputed',y)