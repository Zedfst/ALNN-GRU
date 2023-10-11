import numpy as np
from tqdm import tqdm
import tensorflow as tf
import keras.backend as K

def samplesConstructor(data,features,hadm_ids,stream_length,means,dict_state):

    y=[]
    axis_0=0
    erro_admin=[]
    idxx=0

    Values,Timestamps,Masks=np.zeros((len(hadm_ids),stream_length,len(features))),np.zeros((len(hadm_ids),stream_length,len(features))),np.zeros((len(hadm_ids),stream_length,len(features)))
    print(Values.shape)
    #tqdm is for the progress bar
    for hadmid in tqdm(hadm_ids):
        axis_0=axis_0+1
        past_dic={}
        past_time={}
        for idx,feature in enumerate(features):
            #If the patient has measurements related to the current feature
            if feature in data[hadmid]:
                intermediate_values=np.array(data[hadmid][feature][0])[:,1][:stream_length]
                #padding
                intermediate_values=np.pad(intermediate_values, (0, stream_length-len(intermediate_values)), 'constant', constant_values=intermediate_values[-1])
                #replace nan value by the empirical mean
                intermediate_values=np.nan_to_num(intermediate_values, nan=np.round(means[feature],2))

                intermediate_timestamps=np.array(data[hadmid][feature][0])[:,0][:stream_length]
                #Padd the timestamp vector with the last timestamp observed
                intermediate_timestamps=np.pad(intermediate_timestamps, (0, stream_length-len(intermediate_timestamps)), 'constant', constant_values=intermediate_timestamps[-1])

                intermediate_masks=np.array(data[hadmid][feature][0])[:,2][:stream_length]
                intermediate_masks=np.pad(intermediate_masks, (0, stream_length-len(intermediate_masks)), 'constant',constant_values=0)

                Values[idxx,:len(intermediate_values),idx]=intermediate_values
                Timestamps[idxx,:len(intermediate_timestamps),idx]=intermediate_timestamps
                Masks[idxx,:len(intermediate_masks),idx]=intermediate_masks
            else:
                #If the patient has no measurements related to the current feature, we use the empirical mean
                intermediate_values=np.array([np.round(means[feature],2)]*stream_length)
                Values[idxx,:len(intermediate_values),idx]=intermediate_values
        y.append(dict_state[hadmid])#Save patient's state
        idxx=idxx+1
            
    return Values,Timestamps,Masks,np.array(y)


def deltaTimeMatricesConstructor(Time_,M_,stream_length):
    Delta_time=np.zeros_like(Time_)
    for k in tqdm(range(Time_.shape[0])):
        for i in range(12):
            for j in range(stream_length):
                if(j!=0):
                    if(M_[k][j,i]==1):
                        Delta_time[k][j,i]=Time_[k][j,i]-Time_[k][(j-1),i]
                    elif(M_[k][j,i]==0):
                        Delta_time[k][j,i]=Time_[k][j,i]-Time_[k][(j-1),i]+Delta_time[k][(j-1),i]
    return Delta_time 

def binaryFocalLoss(gamma=2.1, alpha=.10):
    def binary_focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss
    return binary_focal_loss_fixed

def piorHoursData(hour,dataframe):
    hour_=hour//24
    new_dataframe=dataframe.loc[dataframe.LOS>=hour_,:]
    return new_dataframe

def extractPhysiologicalMeasurement(list_hadm_id,dataframe):
    new_dataframe=dataframe.loc[dataframe.HADM_ID.isin(list_hadm_id)]
    return new_dataframe