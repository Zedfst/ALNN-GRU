import numpy as np
import tensorflow as tf
import argparse
from utils import binaryFocalLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score,roc_curve,auc,f1_score,confusion_matrix
import time as tm
from model import ALNN_GRU
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--prior-hours', type=int, default=24)
parser.add_argument('--time-ref-parameter', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=2000)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

prior_hours=args.prior_hours
if prior_hours==48:
    stream_length=120
elif prior_hours==24:
    stream_length=60
else:
    raise Exception(f'The prior_hours parameter is either 24 or 48.')


path="volume/"

M_=np.load(f'{path}Masks_{prior_hours}_imputed.npy')
Time_=np.load(f'{path}Timestamps_{prior_hours}_imputed.npy')
X_=np.load(f'{path}Values_{prior_hours}_imputed.npy')
y=np.load(f'{path}Targets_{prior_hours}_imputed.npy')
Delta_time=np.load(f'{path}Delta_time_{prior_hours}_imputed.npy')

print(f'Values matrix size  {X_.shape}')
print(f'Masks matrix size  {M_.shape}')
print(f'Timestamps matrix size  {Time_.shape}')
print(f'Delta time matrix size  {Delta_time.shape}')
print(f'Target matrix size  {y.shape}')
print('\n')



time_ref_parameter=args.time_ref_parameter
if(time_ref_parameter not in [1,2,3,4]):
    raise Exception(f'The time-ref-parameter must be in [1,2,3,4].')

print("Set the reference time point vector by assigning a value in [1,2,3,4] to time_ref_parameter.")
if(time_ref_parameter==1):
    print(f"If time_ref_parameter={time_ref_parameter}, Δr=1. and #ref_time_points={prior_hours*time_ref_parameter+1}.")
elif(time_ref_parameter==2):
    print(f"If time_ref_parameter={time_ref_parameter}, Δr=0.5 and #ref_time_points={prior_hours*time_ref_parameter+1}.")
elif(time_ref_parameter==3):
    print(f"If time_ref_parameter={time_ref_parameter}, Δr=0.33 and #ref_time_points={prior_hours*time_ref_parameter+1}.")
elif(time_ref_parameter==4):
    print(f"If time_ref_parameter={time_ref_parameter}, Δr=0.25 and #ref_time_points={prior_hours*time_ref_parameter+1}.")
print('\n')

#Training config
kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=14)
bc=tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3, mode='min')


tf.random.set_seed(14)
start= tm.perf_counter()
MAEs,aucs,aucprs,f1scores=[],[],[],[]
epoch=args.epochs
for train, test in kfold.split(X_,y):
    model=ALNN_GRU(max_time=prior_hours,time_interval=time_ref_parameter,type_distance="abs")
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=bc,optimizer=opt,metrics=["accuracy"])

    model.fit(
        [X_[train],Time_[train],
        M_[train],Delta_time[train]], y[train],
        verbose=1,batch_size=args.batch_size,
        epochs=epoch)
    
    loss_test, accuracy_test = model.evaluate([X_[test],Time_[test],M_[test],Delta_time[test]],y[test],verbose=1,batch_size=100,callbacks=[callback])

    y_probas = model.predict([X_[test],Time_[test],M_[test],Delta_time[test]]).ravel()
    fpr,tpr,thresholds=roc_curve(y[test],y_probas)
    aucs.append(auc(fpr,tpr))
    print('AUC= ',auc(fpr,tpr))

    auprc_ = average_precision_score(y[test], y_probas)
    aucprs.append(auprc_)
    print('AUPRC= ', auprc_)

    #Threshold moving
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    y_probas_f=y_probas
    y_probas_f[y_probas_f>=thresholds[ix]]=1
    y_probas_f[y_probas_f<thresholds[ix]]=0


    f1score=f1_score(y[test], y_probas_f)
    f1scores.append(f1score)
    print(f'F1 score= {f1score}')

    ##Uncomment all statements below to visualize the confusion matrices
    # y_pred = np.where(y_probas>thresholds[ix],1,0)
    # cm = confusion_matrix(y[test], y_pred)
    # ax = plt.subplot()
    # sns.set(font_scale=1.2) #edited as suggested
    # sns.heatmap(cm, cbar=False, annot=True, cmap="Blues", fmt="g",annot_kws={"size": 20});  # annot=True to annotate cells
    # ax.xaxis.set_ticklabels(['Predicted 0', 'Predicted 1']);
    # ax.yaxis.set_ticklabels(['Actual 0', 'Actual 1']);
    # plt.show()
    # Specificity=np.round(cm[0][0]/(cm[0][0]+cm[0][1]),2)
    # Sensitivity=cm[1][1]/(cm[1][0]+cm[1][1])
    # print(f'Specificity-> {Specificity}')
    # print(f'Sensitivity-> {Sensitivity}')

    print('\n')

finish=tm.perf_counter()
print(f"Finished in {round(finish-start,2)},second(s)")
print(f'AUC: mean={np.round(np.mean(np.array(aucs)),3)},std={np.round(np.std(np.array(aucs)),3)}')
print(f'AUPRC: mean={np.round(np.mean(np.array(aucprs)),3)},std={np.round(np.std(np.array(aucprs)),3)}')
print(f'F1-score: mean={np.round(np.mean(np.array(f1scores)),3)},std={np.round(np.std(np.array(f1scores)),3)}')
print('\n')


