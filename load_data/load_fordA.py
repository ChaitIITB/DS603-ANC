from scipy.io.arff import loadarff 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Utils import compute_eps_per_channel
import os

train_path="data/FordA/FordA_TRAIN.arff"
test_path="data/FordA/FordA_TEST.arff"

# https://github.com/mnorayr/ford/blob/master/FordA.ipynb

def read_ariff(path):
    raw_data, meta =loadarff(path)
    cols=[x for x in meta]
    data2d=np.zeros([raw_data.shape[0],len(cols)])
    for i,col in zip(range(len(cols)),cols):
        data2d[:,i]=raw_data[col]
    print(data2d.shape)
    return data2d

data2d=read_ariff(train_path)
test2d=read_ariff(test_path)
print(data2d.shape, test2d.shape)

failed=data2d[data2d[:,-1]==1]
not_failed=data2d[data2d[:,-1]==-1]
print(failed.shape, not_failed.shape)

train2d=np.concatenate((failed[:int(0.8*failed.shape[0])],
                       not_failed[:int(0.8*failed.shape[0])]),
                       axis=0)

val2d=np.concatenate((failed[int(0.8*failed.shape[0]):],
                       not_failed[int(0.8*failed.shape[0]):]),
                       axis=0)

np.random.shuffle(train2d)
np.random.shuffle(val2d)

print(train2d.shape, val2d.shape)


def make3d(data):
    df=data.copy()
    x,y=df.shape
    data3d=np.zeros([x,y-1,2])
    for i in range(x):
        data3d[i,:,0]=df[i][:-1].T
        data3d[i,:,1]=np.full((y-1),df[i][-1])
    return data3d

train3d=make3d(train2d)
val3d  =make3d(val2d)
test3d =make3d(test2d)
print(train3d.shape, val3d.shape, test3d.shape)

def scale_d(data):
    df=data.copy()
    df_scaled=np.zeros(df.shape)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    stder = StandardScaler()
    # scaler = scaler.fit(df[0])
    for i in range(df.shape[0]):
        df_scaled[i,:,0]=stder.fit_transform(
                                         df[i,:,0].reshape((df.shape[1], 1))
                                         ).reshape((df.shape[1]))

        df_scaled[i,:,1]=df[i,:,1]
    return df_scaled

train_scaled=scale_d(train3d)
val_scaled  =scale_d(val3d)
test_scaled =scale_d(test3d)

x_train  =np.expand_dims(train_scaled[:,:,0],axis=2)
y_train  =train_scaled[:,:,1]
y_train_e=np.expand_dims(y_train,axis=2)
y_train_s=y_train[:,0]
y_train_s=np.array([1. if x==1 else 0. for x in y_train[:,0]])


x_val   =np.expand_dims(val_scaled[:,:,0],axis=2)
y_val   =val_scaled[:,:,1]
y_val_e =np.expand_dims(y_val,axis=2)
y_val_s =y_val[:,0]
y_val_s=np.array([1. if x==1 else 0. for x in y_val[:,0]])


x_test  =np.expand_dims(test_scaled[:,:,0],axis=2)
y_test  =test_scaled[:,:,1]
y_test_e=np.expand_dims(y_test,axis=2)
y_test_s=y_test[:,0]
y_test_s=np.array([1. if x==1 else 0. for x in y_test[:,0]])

print(x_train.shape,x_val.shape,x_test.shape)
print(y_train.shape,y_val.shape,y_test.shape)
print(y_train_e.shape,y_val_e.shape,y_test_e.shape)
print(y_train_s.shape,y_val_s.shape,y_test_s.shape)

result = {
    'X_train': x_train,
    'X_test': x_test,
    'y_train': y_train,
    'y_test': y_test,
}
eps_per_channel = compute_eps_per_channel(result['X_train'])
result['eps_per_channel'] = eps_per_channel

os.makedirs('data/FordAProcessed', exist_ok=True)

for key, val in result.items():
    np.save(f'data/FordAProcessed/{key}.npy', val)