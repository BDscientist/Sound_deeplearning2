import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sound_data = np.load('/home/libedev/mute/mute-hero/air_save/AIR_Final_DATASET_0.npz')
X_data = sound_data['X']
y_data = sound_data['y']
groups = sound_data['groups']

print(groups[groups > 0])


gss = GroupShuffleSplit(n_splits=1,test_size=0.1)

for train_idx ,test_idx in gss.split(X_data , y_data, groups = groups):
    X_train = X_data[train_idx]
    y_train = y_data[train_idx]
    groups_train = groups[train_idx]
    
    X_test = X_data[test_idx]
    y_test = y_data[test_idx]
    groups_test = groups[test_idx]
    
    print("X_train.shape >> ",X_train.shape ,"\n\n", "X_test.shape >>", X_test.shape)
    
np.savez('/home/libedev/mute/mute-hero/air_save/total_train_air_dataset.npz',X=X_train, y=y_train, groups=groups_train)
np.savez('/home/libedev/mute/mute-hero/air_save/total_test_air_dataset.npz',X=X_test, y=y_test, groups=groups_test)
