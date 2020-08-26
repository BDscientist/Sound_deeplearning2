import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import random
import os
import PYCLASS2_air_Copy1
from torch.utils.data import Dataset, DataLoader
import PYCLASS2_air_test
import CNN_CLASS
import librosa




    
def predict(file_name):
    
    # wav 파일을 딥러닝 모델에 판별하기 위해 전저리 하는 코드 ( wav ---> n*n 배열로 바뀜 )
    
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)       
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    
    s_data = ext_features
    train_data2 = s_data
    x=np.array([0,0,0,0,0,0,0])
    train_data = np.zeros((1,200))
    train_data =  np.concatenate((train_data2,x),axis=None)
    train_data = train_data.reshape(1,1,10,20)
    dtype = torch.FloatTensor
    train_data  = torch.as_tensor(train_data).type(dtype)

    

    
    return train_data



if __name__ == "__main__":
    
     # 학습시켰던 모델 로딩
     
    save_path = '/home/libedev/mute/mute-hero/download/dataset/model2/'
    PATH  = save_path + 'model2.pkl'
    model = CNN_CLASS.CNN2()
    model.load_state_dict(torch.load(PATH))
    
    
    #업로드된 파일이 있는 폴더를 보면서 파일이 있으면 위 방식처럼 진행하여 비행소음인지 판별
    
    file = '/home/libedev/mute/mute-hero/air_save/test/'
    
    for i in os.listdir(str(file)):
        
        if os.path.isfile(str(file)+str(i)):
            file_name = str(file)+str(i)
            
    
    result=predict(file_name)
    out = model(result)
    _, predicted = torch.max(out.data, 1)
    
    
    if predicted[0] == 1:
        
        print("not airplane")   # 비행기 이외의 소음으로 판별되면 false반환
    
    elif predicted[0] ==2 :
        
        print("airplane")    # 비행기 소음으로 판별되면 true 반환.