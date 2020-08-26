import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from random import *
import time

class sound_cut:
    
    def __init__(self,base,base2):
        
        self.base = base
        
        self.base2 = base2
        
    def sound_process(self):
        
        a= []
        
        for filename in os.listdir(str(self.base)):
            
            a.append(filename)
            
        
        for i in a:
            print(i)
            
            count =0
            y,sr = librosa.load(self.base+i)
            
            D = librosa.amplitude_to_db(librosa.stft(y[:]),ref=np.max)
            
            max_y = np.where(y == max(y))
            max_y = max_y[0]
            start_time = (max_y -(sr*5)) 
            end_time  = (max_y +(sr*1))
            
            y2 = y[int(start_time):int(end_time)]
            print(y2)
            a = np.where(y == max_y)
            
            #time = round(y/a)
            
            librosa.output.write_wav(str(self.base2)+str(i), y2, sr) 



            
            
if __name__ == '__main__':

    a =sound_cut('/home/libedev/mute/mute-hero/air_save/PI2/','/home/libedev/mute/mute-hero/air_save/airplane(4s)/')
    a.sound_process()
    
    #b =sound_cut('/home/libedev/mute/mute-hero/air_save/PI2/','/home/libedev/mute/mute-hero/air_save/preprocessing2/')
    #b.sound_process()
       


