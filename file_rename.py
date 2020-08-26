import os
import shutil

base = '/home/libedev/mute/mute-hero/air_save/final_test_sound(car)/'
train = 'new_train/'
test = 'new_test/'
ch_base = '/home/libedev/mute/mute-hero/air_save/FINAL_DATASET/'
count = 0



def rename():
    
    count = 0
        
    for filename in os.listdir(str(base)):
            
        shutil.move(str(base)+str(filename),str(ch_base)+str(count)+'-'+'1'+'-'+'0'+'-'+str(count)+'.wav')
        
        count +=1
        
if __name__ =="__main__":
    
    rename()